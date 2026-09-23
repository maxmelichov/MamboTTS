import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { ChevronRight, Download, Pause, Play } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button, Card } from "./ui";
import { sharedAudioContext } from "../lib/audioContext";
import {
  type ExportQuality,
  exportQualityOption,
  exportQualityOptions,
  readExportQuality,
  storeExportQuality,
  withExtension,
} from "../lib/exportQuality";
import { motion } from "framer-motion";

const END_EPSILON = 0.02;

/**
 * A single Web Audio transport that owns the whole timeline. Streamed WAV
 * chunks and a finalized WAV are both represented as an ordered list of
 * decoded buffers, so play / pause / seek / scrub behave identically while the
 * stream is still arriving and after it has finished. Using one engine avoids
 * the handoff bugs that came from mixing an HTML <audio> element with Web Audio.
 *
 * The timeline belongs to a session (`sessionKey`, one per take). Within a
 * session, sources past the ones already decoded extend the timeline, and a
 * list that shrinks is ignored: the caller releases the streamed chunks once
 * they are decoded and hands over the finished file, which this player already
 * holds as buffers. A new session starts over.
 */
export function WaveformPlayer({
  sources,
  sessionKey,
  onSourcesDecoded,
  downloadPath,
  filename,
  complete = false,
  autoPlayOnce = false,
  onAutoPlayConsumed,
}: {
  sources: string[];
  sessionKey: number;
  /** Called with the number of sources decoded so far in this session. */
  onSourcesDecoded?: (count: number) => void;
  downloadPath: string;
  filename: string;
  complete?: boolean;
  autoPlayOnce?: boolean;
  onAutoPlayConsumed?: () => void;
}) {
  const waveformRef = useRef<HTMLDivElement>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const buffersRef = useRef<AudioBuffer[]>([]);
  const startsRef = useRef<number[]>([]);
  const totalRef = useRef(0);
  const decodedCountRef = useRef(0);
  const sessionKeyRef = useRef<number | null>(null);
  const onSourcesDecodedRef = useRef(onSourcesDecoded);
  // The decode loop reads the list through a ref, so a chunk that arrives
  // while it is running is picked up by the same pass instead of waiting for
  // the next one.
  const sourcesRef = useRef(sources);
  const decodingRef = useRef(false);
  const sessionRef = useRef<{ baseCtxTime: number; baseOffset: number } | null>(null);
  const nodesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const rafRef = useRef(0);
  const playingRef = useRef(false);
  const pausedOffsetRef = useRef(0);
  const completeRef = useRef(complete);
  const autoPlayPendingRef = useRef(autoPlayOnce);
  const onAutoPlayConsumedRef = useRef(onAutoPlayConsumed);
  const dragResumeRef = useRef(false);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [progress, setProgress] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [savedPath, setSavedPath] = useState("");
  const [downloadError, setDownloadError] = useState("");
  const [exportQuality, setExportQuality] = useState<ExportQuality>(readExportQuality);
  const [playbackError, setPlaybackError] = useState("");

  completeRef.current = complete;
  onAutoPlayConsumedRef.current = onAutoPlayConsumed;
  onSourcesDecodedRef.current = onSourcesDecoded;
  sourcesRef.current = sources;
  if (autoPlayOnce) autoPlayPendingRef.current = true;

  const getContext = useCallback(() => {
    contextRef.current = sharedAudioContext();
    return contextRef.current;
  }, []);

  const stopNodes = useCallback(() => {
    for (const node of nodesRef.current) {
      node.onended = null;
      try {
        node.stop();
      } catch {
        // A node that never started throws on stop(); ignore.
      }
    }
    nodesRef.current.clear();
  }, []);

  const computeCurrentTime = useCallback(() => {
    const session = sessionRef.current;
    const context = contextRef.current;
    if (!session || !context) return pausedOffsetRef.current;
    return session.baseOffset + (context.currentTime - session.baseCtxTime);
  }, []);

  const scheduleBuffer = useCallback((index: number) => {
    const session = sessionRef.current;
    const context = contextRef.current;
    if (!session || !context) return;
    const buffer = buffersRef.current[index];
    const segmentStart = startsRef.current[index];
    if (!buffer) return;

    const when = session.baseCtxTime + (segmentStart - session.baseOffset);
    let scheduleAt = when;
    let offsetIntoBuffer = 0;
    if (when < context.currentTime) {
      offsetIntoBuffer = context.currentTime - when;
      scheduleAt = context.currentTime;
    }
    if (offsetIntoBuffer >= buffer.duration) return;

    const node = context.createBufferSource();
    node.buffer = buffer;
    node.connect(context.destination);
    node.onended = () => {
      nodesRef.current.delete(node);
    };
    node.start(scheduleAt, offsetIntoBuffer);
    nodesRef.current.add(node);
  }, []);

  const tick = useCallback(() => {
    if (!playingRef.current) return;
    const total = totalRef.current;
    const time = Math.min(Math.max(computeCurrentTime(), 0), total || 0);
    setCurrentTime(time);
    setProgress(total > 0 ? time / total : 0);

    const reachedEnd =
      completeRef.current
      && !decodingRef.current
      && total > 0
      && time >= total - END_EPSILON
      && nodesRef.current.size === 0;

    if (reachedEnd) {
      playingRef.current = false;
      sessionRef.current = null;
      pausedOffsetRef.current = 0;
      setIsPlaying(false);
      setCurrentTime(total);
      setProgress(1);
      return;
    }
    rafRef.current = requestAnimationFrame(tick);
  }, [computeCurrentTime]);

  const startTicking = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(tick);
  }, [tick]);

  const play = useCallback(async (fromOffset?: number) => {
    const context = getContext();
    setPlaybackError("");
    try {
      await context.resume();
    } catch {
      // Resume can reject if the context was closed mid-teardown; ignore.
    }
    stopNodes();

    let offset = fromOffset ?? pausedOffsetRef.current;
    if (offset >= totalRef.current - END_EPSILON) offset = 0;

    sessionRef.current = { baseCtxTime: context.currentTime, baseOffset: offset };
    playingRef.current = true;
    setIsPlaying(true);

    for (let i = 0; i < buffersRef.current.length; i += 1) {
      const segmentEnd = startsRef.current[i] + buffersRef.current[i].duration;
      if (segmentEnd > offset + END_EPSILON) scheduleBuffer(i);
    }
    startTicking();
  }, [getContext, scheduleBuffer, startTicking, stopNodes]);

  const pause = useCallback(() => {
    pausedOffsetRef.current = Math.min(Math.max(computeCurrentTime(), 0), totalRef.current || 0);
    playingRef.current = false;
    sessionRef.current = null;
    stopNodes();
    cancelAnimationFrame(rafRef.current);
    setIsPlaying(false);
  }, [computeCurrentTime, stopNodes]);

  const decodeSources = useCallback(async () => {
    if (decodingRef.current) return;
    decodingRef.current = true;
    try {
      const context = getContext();
      const session = sessionKeyRef.current;
      while (decodedCountRef.current < sourcesRef.current.length) {
        const index = decodedCountRef.current;
        const url = sourcesRef.current[index];
        const response = await fetch(url);
        if (!response.ok) throw new Error(`could not load audio (${response.status})`);
        const data = await response.arrayBuffer();
        const buffer = await context.decodeAudioData(data);
        // A new take started while this one was decoding.
        if (sessionKeyRef.current !== session) return;

        const segmentStart = totalRef.current;
        buffersRef.current.push(buffer);
        startsRef.current.push(segmentStart);
        totalRef.current = segmentStart + buffer.duration;
        decodedCountRef.current = index + 1;
        setDuration(totalRef.current);
        onSourcesDecodedRef.current?.(decodedCountRef.current);

        if (playingRef.current && sessionRef.current) scheduleBuffer(index);

        if (index === 0 && autoPlayPendingRef.current) {
          autoPlayPendingRef.current = false;
          onAutoPlayConsumedRef.current?.();
          void play(0);
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setPlaybackError(`Could not load audio: ${message}`);
    } finally {
      decodingRef.current = false;
    }
  }, [getContext, play, scheduleBuffer]);

  // Decode incoming sources. A growing list (streaming) extends the timeline; a
  // new session (a new take) resets everything first.
  useEffect(() => {
    if (sessionKeyRef.current !== sessionKey) {
      sessionKeyRef.current = sessionKey;
      stopNodes();
      cancelAnimationFrame(rafRef.current);
      buffersRef.current = [];
      startsRef.current = [];
      totalRef.current = 0;
      decodedCountRef.current = 0;
      sessionRef.current = null;
      playingRef.current = false;
      pausedOffsetRef.current = 0;
      setIsPlaying(false);
      setCurrentTime(0);
      setProgress(0);
      setDuration(0);
    }

    if (sources.length > decodedCountRef.current) void decodeSources();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionKey, sources.join("|")]);

  useEffect(() => () => {
    playingRef.current = false;
    cancelAnimationFrame(rafRef.current);
    stopNodes();
    // The context is shared and was unlocked by a click; keep it for the next take.
    contextRef.current = null;
  }, [stopNodes]);

  const hasAudio = duration > 0 || sources.length > 0;

  const togglePlay = () => {
    if (!hasAudio) return;
    if (playingRef.current) {
      pause();
    } else {
      void play();
    }
  };

  const seekToClientX = useCallback((clientX: number) => {
    const total = totalRef.current;
    if (!waveformRef.current || total <= 0) return;
    const rect = waveformRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const nextProgress = rect.width > 0 ? x / rect.width : 0;
    const nextTime = nextProgress * total;
    pausedOffsetRef.current = nextTime;
    setCurrentTime(nextTime);
    setProgress(nextProgress);
    return nextTime;
  }, []);

  const handlePointerDown = (e: React.PointerEvent) => {
    if (!hasAudio) return;
    e.preventDefault();
    dragResumeRef.current = playingRef.current;
    if (playingRef.current) pause();
    setIsDragging(true);
    seekToClientX(e.clientX);
    waveformRef.current?.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isDragging) return;
    e.preventDefault();
    seekToClientX(e.clientX);
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    if (!isDragging) return;
    const target = seekToClientX(e.clientX);
    setIsDragging(false);
    if (waveformRef.current?.hasPointerCapture(e.pointerId)) {
      waveformRef.current.releasePointerCapture(e.pointerId);
    }
    if (dragResumeRef.current) {
      dragResumeRef.current = false;
      void play(target);
    }
  };

  const downloadAudio = async () => {
    setDownloading(true);
    setDownloadError("");
    try {
      // The take on disk is always WAV; the file saved from it takes the
      // format of the chosen size, and its name has to say so.
      const option = exportQualityOption(exportQuality);
      const destinationPath = await save({
        defaultPath: withExtension(filename, option.extension),
        filters: [{ name: option.extension === "mp3" ? "MP3 audio" : "WAV audio", extensions: [option.extension] }],
      });
      if (!destinationPath) return;

      await invoke("export_audio_file", {
        sourcePath: downloadPath,
        destinationPath,
        quality: exportQuality,
      });
      setSavedPath(destinationPath);
    } catch (err) {
      setDownloadError(String(err));
    } finally {
      setDownloading(false);
    }
  };

  const revealSavedAudio = async () => {
    if (!savedPath) return;
    setDownloadError("");
    try {
      await invoke("reveal_path", { path: savedPath });
    } catch (err) {
      setDownloadError(String(err));
    }
  };

  const formatTime = (seconds: number) => {
    if (isNaN(seconds) || !isFinite(seconds)) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const bars = useMemo(() => Array.from({ length: 50 }, () => 20 + Math.random() * 60), []);

  return (
    <Card className="group flex flex-col gap-5 border-none bg-white p-5 shadow-2xl transition-all hover:shadow-3xl sm:flex-row sm:items-center sm:gap-8">
      <div className="flex items-center gap-4">
        <button
          type="button"
          onClick={togglePlay}
          disabled={!hasAudio}
          className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-primary text-white transition-all hover:scale-105 active:scale-95 shadow-lg shadow-primary/10 cursor-pointer disabled:cursor-not-allowed disabled:opacity-40"
        >
          {isPlaying ? <Pause className="h-5 w-5 fill-current" /> : <Play className="h-5 w-5 fill-current ml-0.5" />}
        </button>
        <div className="min-w-0">
          <p className="text-sm font-bold tracking-tight text-primary">Preview</p>
          <p className="truncate text-[10px] font-bold uppercase tracking-widest text-secondary opacity-30">{filename}</p>
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-2">
        <div
          ref={waveformRef}
          className="relative flex h-10 cursor-pointer items-center gap-[2px] touch-none"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
        >
          {bars.map((height, index) => (
            <motion.div
              key={index}
              initial={false}
              animate={{
                backgroundColor: index / bars.length < progress ? "var(--color-primary)" : "var(--color-border)",
                opacity: index / bars.length < progress ? 1 : 0.8,
              }}
              transition={{ duration: isDragging ? 0 : 0.12 }}
              className="flex-1 rounded-full"
              style={{
                height: `${height}%`,
              }}
            />
          ))}
        </div>
        <div className="flex justify-between font-mono text-[9px] font-bold uppercase tracking-widest text-secondary opacity-30">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      {downloadPath && (
      <div className="flex shrink-0 items-center gap-2 border-l border-border/10 pl-6">
        <label className="relative block">
          <span className="sr-only">Saved file size</span>
          <select
            value={exportQuality}
            onChange={(event) => {
              const next = event.currentTarget.value as ExportQuality;
              storeExportQuality(next);
              setExportQuality(next);
            }}
            disabled={downloading}
            title={`Saved as ${exportQualityOption(exportQuality).format}, about ${exportQualityOption(exportQuality).perMinute.replace("~", "")}`}
            className="h-10 cursor-pointer appearance-none rounded-lg border border-border/40 bg-white pl-3 pr-8 text-xs font-bold tracking-tight text-primary outline-none transition-all hover:border-border focus:border-primary focus:ring-4 focus:ring-primary/5 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {exportQualityOptions.map((option) => (
              <option key={option.id} value={option.id} title={`${option.format}, ${option.perMinute}`}>
                {option.label} · {option.extension.toUpperCase()}
              </option>
            ))}
          </select>
          <ChevronRight className="pointer-events-none absolute right-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 rotate-90 opacity-30" />
        </label>
        <Button
          variant="outline"
          onClick={downloadAudio}
          disabled={downloading}
          className="h-10 w-10 p-0 rounded-full transition-transform hover:scale-110 active:scale-90"
          title={`Save audio (${exportQualityOption(exportQuality).format})`}
        >
          <Download className="h-4 w-4" />
        </Button>
      </div>
      )}

      {(savedPath || downloadError || playbackError) && (
        <div className="basis-full rounded-xl border border-border/40 bg-background/40 p-3 text-xs">
          {playbackError ? (
            <p className="font-medium text-red-800">{playbackError}</p>
          ) : downloadError ? (
            <p className="font-medium text-red-800">{downloadError}</p>
          ) : (
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="min-w-0 truncate font-medium text-secondary">Saved to {savedPath}</p>
              <Button variant="outline" onClick={revealSavedAudio} className="h-8 shrink-0 px-3 text-xs">
                Show in Finder
              </Button>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
