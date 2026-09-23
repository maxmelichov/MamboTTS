import { Channel, convertFileSrc, invoke } from "@tauri-apps/api/core";
import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import { useNavigate } from "react-router-dom";
import type { EditorInputSource, ModelBundle, RunnerInfo, StudioState } from "../../lib/types";
import { AppFrame } from "../../components/AppFrame";
import { CreateStatus } from "../../components/CreateStatus";
import { ErrorBlock } from "../../components/ui";
import { WaveformPlayer } from "../../components/WaveformPlayer";
import { StudioHeader } from "../../components/WorkspaceHeader";
import { EditorCard } from "./EditorCard";
import { VoiceSettings } from "./VoiceSettings";

const PRONUNCIATION_KEY = "editor-pronunciation";
/** The two per-layer switches this one setting replaced. */
const RETIRED_LAYER_KEYS = ["editor-layer-diacritics", "editor-layer-phonemes"];

/**
 * The pronunciation layers (Niqqud and IPA) stay off until someone turns them
 * on, then stay as they were left. Someone who had either of the old separate
 * switches on finds the combined one on.
 */
function readPronunciationEnabled(): boolean {
  try {
    const stored = localStorage.getItem(PRONUNCIATION_KEY);
    if (stored !== null) return stored === "on";
    const migrated = RETIRED_LAYER_KEYS.some((key) => localStorage.getItem(key) === "on");
    storePronunciationEnabled(migrated);
    return migrated;
  } catch {
    return false;
  }
}

function storePronunciationEnabled(enabled: boolean) {
  try {
    localStorage.setItem(PRONUNCIATION_KEY, enabled ? "on" : "off");
    RETIRED_LAYER_KEYS.forEach((key) => localStorage.removeItem(key));
  } catch {
    // Storage can be unavailable; the switch still applies for this session.
  }
}

type PageProps = {
  bundle: ModelBundle | null;
  setBundle: (bundle: ModelBundle) => void;
};

/** What the `synthesize` command resolves with. */
type SpeechResult = {
  path: string;
  /** Chunks sent over the channel; playback is complete once all of them arrived. */
  chunks: number;
};

/** The rejection text of a synthesis stopped with `cancel_synthesis`. */
const SYNTHESIS_CANCELLED = "synthesis cancelled";

/** How long to wait for chunk messages still in flight after the command returns. */
const CHUNK_DRAIN_TIMEOUT_MS = 5000;

type HomePageProps = PageProps & {
  studio: StudioState;
  setStudio: Dispatch<SetStateAction<StudioState>>;
};

export function HomePage({ bundle, setBundle, studio, setStudio }: HomePageProps) {
  const navigate = useNavigate();
  const { text, phonemes, diacritics, diacriticsSource, languages, language, blueVoice, blueVoiceIds, speaker, targetSpeaker, speed, audioPath, streamChunkUrls, generation, audioAutoplayPending, step, status, busy, error } = studio;
  const loadingLanguagesRef = useRef(false);
  const [pronunciationEnabled, setPronunciationState] = useState(readPronunciationEnabled);
  function setPronunciationEnabled(enabled: boolean) {
    storePronunciationEnabled(enabled);
    setPronunciationState(enabled);
  }
  const [stopping, setStopping] = useState(false);
  // The take in flight: its cancel id, whether Stop was pressed, and whether
  // the synthesize command has been sent (only then is there anything to cancel).
  const synthesisIdRef = useRef("");
  const stopRequestedRef = useRef(false);
  const synthesizeSentRef = useRef(false);
  const activeGenerationRef = useRef(generation);
  // How many of the current take's sources the mounted player has decoded.
  const decodedCountRef = useRef(0);

  const audioSrc = useMemo(() => (audioPath ? convertFileSrc(audioPath) : ""), [audioPath]);
  // Prefer the streamed chunks (available first, identical audio) and fall back
  // to the finalized WAV once the chunks are released or were never streamed.
  const playerSources = useMemo(
    () => (streamChunkUrls.length ? streamChunkUrls : audioSrc ? [audioSrc] : []),
    [streamChunkUrls, audioSrc],
  );
  const updateStudio = (patch: Partial<StudioState>) => setStudio((current) => ({ ...current, ...patch }));
  const isHebrew = language === "he" || (language === "auto" && /[֐-׿]/.test(text));
  // Niqqud and IPA are opt-in layers behind one switch. Switched off, they
  // keep their content for when they come back, but are never spoken.
  const diacriticsEnabled = pronunciationEnabled && isHebrew;
  const phonemesEnabled = pronunciationEnabled;
  // Vocalized text only counts while it still matches the plain text it came
  // from; after the text is edited it is kept for reference but not spoken.
  const diacriticsStale = Boolean(diacritics) && diacriticsSource !== text;
  const vocalized = diacriticsEnabled && !diacriticsStale && diacritics.trim() ? diacritics : "";
  // Generate sends the most refined enabled layer present: edited IPA, then
  // vocalized Hebrew (RenikudPlus honors typed niqqud), then the plain text.
  const inputSource: EditorInputSource = phonemesEnabled && phonemes.trim() ? "phonemes" : vocalized ? "diacritics" : "text";
  const synthesisInput = inputSource === "phonemes" ? phonemes : inputSource === "diacritics" ? vocalized : text;

  // Streamed chunks live in memory as blob URLs, never on disk. Once the
  // finished WAV exists and the player has decoded every chunk, the chunks are
  // released: the player keeps its decoded audio, and a later remount plays
  // the finished file instead.
  const releaseChunksIfDone = useCallback(() => {
    setStudio((current) => {
      if (!current.audioPath || current.streamChunkUrls.length === 0) return current;
      if (decodedCountRef.current < current.streamChunkUrls.length) return current;
      current.streamChunkUrls.forEach((url) => URL.revokeObjectURL(url));
      return { ...current, streamChunkUrls: [] };
    });
  }, [setStudio]);

  useEffect(() => {
    releaseChunksIfDone();
  }, [audioPath, releaseChunksIfDone]);

  const handleSourcesDecoded = useCallback((count: number) => {
    decodedCountRef.current = count;
    releaseChunksIfDone();
  }, [releaseChunksIfDone]);

  useEffect(() => {
    if (!bundle?.installed || busy || loadingLanguagesRef.current) return;
    if (languages.length > 1 && blueVoiceIds.length > 0) return;
    const currentBundle = bundle;
    loadingLanguagesRef.current = true;

    async function loadLanguages() {
      try {
        await invoke<RunnerInfo>("start_runner");
        await invoke("load_model", {
          request: {
            runtime: currentBundle.runtime,
            model_path: currentBundle.model_path,
            renikud_path: currentBundle.renikud_path,
            speaker,
            target_speaker: targetSpeaker,
          },
        });
        const supportedLanguages = await invoke<string[]>("get_languages");
        const nextStudio: Partial<StudioState> = {};
        if (supportedLanguages.length) nextStudio.languages = supportedLanguages;
        try {
          const voiceIds = await invoke<string[]>("get_voices");
          if (voiceIds.length) {
            nextStudio.blueVoiceIds = voiceIds;
            // The loaded bundle owns the catalog. A voice held over from an
            // older bundle has to give way, or synthesis fails on a name the
            // engine has never heard of.
            if (!voiceIds.includes(blueVoice)) nextStudio.blueVoice = voiceIds[0];
          }
        } catch {
          // Voice IDs improve the picker, but language loading should still succeed without them.
        }
        if (Object.keys(nextStudio).length) updateStudio(nextStudio);
      } catch {
        updateStudio({ languages: ["auto"] });
      } finally {
        loadingLanguagesRef.current = false;
      }
    }

    loadLanguages();
  }, [bundle, busy, languages.length, blueVoiceIds.length]);

  async function ensureModelLoaded() {
    const current = bundle ?? (await invoke<ModelBundle>("get_model_bundle_for_runtime", { runtime: "blue" }));
    if (!current.installed) throw new Error("Install the Blue model before using phonemes.");
    await invoke<RunnerInfo>("start_runner");
    await invoke("load_model", {
      request: {
        runtime: current.runtime,
        model_path: current.model_path,
        renikud_path: current.renikud_path,
        speaker,
        target_speaker: targetSpeaker,
      },
    });
  }

  /**
   * IPA derives from the vocalized Hebrew when there is some, so the niqqud
   * edits carry through. `freshVocalized` is niqqud produced in the same step,
   * which this render's state has not caught up with yet.
   */
  async function convertToPhonemes(freshVocalized?: string) {
    if (!text.trim()) return;
    await ensureModelLoaded();
    const output = await invoke<string>("phonemize", {
      request: { input: freshVocalized || vocalized || text, language },
    });
    updateStudio({ phonemes: output });
  }

  /** Vocalizes the text and resolves with the vocalized version. */
  async function addDiacritics(): Promise<string> {
    if (!text.trim()) return "";
    await ensureModelLoaded();
    const source = text;
    const output = await invoke<string>("diacritize", { request: { input: source, language: "he" } });
    updateStudio({ diacritics: output, diacriticsSource: source, phonemes: "" });
    return output;
  }

  async function createVoice() {
    const current = bundle ?? (await invoke<ModelBundle>("get_model_bundle_for_runtime", { runtime: "blue" }));
    setBundle(current);
    if (!current.installed) {
      navigate("/onboard", { replace: true });
      return;
    }
    const input = synthesisInput;
    const inputIsPhonemes = inputSource === "phonemes";
    if (!input.trim()) {
      updateStudio({ status: "Input text required." });
      return;
    }

    const takeGeneration = generation + 1;
    const synthesisId = crypto.randomUUID();
    activeGenerationRef.current = takeGeneration;
    synthesisIdRef.current = synthesisId;
    stopRequestedRef.current = false;
    synthesizeSentRef.current = false;
    decodedCountRef.current = 0;
    setStopping(false);
    setStudio((current) => {
      current.streamChunkUrls.forEach((url) => URL.revokeObjectURL(url));
      return {
        ...current,
        busy: true,
        error: "",
        audioPath: "",
        streamChunkUrls: [],
        generation: takeGeneration,
        audioAutoplayPending: false,
      };
    });
    const throwIfStopped = () => {
      if (stopRequestedRef.current) throw new Error(SYNTHESIS_CANCELLED);
    };
    try {
      updateStudio({ step: "starting", status: "Initializing Engine..." });
      await invoke<RunnerInfo>("start_runner");
      throwIfStopped();

      updateStudio({ step: "loading", status: "Loading models..." });
      await invoke("load_model", {
        request: {
          runtime: current.runtime,
          model_path: current.model_path,
          renikud_path: current.renikud_path,
          speaker,
          target_speaker: targetSpeaker,
        },
      });

      throwIfStopped();
      const supportedLanguages = await invoke<string[]>("get_languages");
      const nextStudio: Partial<StudioState> = { languages: supportedLanguages.length ? supportedLanguages : ["auto"] };
      try {
        const voiceIds = await invoke<string[]>("get_voices");
        if (voiceIds.length) {
          nextStudio.blueVoiceIds = voiceIds;
          if (!voiceIds.includes(blueVoice)) nextStudio.blueVoice = voiceIds[0];
        }
      } catch {
        // Keep synthesis usable even if voice listing is unavailable.
      }
      updateStudio(nextStudio);
      // State from updateStudio is not readable until the next render, so the
      // reconciled voice has to come from nextStudio the same way the language
      // below does.
      const synthesisVoice = nextStudio.blueVoice ?? blueVoice;
      const selectedLanguage = supportedLanguages.includes(language) ? language : "auto";
      if (selectedLanguage !== language) updateStudio({ language: "auto" });
      const synthesisLanguage = inputIsPhonemes && selectedLanguage === "auto"
        ? (/[֐-׿]/.test(text) ? "he" : "en")
        : selectedLanguage;

      throwIfStopped();
      updateStudio({ step: "creating", status: "Generating audio..." });

      // Each chunk arrives as raw WAV bytes while inference runs. Messages can
      // still be in flight when the command resolves, so the take is only
      // complete once as many chunks arrived as the command reports.
      let received = 0;
      let expected = Number.POSITIVE_INFINITY;
      let markAllReceived = () => {};
      const allReceived = new Promise<void>((resolve) => {
        markAllReceived = resolve;
      });
      const onChunk = new Channel<ArrayBuffer>();
      onChunk.onmessage = (data) => {
        if (activeGenerationRef.current !== takeGeneration) return;
        const url = URL.createObjectURL(new Blob([data], { type: "audio/wav" }));
        received += 1;
        setStudio((current) => ({
          ...current,
          streamChunkUrls: [...current.streamChunkUrls, url],
          audioAutoplayPending: true,
          status: "Playing generated audio...",
        }));
        if (received >= expected) markAllReceived();
      };

      synthesizeSentRef.current = true;
      const result = await invoke<SpeechResult>("synthesize", {
        request: {
          input,
          voice: synthesisVoice || undefined,
          language: synthesisLanguage,
          input_is_phonemes: inputIsPhonemes,
          speed,
          synthesis_id: synthesisId,
        },
        onChunk,
      });
      expected = result.chunks;
      if (received >= expected) markAllReceived();
      await Promise.race([
        allReceived,
        new Promise((resolve) => setTimeout(resolve, CHUNK_DRAIN_TIMEOUT_MS)),
      ]);
      // The chunk playlist stays until the player has decoded all of it;
      // clearing it here would reset the timeline and cut off playback.
      updateStudio({
        audioPath: result.path,
        audioAutoplayPending: false,
        step: "done",
        status: "Generation complete.",
      });
    } catch (err) {
      if (stopRequestedRef.current || String(err).includes(SYNTHESIS_CANCELLED)) {
        updateStudio({ step: "idle", error: "", status: "Generation stopped." });
      } else {
        updateStudio({ step: "idle", error: String(err), status: "Generation failed." });
      }
    } finally {
      synthesisIdRef.current = "";
      synthesizeSentRef.current = false;
      setStopping(false);
      updateStudio({ busy: false });
    }
  }

  async function stopVoice() {
    if (!busy || stopRequestedRef.current) return;
    stopRequestedRef.current = true;
    setStopping(true);
    updateStudio({ status: "Stopping..." });
    // Before the synthesize command is sent, the steps above notice the flag
    // on their own. After it, the command has to be told.
    if (synthesizeSentRef.current && synthesisIdRef.current) {
      try {
        await invoke("cancel_synthesis", { synthesisId: synthesisIdRef.current });
      } catch {
        // The synthesis may have finished in the meantime; nothing to stop.
      }
    }
  }

  return (
    <AppFrame bundle={bundle}>
      <div className="w-full max-w-[1200px]">
        <StudioHeader bundle={bundle} />
        <div className="grid gap-12 mt-4 lg:grid-cols-[1fr_360px]">
          <div className="space-y-6">
            <EditorCard
              busy={busy}
              text={text}
              setText={(nextText) => updateStudio({ text: nextText, phonemes: "" })}
              language={language}
              isHebrew={isHebrew}
              pronunciationEnabled={pronunciationEnabled}
              setPronunciationEnabled={setPronunciationEnabled}
              inputSource={inputSource}
              synthesisInput={synthesisInput}
              diacritics={diacritics}
              diacriticsStale={diacriticsStale}
              setDiacritics={(nextDiacritics) => updateStudio({ diacritics: nextDiacritics, phonemes: "" })}
              clearDiacritics={() => updateStudio({ diacritics: "", diacriticsSource: "", phonemes: "" })}
              addDiacritics={addDiacritics}
              phonemes={phonemes}
              setPhonemes={(nextPhonemes) => updateStudio({ phonemes: nextPhonemes })}
              convertToPhonemes={convertToPhonemes}
              createVoice={createVoice}
              stopVoice={stopVoice}
              stopping={stopping}
              blueVoice={blueVoice}
              blueVoiceIds={blueVoiceIds}
              speed={speed}
              setBlueVoice={(nextVoice) => updateStudio({ blueVoice: nextVoice })}
              setSpeed={(nextSpeed) => {
                localStorage.setItem("speech-speed", String(nextSpeed));
                updateStudio({ speed: nextSpeed });
              }}
            />

            <AnimatePresence>
              {playerSources.length > 0 && (
                <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}>
                  <WaveformPlayer
                    sources={playerSources}
                    sessionKey={generation}
                    onSourcesDecoded={handleSourcesDecoded}
                    downloadPath={audioPath}
                    complete={Boolean(audioPath) || !busy}
                    filename={(audioPath || "generated-audio.wav").split(/[\\/]/).pop() || "generated-audio.wav"}
                    autoPlayOnce={audioAutoplayPending}
                    onAutoPlayConsumed={() => updateStudio({ audioAutoplayPending: false })}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <aside className="space-y-6">
            <VoiceSettings
              busy={busy}
              language={language}
              languages={languages}
              speaker={speaker}
              targetSpeaker={targetSpeaker}
              setLanguage={(nextLanguage) => updateStudio({ language: nextLanguage })}
              setSpeaker={(nextSpeaker) => updateStudio({ speaker: nextSpeaker, phonemes: "" })}
              setTargetSpeaker={(nextSpeaker) => updateStudio({ targetSpeaker: nextSpeaker, phonemes: "" })}
            />

            <AnimatePresence>
              {busy && (
                <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.98 }}>
                  <CreateStatus step={step} status={status} />
                </motion.div>
              )}
            </AnimatePresence>
            {error && <ErrorBlock className="mt-0">{error}</ErrorBlock>}
          </aside>
        </div>
      </div>
    </AppFrame>
  );
}
