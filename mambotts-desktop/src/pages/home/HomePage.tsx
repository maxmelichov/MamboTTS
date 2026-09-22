import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useRef } from "react";
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

type PageProps = {
  bundle: ModelBundle | null;
  setBundle: (bundle: ModelBundle) => void;
};

type HomePageProps = PageProps & {
  studio: StudioState;
  setStudio: Dispatch<SetStateAction<StudioState>>;
};

export function HomePage({ bundle, setBundle, studio, setStudio }: HomePageProps) {
  const navigate = useNavigate();
  const { text, phonemes, diacritics, diacriticsSource, languages, language, blueVoice, blueVoiceIds, speaker, targetSpeaker, speed, audioPath, streamChunkPaths, audioAutoplayPending, step, status, busy, error } = studio;
  const loadingLanguagesRef = useRef(false);

  const audioSrc = useMemo(() => (audioPath ? convertFileSrc(audioPath) : ""), [audioPath]);
  const streamedAudioSrcs = useMemo(
    () => streamChunkPaths.map((path) => convertFileSrc(path)),
    [streamChunkPaths],
  );
  // Prefer the streamed chunks (available first, identical audio) and fall back
  // to the finalized WAV for replays of history where no chunks were captured.
  const playerSources = useMemo(
    () => (streamedAudioSrcs.length ? streamedAudioSrcs : audioSrc ? [audioSrc] : []),
    [streamedAudioSrcs, audioSrc],
  );
  const updateStudio = (patch: Partial<StudioState>) => setStudio((current) => ({ ...current, ...patch }));
  const isHebrew = language === "he" || (language === "auto" && /[֐-׿]/.test(text));
  // Vocalized text only counts while it still matches the plain text it came
  // from; after the text is edited it is kept for reference but not spoken.
  const diacriticsStale = Boolean(diacritics) && diacriticsSource !== text;
  const vocalized = isHebrew && !diacriticsStale && diacritics.trim() ? diacritics : "";
  // Generate sends the most refined layer present: edited IPA, then vocalized
  // Hebrew (RenikudPlus honors typed niqqud), then the plain text.
  const inputSource: EditorInputSource = phonemes.trim() ? "phonemes" : vocalized ? "diacritics" : "text";
  const synthesisInput = inputSource === "phonemes" ? phonemes : inputSource === "diacritics" ? vocalized : text;

  useEffect(() => {
    const unlisten = listen<string>("synthesis-chunk", ({ payload }) => {
      setStudio((current) => (
        current.streamChunkPaths.includes(payload)
          ? current
          : {
              ...current,
              streamChunkPaths: [...current.streamChunkPaths, payload],
              audioAutoplayPending: true,
              status: "Playing generated audio...",
            }
      ));
    });
    return () => {
      void unlisten.then((remove) => remove());
    };
  }, [setStudio]);

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

  async function convertToPhonemes() {
    if (!text.trim()) return;
    await ensureModelLoaded();
    const output = await invoke<string>("phonemize", {
      request: { input: vocalized || text, language },
    });
    updateStudio({ phonemes: output });
  }

  async function addDiacritics() {
    if (!text.trim()) return;
    await ensureModelLoaded();
    const source = text;
    const output = await invoke<string>("diacritize", { request: { input: source, language: "he" } });
    updateStudio({ diacritics: output, diacriticsSource: source, phonemes: "" });
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

    updateStudio({ busy: true, error: "", audioPath: "", streamChunkPaths: [], audioAutoplayPending: false });
    try {
      updateStudio({ step: "starting", status: "Initializing Engine..." });
      await invoke<RunnerInfo>("start_runner");

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

      updateStudio({ step: "creating", status: "Generating audio..." });
      const output = await invoke<string>("synthesize", {
        request: {
          input,
          voice: synthesisVoice || undefined,
          language: synthesisLanguage,
          input_is_phonemes: inputIsPhonemes,
          speed,
        },
      });
      // Chunk files exist solely for low-latency playback while inference is
      // running. Keep their playlist alive until the Web Audio queue finishes;
      // clearing it here would unmount the scheduler and cut off playback.
      updateStudio({
        audioPath: output,
        audioAutoplayPending: false,
        step: "done",
        status: "Generation complete.",
      });
    } catch (err) {
      updateStudio({ step: "idle", error: String(err), status: "Generation failed." });
    } finally {
      updateStudio({ busy: false });
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
              {(audioPath || streamedAudioSrcs.length > 0) && (
                <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}>
                  <WaveformPlayer
                    sources={playerSources}
                    downloadPath={audioPath}
                    complete={Boolean(audioPath)}
                    filename={(audioPath || streamChunkPaths[0] || "generated-audio.wav").split(/[\\/]/).pop() || "generated-audio.wav"}
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
