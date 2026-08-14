import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useRef } from "react";
import type { Dispatch, SetStateAction } from "react";
import { useNavigate } from "react-router-dom";
import type { ModelBundle, RunnerInfo, StudioState } from "../../lib/types";
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
  advancedMode: boolean;
  hebrewG2pEngine: string;
  phonikudPath: string;
};

export function HomePage({ bundle, setBundle, studio, setStudio, advancedMode, hebrewG2pEngine, phonikudPath }: HomePageProps) {
  const navigate = useNavigate();
  const { text, phonemes, diacritics, languages, language, blueVoice, blueVoiceIds, speaker, targetSpeaker, audioPath, streamChunkPaths, audioAutoplayPending, step, status, busy, error } = studio;
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
            renikud_path: currentBundle.codec_path,
            hebrew_g2p_engine: hebrewG2pEngine,
            phonikud_path: phonikudPath,
            speaker,
            target_speaker: targetSpeaker,
          },
        });
        const supportedLanguages = await invoke<string[]>("get_languages");
        const nextStudio: Partial<StudioState> = {};
        if (supportedLanguages.length) nextStudio.languages = supportedLanguages;
        try {
          const voiceIds = await invoke<string[]>("get_voices");
          if (voiceIds.length) nextStudio.blueVoiceIds = voiceIds;
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
        renikud_path: current.codec_path,
            hebrew_g2p_engine: hebrewG2pEngine,
            phonikud_path: phonikudPath,
            speaker,
            target_speaker: targetSpeaker,
      },
    });
  }

  async function convertToPhonemes() {
    if (!text.trim()) return;
    await ensureModelLoaded();
    const output = await invoke<string>("phonemize", {
      request: { input: diacritics || text, language },
    });
    updateStudio({ phonemes: output });
  }

  async function addDiacritics() {
    await ensureModelLoaded();
    const output = await invoke<string>("diacritize", { request: { input: text, language: "he" } });
    updateStudio({ diacritics: output, phonemes: "" });
  }

  async function createVoice() {
    const current = bundle ?? (await invoke<ModelBundle>("get_model_bundle_for_runtime", { runtime: "blue" }));
    setBundle(current);
    if (!current.installed) {
      navigate("/onboard", { replace: true });
      return;
    }
    const input = advancedMode && phonemes.trim() ? phonemes : text;
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
          renikud_path: current.codec_path,
            hebrew_g2p_engine: hebrewG2pEngine,
            phonikud_path: phonikudPath,
            speaker,
            target_speaker: targetSpeaker,
        },
      });

      const supportedLanguages = await invoke<string[]>("get_languages");
      const nextStudio: Partial<StudioState> = { languages: supportedLanguages.length ? supportedLanguages : ["auto"] };
      try {
        const voiceIds = await invoke<string[]>("get_voices");
        if (voiceIds.length) nextStudio.blueVoiceIds = voiceIds;
      } catch {
        // Keep synthesis usable even if voice listing is unavailable.
      }
      updateStudio(nextStudio);
      const selectedLanguage = supportedLanguages.includes(language) ? language : "auto";
      if (selectedLanguage !== language) updateStudio({ language: "auto" });
      const synthesisLanguage = advancedMode && phonemes.trim() && selectedLanguage === "auto"
        ? (/[֐-׿]/.test(text) ? "he" : "en")
        : selectedLanguage;

      updateStudio({ step: "creating", status: "Generating audio..." });
      const output = await invoke<string>("synthesize", {
        request: {
          input,
          voice: blueVoice,
          language: synthesisLanguage,
          input_is_phonemes: advancedMode && Boolean(phonemes.trim()),
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
              hebrewG2pEngine={hebrewG2pEngine}
              diacritics={diacritics}
              setDiacritics={(nextDiacritics) => updateStudio({ diacritics: nextDiacritics, phonemes: "" })}
              addDiacritics={addDiacritics}
              advancedMode={advancedMode}
              phonemes={phonemes}
              setPhonemes={(nextPhonemes) => updateStudio({ phonemes: nextPhonemes })}
              convertToPhonemes={convertToPhonemes}
              createVoice={createVoice}
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
              blueVoice={blueVoice}
              blueVoiceIds={blueVoiceIds}
              hebrewG2pEngine={hebrewG2pEngine}
              speaker={speaker}
              targetSpeaker={targetSpeaker}
              setLanguage={(nextLanguage) => updateStudio({ language: nextLanguage })}
              setBlueVoice={(nextVoice) => updateStudio({ blueVoice: nextVoice })}
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
