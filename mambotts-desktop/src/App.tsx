import { invoke } from "@tauri-apps/api/core";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import "./App.css";
import { Brand } from "./components/ui";
import { AgentsPage } from "./pages/AgentsPage";
import { HomePage } from "./pages/home/HomePage";
import { OnboardPage } from "./pages/OnboardPage";
import { SettingsPage } from "./pages/SettingsPage";
import { ModelBundle, StudioState } from "./lib/types";
import { SPEED_MAX, SPEED_MIN, SPEED_STORAGE_KEY, STEPS_DEFAULT, STEPS_STORAGE_KEY, clampSteps } from "./pages/home/GenerationControls";
import { sampleText } from "./lib/constants";

/// The speed someone set on the slider, kept inside its range, or null when
/// they never moved it (1.0 then, as on bluetts.com).
/// "speech-speed" held a multiplier on top of that default and is not read:
/// carried over, 0.95 would have meant 0.855 for Hebrew.
function storedSpeed(): number | null {
  const raw = localStorage.getItem(SPEED_STORAGE_KEY);
  if (raw === null) return null;
  const speed = Number(raw);
  if (!Number.isFinite(speed) || speed <= 0) return null;
  return Math.min(SPEED_MAX, Math.max(SPEED_MIN, speed));
}

/// The Quality slider's step count someone set, or the site's default 8.
function storedSteps(): number {
  try {
    const raw = localStorage.getItem(STEPS_STORAGE_KEY);
    return raw === null ? STEPS_DEFAULT : clampSteps(raw);
  } catch {
    return STEPS_DEFAULT;
  }
}

function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const [bundle, setBundle] = useState<ModelBundle | null>(null);
  const [checking, setChecking] = useState(true);
  const [studio, setStudio] = useState<StudioState>({
    text: sampleText,
    phonemes: "",
    diacritics: "",
    diacriticsSource: "",
    languages: ["auto"],
    language: "auto",
    // Left empty on purpose. The catalog arrives with the loaded bundle, and
    // the server falls back to its own default when no voice is sent.
    blueVoice: "",
    blueVoiceIds: [],
    speaker: 0,
    targetSpeaker: 0,
    // A speed someone chose once is a preference, so it outlives the session.
    speed: storedSpeed(),
    steps: storedSteps(),
    audioPath: "",
    streamChunkUrls: [],
    generation: 0,
    audioAutoplayPending: false,
    step: "idle",
    status: "Ready to generate.",
    busy: false,
    error: "",
  });

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      try {
        const blue = await invoke<ModelBundle>("get_model_bundle_for_runtime", { runtime: "blue" });
        if (cancelled) return;
        setBundle(blue);
        if (blue.installed && location.pathname === "/") navigate("/onboard", { replace: true });
        if (!blue.installed && location.pathname !== "/onboard") navigate("/onboard", { replace: true });
      } catch {
        if (!cancelled && location.pathname !== "/onboard") navigate("/onboard", { replace: true });
      } finally {
        if (!cancelled) setChecking(false);
      }
    }

    boot();
    return () => {
      cancelled = true;
    };
  }, [location.pathname, navigate]);

  // Background upgrades, with no onboarding: an earlier release's bundle
  // (BlueV3 3.0.0) is moved to the current one first, reusing every file
  // whose bytes did not change; then an install from before the fp32 default
  // fetches the machine's RenikudPlus. Each download resumes after a failure
  // (retried with a growing pause) or a restart; once verified, the new
  // bundle replaces the old one and the next take's load_model switches to it.
  const upgradingRef = useRef(false);
  const pendingUpgrade = !bundle?.installed
    ? null
    : bundle.bundle_upgrade
      ? "bundle"
      : bundle.hebrew_g2p_upgrade
        ? "hebrew_g2p"
        : null;
  useEffect(() => {
    if (!pendingUpgrade || upgradingRef.current) return;
    upgradingRef.current = true;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const attempt = async (failures: number) => {
      try {
        const upgraded = await invoke<ModelBundle>(
          pendingUpgrade === "bundle" ? "upgrade_model_bundle" : "upgrade_hebrew_g2p",
        );
        if (!stopped) setBundle(upgraded);
      } catch (err) {
        if (stopped) return;
        // A file the Hub no longer serves (401/403/404/410) will not come
        // back on retry; the app keeps running on what it has.
        if (String(err).includes("is no longer available")) {
          console.error(err);
          return;
        }
        const pause = Math.min(10 * 60_000, 15_000 * 2 ** failures);
        timer = setTimeout(() => attempt(failures + 1), pause);
        return;
      }
      upgradingRef.current = false;
    };
    attempt(0);
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
      upgradingRef.current = false;
    };
  }, [pendingUpgrade]);

  useEffect(() => {
    // Retired preferences: the Hebrew G2P engine chooser (Hebrew now always uses
    // RenikudPlus), the Settings-only phoneme toggle (the editor tabs now live
    // in the studio), the engine picker's choice (BlueV3 v3.0.1 is the only
    // model) and the 1.3.x speed multiplier. Dropping the keys keeps old values
    // from lingering; the studio starts on BlueV3's defaults.
    try {
      localStorage.removeItem("hebrew-g2p-engine");
      localStorage.removeItem("advanced-mode");
      localStorage.removeItem("mambotts.runtime");
      localStorage.removeItem("speech-speed");
    } catch {
      // Storage can be unavailable; nothing depends on the cleanup.
    }
  }, []);

  if (checking) {
    return (
      <main className="grid min-h-screen place-items-center bg-background p-8 text-primary">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col items-center space-y-6 text-center"
        >
          <Brand className="scale-150" />
          <div className="flex items-center gap-3 text-secondary">
            <Loader2 className="h-4 w-4 animate-spin" />
            <p className="text-sm font-medium tracking-wide">Preparing your local voice studio...</p>
          </div>
        </motion.div>
      </main>
    );
  }

  return (
    <Routes>
      <Route path="/onboard" element={<OnboardPage bundle={bundle} setBundle={setBundle} />} />
      <Route path="/home" element={<HomePage bundle={bundle} setBundle={setBundle} studio={studio} setStudio={setStudio} />} />
      <Route path="/agents" element={<AgentsPage bundle={bundle} />} />
      <Route path="/settings" element={<SettingsPage bundle={bundle} />} />
      <Route path="*" element={<Navigate to={bundle?.installed ? "/home" : "/onboard"} replace />} />
    </Routes>
  );
}

export default App;
