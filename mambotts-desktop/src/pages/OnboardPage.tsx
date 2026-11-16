import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { motion } from "framer-motion";
import { Check, ChevronRight, Download, Loader2, Sparkles } from "lucide-react";
import { filesize } from "filesize";
import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import type { DownloadProgress, ModelBundle, ModelSource, ModelSources } from "../lib/types";
import { Button, Brand, Card, ErrorBlock, Eyebrow, Progress } from "../components/ui";

type PageProps = {
  bundle: ModelBundle | null;
  setBundle: (bundle: ModelBundle) => void;
};

function estimateBytes(sizeLabel?: string): number {
  if (!sizeLabel) return 560 * 1024 * 1024;
  const match = sizeLabel.match(/([\d.]+)\s*(gb|mb|kb)/i);
  if (!match) return 560 * 1024 * 1024;
  const value = Number(match[1]);
  const unit = match[2].toLowerCase();
  if (unit === "gb") return value * 1024 * 1024 * 1024;
  if (unit === "mb") return value * 1024 * 1024;
  return value * 1024;
}

export function OnboardPage({ bundle, setBundle }: PageProps) {
  const navigate = useNavigate();
  const location = useLocation();
  // BlueV3 v3.0.1 is the only model, so there is nothing to choose: this page
  // installs it, or opens the studio once it is there.
  const [blueBundle, setBlueBundle] = useState<ModelBundle | null>(null);
  const [source, setSource] = useState<ModelSource | null>(null);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const manageMode = new URLSearchParams(location.search).get("manage") === "1";

  useEffect(() => {
    const unlisten = listen<DownloadProgress>("model_download_progress", (event) => {
      setProgress(event.payload);
    });
    return () => {
      unlisten.then((off) => off());
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function refresh() {
      try {
        const [blue, modelSources] = await Promise.all([
          invoke<ModelBundle>("get_model_bundle_for_runtime", { runtime: "blue" }),
          invoke<ModelSources>("get_model_sources"),
        ]);
        if (cancelled) return;
        setBlueBundle(blue);
        setSource(modelSources.runtimes.find((option) => option.id === "blue") ?? null);
      } catch (err) {
        if (!cancelled) setError(String(err));
      }
    }
    refresh();
    return () => {
      cancelled = true;
    };
  }, [bundle?.runtime]);

  async function install() {
    setBusy(true);
    setError("");
    try {
      const installed = blueBundle?.installed
        ? blueBundle
        : await invoke<ModelBundle>("download_model_bundle", { runtime: "blue" });
      setBlueBundle(installed);
      await invoke("stop_runner").catch(() => undefined);
      setBundle(installed);
      navigate("/home", { replace: true });
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  const size = source?.size ?? "~1.7 GB";
  const installed = !!blueBundle?.installed;
  const active = !!bundle?.installed;
  const progressValue = (() => {
    if (progress?.stage === "extracting") return 100;
    if (typeof progress?.progress === "number") {
      return Math.round(Math.min(1, Math.max(0, progress.progress)) * 100);
    }
    if (progress?.downloaded) {
      const total =
        typeof progress.total === "number" && progress.total > 0
          ? progress.total
          : estimateBytes(size);
      return Math.round(Math.min(0.99, progress.downloaded / total) * 100);
    }
    return busy ? 4 : 0;
  })();
  const progressLabel =
    progress?.stage === "extracting"
      ? "100%"
      : `${progressValue}%`;
  const stageLabel =
    progress?.stage === "extracting"
      ? "Optimizing models..."
      : progress?.stage === "upgrading" || progress?.stage === "upgraded"
        ? "Upgrading in the background..."
        : "Downloading voice model...";

  return (
    <main className="grid min-h-screen place-items-center bg-background p-6 text-primary sm:p-12">
      <section className="w-full max-w-[660px]">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
        >
          <div className="flex items-center justify-between gap-4">
            <Brand />
            {manageMode ? (
              <Link
                to="/settings"
                className="inline-flex h-9 items-center gap-2 rounded-full border border-border/50 bg-white px-3 text-[10px] font-black uppercase tracking-[0.16em] text-secondary shadow-sm transition-all hover:border-primary hover:text-primary"
              >
                <ChevronRight className="h-3.5 w-3.5 rotate-180" />
                Back
              </Link>
            ) : null}
          </div>
          <h1 className="mt-8 text-4xl font-semibold tracking-tight text-primary sm:text-5xl">
            {manageMode ? "Your voice" : "Local voice"} <br />
            <span className="text-secondary opacity-40 italic">{manageMode ? "model." : "reimagined."}</span>
          </h1>
          <p className="mt-6 max-w-[480px] text-lg leading-relaxed text-secondary opacity-70">
            {manageMode
              ? "BlueV3 v3.0.1 runs on this computer. Reinstalling restarts the runner."
              : "A professional-grade voice engine that runs entirely on your hardware. Total privacy, zero latency."}
          </p>

          <Card className="mt-10 overflow-hidden border-none bg-white p-0 shadow-2xl">
            <div className="flex flex-col gap-5 bg-background/10 p-6 sm:flex-row sm:items-center sm:justify-between">
              <div className="space-y-1">
                <Eyebrow>Voice model</Eyebrow>
                <h3 className="text-base font-semibold tracking-tight">
                  BlueV3 <span className="text-secondary opacity-50">{blueBundle?.version ?? source?.version ?? "bluev3-3.0.1"}</span>
                </h3>
                <p className="max-w-[320px] text-xs leading-5 text-secondary opacity-55">
                  {source?.description ?? "Local speech in Hebrew, English, Spanish, German, Italian, and Russian"}. Fifteen voices.
                </p>
                <p className="text-xs text-secondary opacity-40">
                  {installed ? "Already installed locally" : `Initial setup: ${size} storage (BlueV3 + RenikudPlus)`}
                </p>
              </div>
              <Button onClick={install} disabled={busy} className="h-11 px-6 text-sm shadow-lg shadow-primary/5">
                {busy ? (
                  <span className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Preparing...
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    {installed ? <Check className="h-4 w-4" /> : <Download className="h-4 w-4" />}
                    {active ? "Open Studio" : installed ? "Use Model" : "Install Model"}
                  </span>
                )}
              </Button>
            </div>

            {(busy || progress) && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="border-t border-border/10 bg-background/30 p-8"
              >
                <div className="mb-4 flex items-center justify-between text-[11px] font-bold uppercase tracking-widest">
                  <span className="flex items-center gap-2 text-primary">
                    {progress?.stage === "extracting" ? <Sparkles className="h-3.5 w-3.5" /> : <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                    {stageLabel}
                  </span>
                  <span className="font-mono text-base">{progressLabel}</span>
                </div>
                <Progress value={progressValue} />
                {progress?.downloaded ? (
                  <p className="mt-3 text-xs font-medium text-secondary opacity-50">
                    {progress.total
                      ? `${filesize(progress.downloaded, { standard: "jedec" })} of ${filesize(progress.total, { standard: "jedec" })}`
                      : `${filesize(progress.downloaded, { standard: "jedec" })} of ~${filesize(estimateBytes(size), { standard: "jedec" })}`}
                  </p>
                ) : null}
              </motion.div>
            )}
          </Card>
        </motion.div>

        {error && <ErrorBlock className="mt-8">{error}</ErrorBlock>}
      </section>
    </main>
  );
}
