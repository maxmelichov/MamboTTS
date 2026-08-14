import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { motion } from "framer-motion";
import { Check, ChevronRight, Download, Loader2, Sparkles } from "lucide-react";
import { filesize } from "filesize";
import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import type { DownloadProgress, ModelBundle, ModelSource, ModelSources } from "../lib/types";
import { cn } from "../lib/classNames";
import { Button, Brand, Card, ErrorBlock, Eyebrow, Progress } from "../components/ui";

type PageProps = {
  bundle: ModelBundle | null;
  setBundle: (bundle: ModelBundle) => void;
};

type RuntimeId = ModelBundle["runtime"];

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
  const [runtime, setRuntime] = useState<RuntimeId>("blue");
  const [bundles, setBundles] = useState<Record<RuntimeId, ModelBundle | null>>({ blue: null });
  const [sources, setSources] = useState<ModelSource[]>([]);
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
    async function refreshBundles() {
      try {
        const [blue, modelSources] = await Promise.all([
          invoke<ModelBundle>("get_model_bundle_for_runtime", { runtime: "blue" }),
          invoke<ModelSources>("get_model_sources"),
        ]);
        if (cancelled) return;
        setBundles({ blue });
        setSources(modelSources.runtimes);
        setRuntime("blue");
      } catch (err) {
        if (!cancelled) setError(String(err));
      }
    }
    refreshBundles();
    return () => {
      cancelled = true;
    };
  }, [bundle?.runtime]);

  async function selectRuntime(nextRuntime: RuntimeId) {
    setBusy(true);
    setError("");
    try {
      setRuntime(nextRuntime);
      const existing = bundles[nextRuntime];
      const selected = existing?.installed ? existing : await invoke<ModelBundle>("download_model_bundle", { runtime: nextRuntime });
      localStorage.setItem("mamborambo.runtime", selected.runtime);
      setBundles((current) => ({ ...current, [selected.runtime]: selected }));
      await invoke("stop_runner").catch(() => undefined);
      setBundle(selected);
      navigate("/home", { replace: true });
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  const options = (sources.length ? sources : [
    {
      id: "blue",
      name: "BlueTTS",
      version: "blue-onnx-v2",
      size: "~560 MB",
      description: "Local Hebrew and English speech",
      files: [],
      directory: "blue-onnx-v2",
      capabilities: { hebrew: true, streaming: true, voice_reference: false, fixed_voices: true },
    },
  ]) as ModelSource[];
  const selectedSource = options.find((option) => option.id === runtime) ?? options[0];
  const progressValue = (() => {
    if (progress?.stage === "extracting") return 100;
    if (typeof progress?.progress === "number") {
      return Math.round(Math.min(1, Math.max(0, progress.progress)) * 100);
    }
    if (progress?.downloaded) {
      const total =
        typeof progress.total === "number" && progress.total > 0
          ? progress.total
          : estimateBytes(selectedSource?.size);
      return Math.round(Math.min(0.99, progress.downloaded / total) * 100);
    }
    return busy ? 4 : 0;
  })();
  const progressLabel =
    progress?.stage === "extracting"
      ? "100%"
      : `${progressValue}%`;
  const stageLabel = progress?.stage === "extracting" ? "Optimizing models..." : "Downloading voice model...";

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
            {manageMode ? "Manage voice" : "Local voice"} <br />
            <span className="text-secondary opacity-40 italic">{manageMode ? "models." : "reimagined."}</span>
          </h1>
          <p className="mt-6 max-w-[480px] text-lg leading-relaxed text-secondary opacity-70">
            {manageMode
              ? "Install or switch between local engines. The current runner restarts automatically after a change."
              : "A professional-grade voice engine that runs entirely on your hardware. Total privacy, zero latency."}
          </p>

          <Card className="mt-10 overflow-hidden border-none bg-white p-0 shadow-2xl">
            <div className="grid gap-3 border-b border-border/10 bg-white p-3 sm:grid-cols-2">
              {options.map((option) => {
                const installed = !!bundles[option.id]?.installed;
                const selected = !!bundle?.installed && bundle.runtime === option.id;
                return (
                  <button
                    key={option.id}
                    onClick={() => setRuntime(option.id)}
                    className={cn(
                      "flex min-h-[150px] flex-col justify-between rounded-xl border p-4 text-left transition-all",
                      runtime === option.id ? "border-primary bg-background/50 shadow-sm ring-1 ring-primary" : "border-border/40 bg-white hover:border-secondary/40",
                    )}
                  >
                    <div className="space-y-4">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <h3 className="text-base font-semibold tracking-tight">{option.name}</h3>
                          <p className="mt-1 text-[10px] font-black uppercase tracking-[0.18em] text-secondary opacity-35">{option.size}</p>
                        </div>
                        <span
                          className={cn(
                            "inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-1 text-[8px] font-bold uppercase tracking-widest",
                            selected
                              ? "bg-primary text-white"
                              : installed
                                ? "bg-green-50 text-green-700"
                                : "bg-primary text-white",
                          )}
                        >
                          {selected || installed ? <Check className="h-3 w-3" /> : null}
                          {selected ? "Selected" : installed ? "Installed" : "Available"}
                        </span>
                      </div>
                      <p className="max-w-[220px] text-xs font-semibold leading-5 text-secondary opacity-55">{option.description}</p>
                    </div>
                    <div className="mt-4 flex items-center gap-2 text-[9px] font-black uppercase tracking-[0.18em] text-secondary opacity-35">
                      <div className={cn("h-1.5 w-1.5 rounded-full", runtime === option.id ? "bg-primary" : "bg-secondary/30")} />
                      Local Runtime
                    </div>
                  </button>
                );
              })}
            </div>
            <div className="flex flex-col gap-5 bg-background/10 p-6 sm:flex-row sm:items-center sm:justify-between">
              <div className="space-y-1">
                <Eyebrow>Local Infrastructure</Eyebrow>
                <h3 className="text-base font-semibold tracking-tight">{bundles[runtime]?.version ?? "blue-onnx-v2"}</h3>
                <p className="text-xs text-secondary opacity-40">
                  {bundles[runtime]?.installed ? "Already installed locally" : "Initial setup: ~560 MB storage (BlueTTS + RenikudPlus)"}
                </p>
              </div>
              <Button onClick={() => selectRuntime(runtime)} disabled={busy} className="h-11 px-6 text-sm shadow-lg shadow-primary/5">
                {busy ? (
                  <span className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Preparing...
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    {bundle?.installed && bundle.runtime === runtime ? <Check className="h-4 w-4" /> : bundles[runtime]?.installed ? <Check className="h-4 w-4" /> : <Download className="h-4 w-4" />}
                    {bundle?.installed && bundle.runtime === runtime ? "Open Studio" : bundles[runtime]?.installed ? "Use Model" : "Install Models"}
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
                      : `${filesize(progress.downloaded, { standard: "jedec" })} of ~${filesize(estimateBytes(selectedSource?.size), { standard: "jedec" })}`}
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
