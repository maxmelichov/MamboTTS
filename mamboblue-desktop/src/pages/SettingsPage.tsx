import { invoke } from "@tauri-apps/api/core";
import { openPath } from "@tauri-apps/plugin-opener";
import { Check, ChevronRight, FolderOpen, Sparkles } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import type { ModelBundle } from "../lib/types";
import { AppFrame } from "../components/AppFrame";
import { Button, Card, ErrorBlock } from "../components/ui";

type SettingsPageProps = {
  bundle: ModelBundle | null;
  advancedMode: boolean;
  setAdvancedMode: (enabled: boolean) => void;
  hebrewG2pEngine: string;
  setHebrewG2pEngine: (engine: string) => void;
  setPhonikudPath: (path: string) => void;
};

export function SettingsPage({ bundle, advancedMode, setAdvancedMode, hebrewG2pEngine, setHebrewG2pEngine, setPhonikudPath }: SettingsPageProps) {
  const [error, setError] = useState("");
  const [downloadingPhonikud, setDownloadingPhonikud] = useState(false);

  async function choosePhonikud() {
    setDownloadingPhonikud(true);
    setError("");
    try {
      const model = await invoke<{ path: string }>("download_phonikud_bundle");
      setPhonikudPath(model.path);
      setHebrewG2pEngine("phonikud");
    } catch (err) {
      setError(String(err));
    } finally {
      setDownloadingPhonikud(false);
    }
  }

  async function openModelsFolder() {
    setError("");
    try {
      const current = bundle ?? (await invoke<ModelBundle>("get_model_bundle"));
      await openPath(current.model_dir);
    } catch (err) {
      setError(String(err));
    }
  }

  return (
    <AppFrame bundle={bundle}>
      <section className="w-full max-w-[640px] space-y-12">
        <header className="space-y-6">
          <Link to="/home" className="inline-flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.2em] text-secondary opacity-40 transition-all hover:text-primary hover:opacity-100">
            <ChevronRight className="h-3 w-3 rotate-180" />
            Studio
          </Link>
          <div className="space-y-1">
            <h1 className="text-3xl font-semibold tracking-tight text-primary sm:text-4xl">System Settings</h1>
            <p className="max-w-[440px] text-base text-secondary opacity-60">Manage local models and high-fidelity voice assets.</p>
          </div>
        </header>

        <div className="space-y-8">
          <div className="space-y-4">
            <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-secondary opacity-30">Infrastructure & Storage</h3>
            <Card className="overflow-hidden border-none shadow-xl">
              <div className="flex flex-col gap-8 p-8 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0 flex-1 space-y-1">
                  <p className="text-[9px] font-bold uppercase tracking-widest text-secondary opacity-30">Models Directory</p>
                  <p className="font-mono text-[11px] text-secondary/70 bg-background/50 px-3 py-2 rounded-lg border border-border/10 truncate">
                    {bundle?.model_dir ?? "Resolving system path..."}
                  </p>
                </div>
                <Button variant="outline" onClick={openModelsFolder} className="gap-2 h-10 px-4 shrink-0 text-xs">
                  <FolderOpen className="h-4 w-4" />
                  Open Models Folder
                </Button>
              </div>

              <div className="flex flex-col gap-8 p-8 sm:flex-row sm:items-center sm:justify-between bg-background/10">
                <div className="space-y-0.5">
                  <p className="text-[9px] font-bold uppercase tracking-widest text-secondary opacity-30">Engine Specification</p>
                  <p className="text-xl font-semibold tracking-tight text-primary">{bundle?.version ?? "blue-onnx-v2"}</p>
                </div>
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white border border-border/40 text-[9px] font-black uppercase tracking-[0.2em] text-green-600 shadow-sm">
                  <div className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                  Live System
                </div>
              </div>
              <div className="flex flex-col gap-4 border-t border-border/10 p-8 sm:flex-row sm:items-center sm:justify-between">
                <div className="space-y-1">
                  <p className="text-[9px] font-bold uppercase tracking-widest text-secondary opacity-30">Runtime</p>
                  <p className="text-sm font-semibold tracking-tight text-primary">BlueTTS</p>
                </div>
                <Link
                  to="/onboard?manage=1"
                  className="inline-flex h-10 shrink-0 items-center justify-center gap-2 rounded-lg border border-border/80 bg-white px-4 text-xs font-semibold text-primary shadow-sm transition-all hover:border-primary"
                >
                  Reinstall Model
                  <ChevronRight className="h-3.5 w-3.5" />
                </Link>
              </div>
            </Card>
          </div>
          <div className="space-y-4">
            <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-secondary opacity-30">Hebrew G2P Engine</h3>
            <Card className="space-y-5 border-none p-8 shadow-xl">
              <div className="space-y-1">
                <p className="text-base font-semibold tracking-tight text-primary">Hebrew pronunciation</p>
                <p className="text-sm text-secondary/60">Select the engine that creates IPA before speech generation.</p>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <button
                  type="button"
                  onClick={() => setHebrewG2pEngine("renikud")}
                  className={`relative rounded-xl border p-5 text-left transition-all ${hebrewG2pEngine === "renikud" ? "border-primary bg-primary text-white shadow-lg shadow-primary/10" : "border-border/40 bg-background/20 text-primary hover:border-primary/50"}`}
                >
                  {hebrewG2pEngine === "renikud" && <Check className="absolute right-4 top-4 h-4 w-4" />}
                  <p className="text-sm font-bold">RenikudPlus</p>
                  <p className={`mt-1 text-xs leading-relaxed ${hebrewG2pEngine === "renikud" ? "text-white/70" : "text-secondary/60"}`}>Hebrew-to-IPA via the RenikudPlus ONNX model with speaker/target conditioning.</p>
                </button>
                <button
                  type="button"
                  onClick={choosePhonikud}
                  disabled={downloadingPhonikud}
                  className={`relative rounded-xl border p-5 text-left transition-all disabled:cursor-wait ${hebrewG2pEngine === "phonikud" ? "border-primary bg-primary text-white shadow-lg shadow-primary/10" : "border-border/40 bg-background/20 text-primary hover:border-primary/50"}`}
                >
                  {hebrewG2pEngine === "phonikud" ? <Check className="absolute right-4 top-4 h-4 w-4" /> : <Sparkles className="absolute right-4 top-4 h-4 w-4 text-secondary/40" />}
                  <p className="text-sm font-bold">{downloadingPhonikud ? "Downloading model…" : "Phonikud"}</p>
                  <p className={`mt-1 text-xs leading-relaxed ${hebrewG2pEngine === "phonikud" ? "text-white/70" : "text-secondary/60"}`}>Adds vocalization controls and a Diacritics editor for Hebrew text.</p>
                </button>
              </div>
              <div className="flex items-start gap-2 rounded-lg bg-background/30 px-3 py-2.5 text-xs text-secondary/60">
                <Sparkles className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary/60" />
                <span>Phonikud downloads its diacritics model once. Its Diacritics tab becomes available in the studio when selected.</span>
              </div>
            </Card>
          </div>
          <div className="space-y-4">
            <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-secondary opacity-30">Advanced</h3>
            <Card className="border-none p-8 shadow-xl">
              <label className="flex cursor-pointer items-center justify-between gap-6">
                <span className="space-y-1">
                  <span className="block text-sm font-semibold tracking-tight text-primary">Phoneme editor</span>
                  <span className="block max-w-[420px] text-sm text-secondary/60">
                    Edit the model&apos;s IPA input directly and use the exact phoneme inventory from the loaded model.
                  </span>
                </span>
                <input
                  type="checkbox"
                  checked={advancedMode}
                  onChange={(event) => setAdvancedMode(event.currentTarget.checked)}
                  className="h-5 w-5 shrink-0 accent-primary"
                />
              </label>
            </Card>
          </div>
          {error && <ErrorBlock>{error}</ErrorBlock>}
        </div>
      </section>
    </AppFrame>
  );
}
