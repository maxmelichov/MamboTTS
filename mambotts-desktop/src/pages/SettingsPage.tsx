import { invoke } from "@tauri-apps/api/core";
import { openPath } from "@tauri-apps/plugin-opener";
import { ChevronRight, FolderOpen } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import type { ModelBundle } from "../lib/types";
import { AppFrame } from "../components/AppFrame";
import { Button, Card, ErrorBlock } from "../components/ui";

type SettingsPageProps = {
  bundle: ModelBundle | null;
};

export function SettingsPage({ bundle }: SettingsPageProps) {
  const [error, setError] = useState("");

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
          {error && <ErrorBlock>{error}</ErrorBlock>}
        </div>
      </section>
    </AppFrame>
  );
}
