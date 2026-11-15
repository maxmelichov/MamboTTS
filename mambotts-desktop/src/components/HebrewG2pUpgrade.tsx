import { listen } from "@tauri-apps/api/event";
import { filesize } from "filesize";
import { Download } from "lucide-react";
import { useEffect, useState } from "react";
import type { DownloadProgress, ModelBundle } from "../lib/types";
import { Card, Progress } from "./ui";

/**
 * The background upgrades' progress (an earlier release's bundle moving to
 * the current one, then RenikudPlus to fp32), from the same
 * `model_download_progress` event the onboarding download shows. Nothing is
 * blocked meanwhile: synthesis keeps running on what is installed.
 */
export function HebrewG2pUpgrade({ bundle }: { bundle: ModelBundle | null }) {
  const [progress, setProgress] = useState<DownloadProgress | null>(null);

  useEffect(() => {
    const unlisten = listen<DownloadProgress>("model_download_progress", (event) => {
      if (event.payload.stage === "upgrading" || event.payload.stage === "upgraded") setProgress(event.payload);
    });
    return () => {
      unlisten.then((off) => off());
    };
  }, []);

  if (!bundle?.installed || !(bundle.bundle_upgrade || bundle.hebrew_g2p_upgrade)) return null;
  const value = Math.round(Math.min(1, Math.max(0, progress?.progress ?? 0)) * 100);
  const verifying = progress?.stage === "upgrading" && progress.total != null && progress.downloaded >= progress.total;
  const release = bundle.bundle_upgrade;
  const title = release
    ? progress?.stage === "upgraded"
      ? "Switching to the new voice model..."
      : verifying
        ? "Verifying the new voice model..."
        : `Upgrading to ${release}`
    : progress?.stage === "upgraded"
      ? "Switching Hebrew model..."
      : verifying
        ? "Verifying Hebrew model..."
        : "Upgrading Hebrew model";
  const note = release
    ? "A new BlueV3 release. Files that did not change are reused, and MamboTTS keeps working on the current model until the new one is ready."
    : "The full-precision RenikudPlus reads Hebrew as bluetts.com does. MamboTTS keeps working on the compact model until it is ready.";

  return (
    <Card className="space-y-3 border-none p-5 shadow-xl" role="status" aria-live="polite">
      <div className="flex items-center justify-between gap-3 text-[11px] font-bold uppercase tracking-widest">
        <span className="flex items-center gap-2 text-primary">
          <Download className="h-3.5 w-3.5" />
          {title}
        </span>
        <span className="font-mono text-sm">{value}%</span>
      </div>
      <Progress value={value} />
      <p className="text-xs text-secondary/55">
        {progress?.total
          ? `${filesize(progress.downloaded, { standard: "jedec" })} of ${filesize(progress.total, { standard: "jedec" })}. `
          : ""}
        {note}
      </p>
    </Card>
  );
}
