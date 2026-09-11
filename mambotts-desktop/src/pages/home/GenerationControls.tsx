import { ChevronRight, Gauge, UserRound } from "lucide-react";
import { CSSProperties } from "react";

// Decoration only. The bundle decides which voices exist, so anything not
// listed here still renders from its own name rather than disappearing.
const voiceLabels: Record<string, string> = {
  Noa: "Noa · clear feminine",
  Lily: "Lily · bright feminine",
  Daniel: "Daniel · clear masculine",
  Adam: "Adam · warm masculine",
};

// Measured on Windows: from 0.75 upward the pace changes cleanly. Slower than
// that the model spreads the text over a canvas it cannot fill and the audio
// loses level and articulation, so the slider does not offer it.
export const SPEED_MIN = 0.75;
export const SPEED_MAX = 1.5;

/// The two choices that decide how a take sounds. They sit with the text and
/// the Generate button because they get changed between takes, unlike the
/// language and Hebrew context, which are set once and left alone.
export function GenerationControls({
  busy,
  blueVoice,
  blueVoiceIds,
  speed,
  setBlueVoice,
  setSpeed,
}: {
  busy: boolean;
  blueVoice: string;
  blueVoiceIds: string[];
  speed: number;
  setBlueVoice: (voice: string) => void;
  setSpeed: (speed: number) => void;
}) {
  const speedFill = ((speed - SPEED_MIN) / (SPEED_MAX - SPEED_MIN)) * 100;
  const selectClass =
    "h-10 w-full appearance-none rounded-lg border border-border/40 bg-white pl-9 pr-9 text-xs font-bold tracking-tight text-primary outline-none transition-all hover:border-border focus:border-primary focus:ring-4 focus:ring-primary/5 disabled:cursor-not-allowed disabled:opacity-50";

  return (
    <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
      <label className="relative block">
        <span className="sr-only">Voice</span>
        <UserRound className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-secondary opacity-40" />
        <select
          value={blueVoice}
          onChange={(event) => setBlueVoice(event.currentTarget.value)}
          disabled={busy || blueVoiceIds.length === 0}
          aria-label="Voice"
          className={selectClass}
        >
          {blueVoiceIds.length === 0 && <option value="">Loading voices…</option>}
          {blueVoiceIds.map((voice) => (
            <option key={voice} value={voice}>
              {voiceLabels[voice] ?? voice}
            </option>
          ))}
        </select>
        <ChevronRight className="pointer-events-none absolute right-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 rotate-90 opacity-30" />
      </label>

      <div className="flex h-10 items-center gap-3 rounded-lg border border-border/40 bg-white px-3">
        <Gauge className="h-4 w-4 shrink-0 text-secondary opacity-40" />
        {/* The engine re-paces the speech as it generates rather than stretching
            the finished audio, so pitch holds at either end of the range. */}
        <input
          type="range"
          min={SPEED_MIN}
          max={SPEED_MAX}
          step={0.05}
          value={speed}
          onChange={(event) => setSpeed(Number(event.currentTarget.value))}
          disabled={busy}
          aria-label="Speech speed"
          className="speed-range min-w-0 flex-1"
          style={{ "--speed-fill": `${speedFill}%` } as CSSProperties}
        />
        <button
          type="button"
          onClick={() => setSpeed(1)}
          disabled={busy}
          title="Reset to normal speed"
          className="shrink-0 rounded-md bg-background px-2 py-1 text-[11px] font-bold tabular-nums text-primary transition-colors hover:bg-accent disabled:cursor-not-allowed"
        >
          {speed.toFixed(2)}×
        </button>
      </div>
    </div>
  );
}
