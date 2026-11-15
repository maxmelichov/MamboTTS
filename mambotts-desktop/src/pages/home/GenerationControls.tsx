import { ChevronRight, Gauge, Sparkles, UserRound } from "lucide-react";
import { CSSProperties } from "react";

// Decoration only. The bundle decides which voices exist, so anything not
// listed here still renders from its own name rather than disappearing.
// BlueV3 ships these fifteen, all real voices (no aliases), listed here in
// the bundle's order; the picker follows the server's list, which is the
// same. The descriptors say only what is certain about each voice.
const voiceLabels: Record<string, string> = {
  Maya: "Maya · feminine",
  Noa: "Noa · feminine",
  Lily: "Lily · feminine",
  Tamar: "Tamar · feminine",
  Shira: "Shira · feminine",
  Yael: "Yael · feminine",
  Emma: "Emma · feminine",
  Sara: "Sara · feminine",
  Adam: "Adam · masculine",
  Daniel: "Daniel · masculine",
  David: "David · masculine",
  Jonathan: "Jonathan · masculine",
  Ethan: "Ethan · masculine",
  Michael: "Michael · masculine",
  Ariel: "Ariel · masculine",
};

// The slider is bluetts.com's: the engine's speed itself, 0.8 to 1.2, 1.0 until
// someone moves it in every language (BlueV3 v3.0.1: each voice's speed_scale
// already sets its natural pace per language, and multiplies this speed).
export const SPEED_MIN = 0.8;
export const SPEED_MAX = 1.2;
export const SPEED_DEFAULT = 1.0;
export const SPEED_STORAGE_KEY = "speech-speed-engine";

// The Quality slider is bluetts.com's: the step count itself, 5 to 16 in
// steps of 1, 8 by default, named by the site's words for its ranges.
export const STEPS_MIN = 5;
export const STEPS_MAX = 16;
export const STEPS_DEFAULT = 8;
export const STEPS_STORAGE_KEY = "speech-quality-steps";

/// The site's word for a step count: Fast 5-6, Mid 7-9, High 10-12, Best 13-16.
export function qualityLabel(steps: number): string {
  return steps <= 6 ? "Fast" : steps <= 9 ? "Mid" : steps <= 12 ? "High" : "Best";
}

/// A stored step count kept inside the slider's range, or the default.
export function clampSteps(value: unknown): number {
  const steps = Math.round(Number(value));
  if (!Number.isFinite(steps) || steps <= 0) return STEPS_DEFAULT;
  return Math.min(STEPS_MAX, Math.max(STEPS_MIN, steps));
}

/// The two choices that decide how a take sounds. They sit with the text and
/// the Generate button because they get changed between takes, unlike the
/// language and Hebrew context, which are set once and left alone.
export function GenerationControls({
  busy,
  blueVoice,
  blueVoiceIds,
  speed: chosenSpeed,
  steps,
  setBlueVoice,
  setSpeed,
  setSteps,
}: {
  busy: boolean;
  blueVoice: string;
  blueVoiceIds: string[];
  speed: number | null;
  steps: number;
  setBlueVoice: (voice: string) => void;
  setSpeed: (speed: number | null) => void;
  setSteps: (steps: number) => void;
}) {
  const speed = chosenSpeed ?? SPEED_DEFAULT;
  const speedFill = ((speed - SPEED_MIN) / (SPEED_MAX - SPEED_MIN)) * 100;
  const stepsFill = ((steps - STEPS_MIN) / (STEPS_MAX - STEPS_MIN)) * 100;
  const selectClass =
    "h-10 w-full appearance-none rounded-lg border border-border/40 bg-white pl-9 pr-9 text-xs font-bold tracking-tight text-primary outline-none transition-all hover:border-border focus:border-primary focus:ring-4 focus:ring-primary/5 disabled:cursor-not-allowed disabled:opacity-50";

  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)]">
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

      <div className="flex h-10 items-center gap-3 rounded-lg border border-border/40 bg-white px-3" title="Quality: more steps sound cleaner and take longer">
        <Sparkles className="h-4 w-4 shrink-0 text-secondary opacity-40" />
        <input
          type="range"
          min={STEPS_MIN}
          max={STEPS_MAX}
          step={1}
          value={steps}
          onChange={(event) => setSteps(Number(event.currentTarget.value))}
          disabled={busy}
          aria-label="Quality"
          aria-valuetext={`${qualityLabel(steps)} (${steps} steps)`}
          className="speed-range min-w-0 flex-1"
          style={{ "--speed-fill": `${stepsFill}%` } as CSSProperties}
        />
        {/* Like the site: the word is shown, screen readers hear "Mid (8 steps)". */}
        <button
          type="button"
          onClick={() => setSteps(STEPS_DEFAULT)}
          disabled={busy}
          title={`${steps} steps · reset to ${STEPS_DEFAULT}`}
          aria-hidden="true"
          tabIndex={-1}
          className="w-11 shrink-0 rounded-md bg-background px-2 py-1 text-center text-[11px] font-bold text-primary transition-colors hover:bg-accent disabled:cursor-not-allowed"
        >
          {qualityLabel(steps)}
        </button>
      </div>

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
          onClick={() => setSpeed(null)}
          disabled={busy}
          title="Reset to the natural speed (1.00×)"
          className="shrink-0 rounded-md bg-background px-2 py-1 text-[11px] font-bold tabular-nums text-primary transition-colors hover:bg-accent disabled:cursor-not-allowed"
        >
          {speed.toFixed(2)}×
        </button>
      </div>
    </div>
  );
}
