import { AudioLines, ChevronRight, Gauge, Languages, UserRound } from "lucide-react";
import { CSSProperties } from "react";

import { Button, Card, Eyebrow } from "../../components/ui";

// Decoration only. The bundle decides which voices exist, so anything not
// listed here still renders from its own name rather than disappearing.
const voiceLabels: Record<string, { name: string; detail: string }> = {
  Noa: { name: "Noa", detail: "Clear feminine BlueTTS voice" },
  Lily: { name: "Lily", detail: "Bright feminine BlueTTS voice" },
  Daniel: { name: "Daniel", detail: "Clear masculine BlueTTS voice" },
  Adam: { name: "Adam", detail: "Warm masculine BlueTTS voice" },
};

// Measured on Windows: from 0.75 upward the pace changes cleanly. Slower than
// that the model spreads the text over a canvas it cannot fill and the audio
// loses level and articulation, so the slider does not offer it.
const SPEED_MIN = 0.75;
const SPEED_MAX = 1.5;

const speedPresets = [
  { label: "Slower", value: 0.85 },
  { label: "Normal", value: 1 },
  { label: "Faster", value: 1.25 },
];

const languageLabels: Record<string, string> = {
  auto: "Detect automatically",
  he: "Hebrew",
  en: "English",
  de: "German",
  es: "Spanish",
  it: "Italian",
};

export function VoiceSettings({
  busy,
  language,
  languages,
  blueVoice,
  blueVoiceIds,
  hebrewG2pEngine,
  speaker,
  targetSpeaker,
  speed,
  setLanguage,
  setBlueVoice,
  setSpeaker,
  setTargetSpeaker,
  setSpeed,
}: {
  busy: boolean;
  language: string;
  languages: string[];
  blueVoice: string;
  blueVoiceIds: string[];
  hebrewG2pEngine: string;
  speaker: number;
  targetSpeaker: number;
  speed: number;
  setLanguage: (language: string) => void;
  setBlueVoice: (voice: string) => void;
  setSpeaker: (speaker: number) => void;
  setTargetSpeaker: (speaker: number) => void;
  setSpeed: (speed: number) => void;
}) {
  // The track paints its own filled portion, so the thumb sits at the end of a
  // solid bar rather than in the middle of an undifferentiated line.
  const speedFill = ((speed - SPEED_MIN) / (SPEED_MAX - SPEED_MIN)) * 100;

  // Never invent a catalog. Naming a voice the loaded bundle does not have is
  // what produced "unknown Blue voice `Rotem`" after the voices were renamed.
  const voices = blueVoiceIds;

  return (
    <Card className="space-y-8 border-none p-6 shadow-xl">
      <div className="space-y-5">
        <div className="flex items-center gap-2.5">
          <AudioLines className="h-4 w-4 text-secondary opacity-40" />
          <Eyebrow className="mb-0">BlueTTS Voice</Eyebrow>
        </div>
        <div className="grid gap-2">
          {voices.map((voice) => {
            const selected = voice === blueVoice;
            const label = voiceLabels[voice] ?? { name: voice, detail: "BlueTTS saved voice" };
            return (
              <Button
                key={voice}
                variant={selected ? "primary" : "outline"}
                disabled={busy}
                onClick={() => setBlueVoice(voice)}
                className="h-auto justify-start gap-3 px-4 py-3 text-left"
              >
                <UserRound className="h-4 w-4 shrink-0" />
                <span className="min-w-0">
                  <span className="block text-xs font-bold">{label.name}</span>
                  <span className="block text-[10px] font-medium opacity-65">{label.detail}</span>
                </span>
              </Button>
            );
          })}
        </div>
      </div>

      <div className="h-px bg-border/10" />

      <div className="space-y-4">
        <div className="flex items-center justify-between gap-2.5">
          <div className="flex items-center gap-2.5">
            <Gauge className="h-4 w-4 text-secondary opacity-40" />
            <Eyebrow className="mb-0">Speed</Eyebrow>
          </div>
          <span className="rounded-md bg-background px-2 py-1 text-[11px] font-bold tabular-nums text-primary">
            {speed.toFixed(2)}×
          </span>
        </div>

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
          className="speed-range w-full"
          style={{ "--speed-fill": `${speedFill}%` } as CSSProperties}
        />

        <div className="grid grid-cols-3 gap-2">
          {speedPresets.map((preset) => (
            <Button
              key={preset.label}
              variant={Math.abs(speed - preset.value) < 0.001 ? "primary" : "outline"}
              disabled={busy}
              onClick={() => setSpeed(preset.value)}
              className="h-8 px-2 text-[10px] font-bold uppercase tracking-[0.14em]"
            >
              {preset.label}
            </Button>
          ))}
        </div>
      </div>

      {hebrewG2pEngine === "renikud" && (
        <>
      <div className="h-px bg-border/10" />

      <div className="space-y-4">
        <div className="flex items-center gap-2.5">
          <Languages className="h-4 w-4 text-secondary opacity-40" />
          <Eyebrow className="mb-0">Hebrew speaker context</Eyebrow>
        </div>
        <p className="text-xs text-secondary/55">RenikudPlus uses these hints when choosing Hebrew IPA. They affect pronunciation, not the generated voice; use the voice buttons above to change the voice.</p>
        <div className="grid grid-cols-2 gap-2">
          {[
            ["Source", speaker, setSpeaker],
            ["Target", targetSpeaker, setTargetSpeaker],
          ].map(([label, value, setValue]) => (
            <label key={label as string} className="space-y-1 text-[10px] font-bold uppercase tracking-wider text-secondary/45">
              {label as string}
              <select value={value as number} onChange={(event) => (setValue as (value: number) => void)(Number(event.currentTarget.value))} disabled={busy} className="h-10 w-full rounded-lg border border-border/30 bg-white px-2 text-xs font-semibold normal-case tracking-normal text-primary">
                <option value={0}>Unknown</option><option value={1}>Male</option><option value={2}>Female</option>
              </select>
            </label>
          ))}
        </div>
      </div>

      <div className="h-px bg-border/10" />
        </>
      )}

      <div className="space-y-5">
        <div className="flex items-center gap-2.5">
          <Languages className="h-4 w-4 text-secondary opacity-40" />
          <Eyebrow className="mb-0">Language</Eyebrow>
        </div>
        <div className="relative">
          <select
            value={language}
            onChange={(event) => setLanguage(event.currentTarget.value)}
            disabled={busy}
            className="h-12 w-full appearance-none rounded-xl border border-border/30 bg-white px-4 text-xs font-bold tracking-tight text-primary outline-none transition-all focus:border-primary focus:ring-4 focus:ring-primary/5"
          >
            {languages.map((item) => (
              <option key={item} value={item}>
                {languageLabels[item] ?? item.toUpperCase()}
              </option>
            ))}
          </select>
          <ChevronRight className="pointer-events-none absolute right-4 top-1/2 h-3.5 w-3.5 -translate-y-1/2 rotate-90 opacity-30" />
        </div>
      </div>
    </Card>
  );
}
