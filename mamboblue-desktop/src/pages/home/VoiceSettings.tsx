import { AudioLines, ChevronRight, Languages, UserRound } from "lucide-react";

import { Button, Card, Eyebrow } from "../../components/ui";

const voiceLabels: Record<string, { name: string; detail: string }> = {
  Rotem: { name: "Rotem", detail: "Clear feminine BlueTTS voice" },
  Roi: { name: "Roi", detail: "Clear masculine BlueTTS voice" },
  female1: { name: "Rotem", detail: "Clear feminine BlueTTS voice" },
  male1: { name: "Roi", detail: "Clear masculine BlueTTS voice" },
};

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
  setLanguage,
  setBlueVoice,
  setSpeaker,
  setTargetSpeaker,
}: {
  busy: boolean;
  language: string;
  languages: string[];
  blueVoice: string;
  blueVoiceIds: string[];
  hebrewG2pEngine: string;
  speaker: number;
  targetSpeaker: number;
  setLanguage: (language: string) => void;
  setBlueVoice: (voice: string) => void;
  setSpeaker: (speaker: number) => void;
  setTargetSpeaker: (speaker: number) => void;
}) {
  const voices = blueVoiceIds.length ? blueVoiceIds : ["Rotem", "Roi"];

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

      {hebrewG2pEngine === "renikud" && (
        <>
      <div className="h-px bg-border/10" />

      <div className="space-y-4">
        <div className="flex items-center gap-2.5">
          <Languages className="h-4 w-4 text-secondary opacity-40" />
          <Eyebrow className="mb-0">Hebrew speaker context</Eyebrow>
        </div>
        <p className="text-xs text-secondary/55">RenikudPlus uses these hints when choosing Hebrew IPA. They affect pronunciation, not the generated voice; use Rotem or Roi above to change the voice.</p>
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
