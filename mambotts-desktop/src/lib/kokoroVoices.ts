export type KokoroVoiceFilter = "all" | "male" | "female" | "english" | "japanese" | "spanish" | "french" | "hindi" | "italian" | "portuguese" | "chinese";

export type KokoroVoiceInfo = {
  id: string;
  name: string;
  language: string;
  flag: string;
  gender: "male" | "female";
};

type PrefixInfo = Omit<KokoroVoiceInfo, "id" | "name">;

const kokoroVoicePrefixes: Record<string, PrefixInfo> = {
  af: { language: "American English", flag: "🇺🇸", gender: "female" },
  am: { language: "American English", flag: "🇺🇸", gender: "male" },
  bf: { language: "British English", flag: "🇬🇧", gender: "female" },
  bm: { language: "British English", flag: "🇬🇧", gender: "male" },
  ef: { language: "Spanish", flag: "🇪🇸", gender: "female" },
  em: { language: "Spanish", flag: "🇪🇸", gender: "male" },
  ff: { language: "French", flag: "🇫🇷", gender: "female" },
  hf: { language: "Hindi", flag: "🇮🇳", gender: "female" },
  hm: { language: "Hindi", flag: "🇮🇳", gender: "male" },
  if: { language: "Italian", flag: "🇮🇹", gender: "female" },
  im: { language: "Italian", flag: "🇮🇹", gender: "male" },
  jf: { language: "Japanese", flag: "🇯🇵", gender: "female" },
  jm: { language: "Japanese", flag: "🇯🇵", gender: "male" },
  pf: { language: "Brazilian Portuguese", flag: "🇧🇷", gender: "female" },
  pm: { language: "Brazilian Portuguese", flag: "🇧🇷", gender: "male" },
  zf: { language: "Chinese", flag: "🇨🇳", gender: "female" },
  zm: { language: "Chinese", flag: "🇨🇳", gender: "male" },
};

export const kokoroVoiceFilters: Array<{ id: KokoroVoiceFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "female", label: "Female ♀" },
  { id: "male", label: "Male ♂" },
  { id: "english", label: "🇺🇸/🇬🇧 English" },
  { id: "japanese", label: "🇯🇵 Japanese" },
  { id: "spanish", label: "🇪🇸 Spanish" },
  { id: "french", label: "🇫🇷 French" },
  { id: "hindi", label: "🇮🇳 Hindi" },
  { id: "italian", label: "🇮🇹 Italian" },
  { id: "portuguese", label: "🇧🇷 Portuguese" },
  { id: "chinese", label: "🇨🇳 Chinese" },
];

export function describeKokoroVoice(id: string): KokoroVoiceInfo {
  const prefix = id.split("_")[0] ?? "";
  const info = kokoroVoicePrefixes[prefix] ?? { language: "Unknown", flag: "🌐", gender: "female" as const };
  return {
    id,
    name: formatKokoroVoiceName(id),
    ...info,
  };
}

export function describeKokoroVoices(ids: string[]) {
  return ids.map(describeKokoroVoice);
}

export function filterKokoroVoices(voices: KokoroVoiceInfo[], filter: KokoroVoiceFilter) {
  return voices.filter((voice) => {
    if (filter === "all") return true;
    if (filter === "male" || filter === "female") return voice.gender === filter;
    if (filter === "english") return voice.language.includes("English");
    if (filter === "portuguese") return voice.language.includes("Portuguese");
    return voice.language.toLowerCase().includes(filter);
  });
}

function formatKokoroVoiceName(id: string) {
  const rawName = id.includes("_") ? id.split("_").slice(1).join("_") : id;
  return rawName
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
