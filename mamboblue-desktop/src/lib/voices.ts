import voiceCatalogJson from "../assets/voices.json";
import type { VoiceCatalog } from "./types";

export const voiceCatalog = voiceCatalogJson as VoiceCatalog;
export type VoiceFilter = "all" | "male" | "female" | "american" | "british";

export const voiceFilters: Array<{ id: VoiceFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "female", label: "Female ♀" },
  { id: "male", label: "Male ♂" },
  { id: "american", label: "🇺🇸 American" },
  { id: "british", label: "🇬🇧 British" },
];
