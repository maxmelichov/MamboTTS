/** The sizes a saved recording can be written at. The ids are what the
 * `export_audio_file` command expects (mambotts_audio::ExportQuality). */
export type ExportQuality = "very_small" | "small" | "normal" | "large" | "very_large";

export type ExportQualityOption = {
  id: ExportQuality;
  label: string;
  format: string;
  /** Rough size of one minute of speech, for the picker. */
  perMinute: string;
  extension: "mp3" | "wav";
};

// Speech is mono, so MP3 stays clear well below music bitrates. The sizes are
// bitrate arithmetic; WAV is 44.1 kHz 16-bit mono, the untouched output.
export const exportQualityOptions: ExportQualityOption[] = [
  { id: "very_small", label: "Very small", format: "MP3 64 kbps", perMinute: "~0.5 MB/min", extension: "mp3" },
  { id: "small", label: "Small", format: "MP3 96 kbps", perMinute: "~0.7 MB/min", extension: "mp3" },
  { id: "normal", label: "Normal", format: "MP3 128 kbps", perMinute: "~1 MB/min", extension: "mp3" },
  { id: "large", label: "Large", format: "MP3 192 kbps", perMinute: "~1.4 MB/min", extension: "mp3" },
  { id: "very_large", label: "Very large", format: "WAV lossless", perMinute: "~5.3 MB/min", extension: "wav" },
];

const STORAGE_KEY = "export-quality";
export const DEFAULT_EXPORT_QUALITY: ExportQuality = "normal";

export function exportQualityOption(id: ExportQuality): ExportQualityOption {
  return exportQualityOptions.find((option) => option.id === id) ?? exportQualityOptions[2];
}

/** The size picked last time, so a preference set once holds across sessions. */
export function readExportQuality(): ExportQuality {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return exportQualityOptions.some((option) => option.id === stored) ? (stored as ExportQuality) : DEFAULT_EXPORT_QUALITY;
  } catch {
    return DEFAULT_EXPORT_QUALITY;
  }
}

export function storeExportQuality(quality: ExportQuality) {
  try {
    localStorage.setItem(STORAGE_KEY, quality);
  } catch {
    // Storage can be unavailable; the choice still applies for this session.
  }
}

/** Swap a file name's extension for the one the chosen size is saved as. */
export function withExtension(filename: string, extension: string): string {
  const base = filename.replace(/\.[^./\\]+$/, "");
  return `${base || "generated-audio"}.${extension}`;
}
