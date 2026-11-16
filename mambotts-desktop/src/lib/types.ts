export type ModelBundle = {
  installed: boolean;
  runtime: string;
  model_path: string;
  codec_path: string;
  /** Hebrew G2P model; BlueTTS reads IPA rather than Hebrew script. */
  renikud_path: string;
  voices_path?: string;
  model_dir: string;
  version: string;
  url: string;
  /** The RenikudPlus weights `renikud_path` names. */
  hebrew_g2p?: "fp32" | "int8" | null;
  /** Set while the machine's own RenikudPlus downloads in the background. */
  hebrew_g2p_upgrade?: "fp32" | "int8" | null;
  /**
   * Set while this bundle is an earlier release's (`version`): the release
   * the background upgrade is moving it to. The app keeps running on it.
   */
  bundle_upgrade?: string | null;
};

export type ModelSourceFile = {
  name: string;
  url: string;
};

export type ModelSource = {
  id: string;
  name: string;
  version: string;
  size: string;
  description: string;
  files: ModelSourceFile[];
  directory: string;
  capabilities: RuntimeCapabilities;
};

export type RuntimeCapabilities = {
  hebrew: boolean;
  streaming: boolean;
  fixed_voices: boolean;
};

export type ModelSources = {
  runtimes: ModelSource[];
  default_paths: string[];
};

export type RunnerInfo = {
  base_url: string;
};

export type DownloadProgress = {
  downloaded: number;
  total?: number | null;
  progress?: number | null;
  stage: "downloading" | "extracting" | "upgrading" | "upgraded";
};

export type CreateStep = "idle" | "starting" | "loading" | "creating" | "done";

export type StudioState = {
  text: string;
  phonemes: string;
  /** Hebrew text with niqqud. When present (and the text is Hebrew) it is what gets synthesized. */
  diacritics: string;
  /** The plain text `diacritics` was generated from, used to flag it as stale after edits. */
  diacriticsSource: string;
  languages: string[];
  language: string;
  blueVoice: string;
  blueVoiceIds: string[];
  speaker: number;
  targetSpeaker: number;
  /** The engine speed the slider set, or null for the default 1.0 (the voice's natural pace). */
  speed: number | null;
  /** The Quality slider: flow-matching steps, 5 to 16 (bluetts.com's, default 8). */
  steps: number;
  audioPath: string;
  /** In-memory blob URLs of the streamed chunks, released once the finished WAV takes over. */
  streamChunkUrls: string[];
  /** Bumped per take so the player knows a new timeline started. */
  generation: number;
  audioAutoplayPending: boolean;
  step: CreateStep;
  status: string;
  busy: boolean;
  error: string;
};

/** Which editor layer the Generate button sends, most refined first. */
export type EditorInputSource = "text" | "diacritics" | "phonemes";
