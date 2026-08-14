export type ModelBundle = {
  installed: boolean;
  runtime: string;
  model_path: string;
  codec_path: string;
  voices_path?: string;
  espeak_data_path?: string;
  model_dir: string;
  version: string;
  url: string;
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
  archive_url?: string | null;
  directory: string;
  capabilities: RuntimeCapabilities;
};

export type RuntimeCapabilities = {
  hebrew: boolean;
  streaming: boolean;
  voice_reference: boolean;
  fixed_voices: boolean;
};

export type ModelSources = {
  runtimes: ModelSource[];
  voices_url: string;
  default_paths: string[];
};

export type RunnerInfo = {
  base_url: string;
};

export type DownloadProgress = {
  downloaded: number;
  total?: number | null;
  progress?: number | null;
  stage: "downloading" | "extracting";
};

export type CreateStep = "idle" | "starting" | "loading" | "creating" | "done";

export type StudioState = {
  text: string;
  phonemes: string;
  diacritics: string;
  referencePath: string;
  languages: string[];
  language: string;
  blueVoice: string;
  blueVoiceIds: string[];
  speaker: number;
  targetSpeaker: number;
  audioPath: string;
  streamChunkPaths: string[];
  audioAutoplayPending: boolean;
  step: CreateStep;
  status: string;
  busy: boolean;
  error: string;
};

export type VoicePreset = {
  id: string;
  name: string;
  description: string;
  language: string;
  url: string;
};

export type VoiceCatalog = {
  version: string;
  source: string;
  text: string;
  voices: VoicePreset[];
};

export type DownloadedVoice = {
  path: string;
};
