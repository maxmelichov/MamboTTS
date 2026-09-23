// One AudioContext for the whole app. WebKit (the macOS webview) only lets a
// context make sound once it has been resumed inside a user gesture, and the
// player first needs one seconds after Generate was clicked, when the first
// chunk arrives. Creating and resuming it in the Generate click unlocks it, so
// the take can start playing by itself.
let shared: AudioContext | null = null;

export function sharedAudioContext(): AudioContext {
  if (!shared || shared.state === "closed") shared = new AudioContext();
  return shared;
}

/** Call synchronously from a click handler. */
export function unlockAudio() {
  void sharedAudioContext().resume().catch(() => {});
}
