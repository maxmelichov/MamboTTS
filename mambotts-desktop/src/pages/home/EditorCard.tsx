import { Loader2, Play, RotateCw, Square, X } from "lucide-react";
import { useRef, useState, type KeyboardEvent } from "react";
import { cn } from "../../lib/classNames";
import type { EditorInputSource } from "../../lib/types";
import { Button, Card } from "../../components/ui";
import { GenerationControls } from "./GenerationControls";
import { unlockAudio } from "../../lib/audioContext";

const hebrewPhonemeGroups = [
  { label: "Stress", items: [{ symbol: "ˈ", name: "Stress" }] },
  {
    label: "Vowels",
    items: [
      { symbol: "a", name: "Shamar" },
      { symbol: "e", name: "Shemer" },
      { symbol: "i", name: "Shimer" },
      { symbol: "o", name: "Shomer" },
      { symbol: "u", name: "Shumar" },
    ],
  },
  {
    label: "Consonants",
    items: [
      { symbol: "b", name: "Bet" }, { symbol: "v", name: "Vet, Vav" },
      { symbol: "d", name: "Daled" }, { symbol: "h", name: "Hey" },
      { symbol: "z", name: "Zain" }, { symbol: "χ", name: "Het, Haf" },
      { symbol: "t", name: "Taf, Tet" }, { symbol: "j", name: "Yud" },
      { symbol: "k", name: "Kuf, Kaf" }, { symbol: "l", name: "Lamed" },
      { symbol: "m", name: "Mem" }, { symbol: "n", name: "Nun" },
      { symbol: "s", name: "Sin, Samekh" }, { symbol: "f", name: "Fey" },
      { symbol: "p", name: "Pey" }, { symbol: "ts", name: "Tsadik" },
      { symbol: "tʃ", name: "Tsadik with Geresh" }, { symbol: "w", name: "Vav" },
      { symbol: "ʔ", name: "Alef, Ayin" }, { symbol: "ɡ", name: "Gimel" },
      { symbol: "ʁ", name: "Resh" }, { symbol: "ʃ", name: "Shin" },
      { symbol: "ʒ", name: "Zain with Geresh" }, { symbol: "dʒ", name: "Gimel with Geresh" },
    ],
  },
] as const;

const diacriticGroups = [
  { label: "Vowels", marks: [["ְ", "Sheva"], ["ֱ", "Hataf Segol"], ["ֲ", "Hataf Patah"], ["ֳ", "Hataf Qamats"], ["ִ", "Hiriq"], ["ֵ", "Tsere"], ["ֶ", "Segol"], ["ַ", "Patah"], ["ָ", "Qamats"], ["ֹ", "Holam"], ["ֻ", "Qubuts"], ["ׇ", "Qamats Qatan"]] },
  { label: "Letter marks", marks: [["ּ", "Dagesh"], ["ֿ", "Rafe"], ["ׁ", "Shin dot"], ["ׂ", "Sin dot"]] },
  { label: "Pronunciation", marks: [["ֽ", "Meteg"], ["\u05ab", "Hatama (stress)"]] },
] as const;

/** RenikudPlus marks the stressed letter with the accent ole (U+05AB) and reads one per word back as the stress. */
const HATAMA = "\u05ab";
const vowelMarks = new Set<string>(diacriticGroups[0].marks.map(([mark]) => mark));
const hebrewLetter = /[\u05d0-\u05ea]/;
const hebrewMark = /[\u0591-\u05c7]/;

function hebrewLetterClusters(text: string) {
  const chars = Array.from(text);
  const clusters: Array<{ start: number; value: string } | { value: string }> = [];
  for (let index = 0; index < chars.length; index += 1) {
    if (!hebrewLetter.test(chars[index])) {
      clusters.push({ value: chars[index] });
      continue;
    }
    let value = chars[index];
    let end = index + 1;
    while (end < chars.length && hebrewMark.test(chars[end])) {
      value += chars[end];
      end += 1;
    }
    clusters.push({ start: index, value });
    index = end - 1;
  }
  return clusters;
}

function groupHebrewWords(clusters: ReturnType<typeof hebrewLetterClusters>) {
  const words: Array<typeof clusters> = [];
  let word: typeof clusters = [];
  for (const cluster of clusters) {
    if (!("start" in cluster) && !cluster.value.trim()) {
      if (word.length) words.push(word);
      word = [];
    } else {
      word.push(cluster);
    }
  }
  if (word.length) words.push(word);
  return words;
}

type EditorTab = "text" | "diacritics" | "phonemes";

/** The tiny uppercase label style the tabs use, shared by everything quiet in the editor chrome. */
const chromeLabel = "text-[10px] font-bold uppercase tracking-[0.16em]";
const groupLabel = "text-[10px] font-bold uppercase tracking-[0.16em] text-secondary/40";
/** A compact key in the Niqqud and IPA palettes. */
const keycap = "grid h-8 min-w-8 place-items-center rounded-md border px-2 text-primary transition-colors disabled:opacity-50";
/** The full-bleed editing surface every tab shares. */
const surface = "w-full resize-none bg-white p-8 text-start text-lg font-medium leading-relaxed text-primary outline-none placeholder:text-secondary/20";

type EditorCardProps = {
  blueVoice: string;
  blueVoiceIds: string[];
  speed: number;
  setBlueVoice: (voice: string) => void;
  setSpeed: (speed: number) => void;
  busy: boolean;
  text: string;
  setText: (text: string) => void;
  language: string;
  isHebrew: boolean;
  /** The one switch behind both pronunciation layers (Niqqud only exists for Hebrew). */
  pronunciationEnabled: boolean;
  setPronunciationEnabled: (enabled: boolean) => void;
  /** The layer Generate will send, and its content. */
  inputSource: EditorInputSource;
  synthesisInput: string;
  diacritics: string;
  /** True once the plain text changed after the diacritics were generated. */
  diacriticsStale: boolean;
  setDiacritics: (text: string) => void;
  clearDiacritics: () => void;
  /** Resolves with the vocalized text. */
  addDiacritics: () => Promise<string>;
  phonemes: string;
  setPhonemes: (phonemes: string) => void;
  /** Takes niqqud produced in the same step, before state has caught up with it. */
  convertToPhonemes: (freshVocalized?: string) => Promise<void>;
  createVoice: () => void;
  /** Stops the generation in progress. */
  stopVoice: () => void;
  stopping: boolean;
};

export function EditorCard({
  busy,
  text,
  setText,
  language,
  isHebrew,
  pronunciationEnabled,
  setPronunciationEnabled,
  inputSource,
  synthesisInput,
  diacritics,
  diacriticsStale,
  setDiacritics,
  clearDiacritics,
  addDiacritics,
  phonemes,
  setPhonemes,
  convertToPhonemes,
  createVoice,
  stopVoice,
  stopping,
  blueVoice,
  blueVoiceIds,
  speed,
  setBlueVoice,
  setSpeed,
}: EditorCardProps) {
  const diacriticsEnabled = pronunciationEnabled && isHebrew;
  const [selectedTab, setTab] = useState<EditorTab>("text");
  // A layer's tab only exists while the switch is on (and Niqqud only for
  // Hebrew); fall back to the text when the selected one goes away.
  const tab: EditorTab = (selectedTab === "diacritics" && !diacriticsEnabled) || (selectedTab === "phonemes" && !pronunciationEnabled)
    ? "text"
    : selectedTab;
  const phonemeInput = useRef<HTMLTextAreaElement>(null);
  const [converting, setConverting] = useState<"diacritics" | "phonemes" | null>(null);
  const [editorError, setEditorError] = useState("");
  const [selectedLetter, setSelectedLetter] = useState<number | null>(null);
  const [direction, setDirection] = useState<"rtl" | "ltr" | null>(null);
  function changeDirection(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (!event.ctrlKey || !event.shiftKey || event.altKey || event.metaKey) return;
    const next = event.code === "KeyR" || event.code === "ShiftRight" ? "rtl"
      : event.code === "KeyL" || event.code === "ShiftLeft" ? "ltr" : null;
    if (next) {
      event.preventDefault();
      setDirection(next);
    }
  }
  const textPlaceholder = isHebrew ? "הדביקו כאן טקסט בעברית..." : "Paste your text here...";

  function insertPhoneme(phoneme: string) {
    const textarea = phonemeInput.current;
    const start = textarea?.selectionStart ?? phonemes.length;
    const end = textarea?.selectionEnd ?? phonemes.length;
    setPhonemes(`${phonemes.slice(0, start)}${phoneme}${phonemes.slice(end)}`);
    requestAnimationFrame(() => {
      textarea?.focus();
      textarea?.setSelectionRange(start + phoneme.length, start + phoneme.length);
    });
  }

  /**
   * Fills the given layers from the text. Niqqud goes first, because IPA is
   * read from the vocalized Hebrew when there is some.
   */
  async function runConversion(kinds: Array<"diacritics" | "phonemes">) {
    setEditorError("");
    try {
      let vocalized: string | undefined;
      if (kinds.includes("diacritics")) {
        setConverting("diacritics");
        setSelectedLetter(null);
        vocalized = await addDiacritics();
      }
      if (kinds.includes("phonemes")) {
        setConverting("phonemes");
        await convertToPhonemes(vocalized);
      }
    } catch (error) {
      setEditorError(String(error));
    } finally {
      setConverting(null);
    }
  }

  function selectTab(next: EditorTab) {
    setEditorError("");
    setTab(next);
  }
  function togglePronunciation() {
    const enabled = !pronunciationEnabled;
    setPronunciationEnabled(enabled);
    if (!enabled) {
      // Off returns to the plain text; the layers keep their contents.
      selectTab("text");
      return;
    }
    selectTab(isHebrew ? "diacritics" : "phonemes");
    // The layers are there to correct the reading, so they open already
    // holding the model's reading of the text rather than empty. One that
    // still has content (even stale niqqud, which says so) is left alone,
    // except that fresh niqqud clears the IPA, which then follows it.
    if (!text.trim()) return;
    const fillDiacritics = isHebrew && !diacritics.trim();
    const fillPhonemes = fillDiacritics || !phonemes.trim();
    const kinds = [
      ...(fillDiacritics ? ["diacritics" as const] : []),
      ...(fillPhonemes ? ["phonemes" as const] : []),
    ];
    if (kinds.length) void runConversion(kinds);
  }
  function changeDiacritic(mark: string, isVowel: boolean) {
    if (selectedLetter === null) return;
    const chars = Array.from(diacritics);
    if (!hebrewLetter.test(chars[selectedLetter] ?? "")) return;
    let end = selectedLetter + 1;
    while (end < chars.length && hebrewMark.test(chars[end])) end += 1;
    const existing = chars.slice(selectedLetter + 1, end);
    const hasMark = existing.includes(mark);
    const nextMarks = existing
      .filter((current) => current !== mark)
      .filter((current) => !(isVowel && vowelMarks.has(current)));
    if (!hasMark) nextMarks.push(mark);
    chars.splice(selectedLetter + 1, end - selectedLetter - 1, ...nextMarks);
    if (mark === HATAMA && !hasMark) {
      // A word has one stress, and RenikudPlus keeps only the last hatama it
      // finds in a word, so moving the stress has to take it off the letter
      // that had it. Words are whitespace-delimited, as they are for the model.
      let wordStart = selectedLetter;
      while (wordStart > 0 && !/\s/.test(chars[wordStart - 1])) wordStart -= 1;
      let wordEnd = selectedLetter;
      while (wordEnd < chars.length && !/\s/.test(chars[wordEnd])) wordEnd += 1;
      const keep = selectedLetter + 1 + nextMarks.indexOf(HATAMA);
      let removedBefore = 0;
      for (let index = wordEnd - 1; index >= wordStart; index -= 1) {
        if (chars[index] !== HATAMA || index === keep) continue;
        chars.splice(index, 1);
        if (index < selectedLetter) removedBefore += 1;
      }
      // Dropping a hatama earlier in the word moves the selected letter left.
      setDiacritics(chars.join(""));
      setSelectedLetter(selectedLetter - removedBefore);
      return;
    }
    setDiacritics(chars.join(""));
    setSelectedLetter(selectedLetter);
  }

  const selectedCharacter = selectedLetter === null ? "" : Array.from(diacritics)[selectedLetter] ?? "";
  const diacriticLetters = hebrewLetterClusters(diacritics);
  const selectedCluster = diacriticLetters.find((cluster) => "start" in cluster && cluster.start === selectedLetter)?.value ?? selectedCharacter;
  const selectedMarks = new Set(Array.from(selectedCluster).slice(1));
  const diacriticWords = groupHebrewWords(diacriticLetters);

  const tabs: Array<{ id: EditorTab; label: string }> = [
    { id: "text", label: "Text" },
    ...(diacriticsEnabled ? [{ id: "diacritics" as const, label: "Niqqud" }] : []),
    ...(pronunciationEnabled ? [{ id: "phonemes" as const, label: "IPA" }] : []),
  ];
  const switchLabel = isHebrew ? "Niqqud & IPA" : "IPA";
  const locked = busy || converting !== null;

  /** The slim line under a layer: one quiet hint on the left, small actions on the right. */
  function layerToolbar(hint: string, onRefresh: () => void, onClear: (() => void) | null, refreshLabel: string, refreshing: boolean) {
    return (
      <div className="flex min-h-10 flex-wrap items-center justify-between gap-x-4 gap-y-1 px-8 pb-3">
        <p className={cn("min-w-0 text-xs", editorError ? "text-red-600" : "text-secondary/50")}>{editorError || hint}</p>
        <div className="flex shrink-0 items-center gap-1">
          {onClear && (
            <Button variant="ghost" onClick={onClear} disabled={locked} className="h-7 gap-1.5 rounded-md px-2.5 text-xs">
              <X className="h-3.5 w-3.5" />
              <span className="text-xs">Clear</span>
            </Button>
          )}
          <Button variant="ghost" onClick={onRefresh} disabled={locked || !text.trim()} className="h-7 gap-1.5 rounded-md px-2.5 text-xs">
            {refreshing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RotateCw className="h-3.5 w-3.5" />}
            <span className="text-xs">{refreshLabel}</span>
          </Button>
        </div>
      </div>
    );
  }

  return (
    <Card className="relative overflow-hidden p-0 shadow-xl border-none">
      <div className="flex flex-wrap items-end justify-between gap-x-4 border-b border-border/10 bg-background/10 px-8 pt-5">
        <div role="tablist" aria-label="Editor layer" className="flex items-center gap-2">
          {tabs.map((item) => (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={tab === item.id}
              onClick={() => selectTab(item.id)}
              title={inputSource === item.id ? "Generate uses this layer" : undefined}
              className={cn("flex items-center gap-1.5 border-b-2 px-3 pb-3", chromeLabel, "transition-colors", tab === item.id ? "border-primary text-primary" : "border-transparent text-secondary/40 hover:text-primary")}
            >
              {item.label}
              {inputSource === item.id && item.id !== "text" && <span className="h-1.5 w-1.5 rounded-full bg-primary" aria-hidden />}
            </button>
          ))}
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={pronunciationEnabled}
          onClick={togglePronunciation}
          disabled={locked}
          title={pronunciationEnabled
            ? `Hide ${switchLabel}; Generate goes back to the plain text`
            : `Fix the pronunciation by editing the ${isHebrew ? "niqqud and IPA" : "IPA"} of your text`}
          className={cn("group flex items-center gap-2 pb-3", chromeLabel, "transition-colors disabled:cursor-not-allowed disabled:opacity-50", pronunciationEnabled ? "text-primary" : "text-secondary/40 hover:text-primary")}
        >
          {converting && <Loader2 className="h-3 w-3 animate-spin" aria-hidden />}
          {switchLabel}
          <span aria-hidden className={cn("relative h-3.5 w-6 rounded-full transition-colors", pronunciationEnabled ? "bg-primary" : "bg-secondary/20 group-hover:bg-secondary/35")}>
            <span className={cn("absolute top-0.5 h-2.5 w-2.5 rounded-full bg-white transition-all", pronunciationEnabled ? "left-3" : "left-0.5")} />
          </span>
        </button>
      </div>
      {tab === "text" ? (
        <textarea
          id="text"
          value={text}
          placeholder={textPlaceholder}
          dir={direction ?? (isHebrew ? "rtl" : "ltr")}
          onKeyDown={changeDirection}
          title="Ctrl+Shift+R: RTL · Ctrl+Shift+L: LTR"
          lang={isHebrew ? "he" : language === "auto" ? undefined : language}
          onChange={(event) => setText(event.currentTarget.value)}
          disabled={busy}
          className={cn(surface, "min-h-[320px]")}
        />
      ) : tab === "diacritics" ? (
        <div className="bg-white">
          <textarea
            value={diacritics}
            placeholder={converting === "diacritics" ? "Adding niqqud..." : "Add niqqud from your text to edit it here."}
            onChange={(event) => setDiacritics(event.currentTarget.value)}
            dir={direction ?? "rtl"}
            onKeyDown={changeDirection}
            lang="he"
            disabled={busy || !diacritics}
            className={cn(surface, "min-h-[200px] pb-4")}
          />
          {layerToolbar(
            diacriticsStale
              ? "The text changed since this niqqud was added, so Generate reads the plain text. Refresh to catch up."
              : "Edit freely. Generate speaks this vocalized text.",
            () => void runConversion(["diacritics"]),
            diacritics ? clearDiacritics : null,
            diacritics ? "Refresh from text" : "Add from text",
            converting === "diacritics",
          )}
          {diacritics && (
            <div className="space-y-4 border-t border-border/10 bg-background/10 px-8 py-5">
              <div>
                <p className={cn("mb-2", groupLabel)}>{selectedCharacter ? "Letter" : "Pick a letter to change its marks"}</p>
                <div dir="rtl" lang="he" className="flex flex-wrap gap-x-4 gap-y-1 text-xl leading-relaxed text-primary">
                  {diacriticWords.map((word, wordIndex) => (
                    <span key={wordIndex} className="inline-flex">
                      {word.map((cluster, index) => "start" in cluster ? (
                        <button
                          key={`${cluster.start}-${index}`}
                          type="button"
                          onClick={() => setSelectedLetter(selectedLetter === cluster.start ? null : cluster.start)}
                          title={cluster.value.includes(HATAMA) ? "Stressed" : undefined}
                          aria-pressed={selectedLetter === cluster.start}
                          className={cn("relative rounded px-px transition-colors", selectedLetter === cluster.start ? "bg-primary text-white" : "hover:bg-primary/5")}
                        >
                          {cluster.value}
                          {cluster.value.includes(HATAMA) && <span aria-hidden className={cn("absolute inset-x-1 -bottom-0.5 h-px", selectedLetter === cluster.start ? "bg-white/70" : "bg-primary/40")} />}
                        </button>
                      ) : <span key={`symbol-${index}`} className="text-secondary/40">{cluster.value}</span>)}
                    </span>
                  ))}
                </div>
              </div>
              {selectedCharacter && (
                <div className="flex flex-wrap gap-x-6 gap-y-3">
                  {diacriticGroups.map((group, groupIndex) => (
                    <div key={group.label}>
                      <p className={cn("mb-1.5", groupLabel)}>{group.label}</p>
                      <div className="flex flex-wrap gap-1">
                        {group.marks.map(([mark, name]) => (
                          <button
                            key={mark}
                            type="button"
                            title={selectedMarks.has(mark) ? `Remove ${name}` : name}
                            aria-label={name}
                            aria-pressed={selectedMarks.has(mark)}
                            onClick={() => changeDiacritic(mark, groupIndex === 0)}
                            disabled={busy}
                            className={cn(keycap, selectedMarks.has(mark) ? "border-primary bg-primary text-white" : "border-border/30 bg-white hover:border-primary/50")}
                          >
                            <span dir="rtl" lang="he" className="text-xl leading-none">{selectedCharacter}{mark}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      ) : (
        <div className="bg-white">
          <textarea
            ref={phonemeInput}
            value={phonemes}
            placeholder={converting === "phonemes" ? "Converting to IPA..." : "Refresh from text, or type IPA yourself."}
            onChange={(event) => setPhonemes(event.currentTarget.value)}
            dir="ltr"
            disabled={busy}
            className={cn(surface, "min-h-[200px] pb-4")}
          />
          {layerToolbar(
            diacriticsEnabled && diacritics && !diacriticsStale
              ? "Read from your niqqud. Generate sends this IPA as is."
              : "Generate sends this IPA as is, instead of the text.",
            () => void runConversion(["phonemes"]),
            phonemes ? () => setPhonemes("") : null,
            phonemes ? "Refresh from text" : "Convert text",
            converting === "phonemes",
          )}
          {isHebrew && (
            <div className="flex flex-wrap gap-x-6 gap-y-3 border-t border-border/10 bg-background/10 px-8 py-5">
              {hebrewPhonemeGroups.map((group) => (
                <div key={group.label} className="min-w-0">
                  <p className={cn("mb-1.5", groupLabel)}>{group.label}</p>
                  <div className="flex flex-wrap gap-1">
                    {group.items.map((phoneme) => (
                      <button
                        key={phoneme.symbol}
                        type="button"
                        title={`${phoneme.symbol}: ${phoneme.name}`}
                        onClick={() => insertPhoneme(phoneme.symbol)}
                        disabled={busy}
                        className={cn(keycap, "border-border/30 bg-white hover:border-primary/50")}
                      >
                        <span className="text-base leading-none">{phoneme.symbol}</span>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      <div className="space-y-4 border-t border-border/10 bg-background/10 px-8 py-5">
        <GenerationControls
          busy={busy}
          blueVoice={blueVoice}
          blueVoiceIds={blueVoiceIds}
          speed={speed}
          setBlueVoice={setBlueVoice}
          setSpeed={setSpeed}
        />
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4 text-[10px] font-bold uppercase tracking-[0.2em] text-secondary">
            <span className={cn("opacity-40 transition-colors", synthesisInput.length > 500 ? "text-amber-600 opacity-100" : "")}>{synthesisInput.length} Characters</span>
          </div>
          {busy ? (
            // While a take is generating the same slot becomes Stop, so a long
            // document started by mistake can be abandoned without quitting.
            // It keeps Generate's dark fill on purpose: the pointer is still
            // resting on this slot from the click, and the pointing hand is
            // white, so a white button here made the cursor all but vanish
            // for the whole generation.
            <Button onClick={stopVoice} disabled={stopping} title="Stop generating" className="h-12 px-8 text-sm shadow-xl shadow-primary/5">
              <span className="flex items-center gap-2">
                {stopping ? <Loader2 className="h-4 w-4 animate-spin" /> : <Square className="h-3.5 w-3.5 fill-current" />}
                {stopping ? "Stopping..." : "Stop"}
              </span>
            </Button>
          ) : (
            <Button onClick={() => { unlockAudio(); createVoice(); }} disabled={!synthesisInput.trim()} className="h-12 px-8 text-sm shadow-xl shadow-primary/5 transition-transform hover:scale-[1.01]">
              <span className="flex items-center gap-2">
                <Play className="h-3.5 w-3.5 fill-current" />
                Generate
              </span>
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
}
