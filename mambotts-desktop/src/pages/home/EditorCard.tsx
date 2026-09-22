import { Loader2, Play, Sparkles, X } from "lucide-react";
import { useRef, useState, type KeyboardEvent } from "react";
import { cn } from "../../lib/classNames";
import type { EditorInputSource } from "../../lib/types";
import { Button, Card } from "../../components/ui";
import { GenerationControls } from "./GenerationControls";

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

const sourceLabels: Record<EditorInputSource, string> = {
  text: "Plain text",
  diacritics: "Vocalized Hebrew",
  phonemes: "IPA input",
};

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
  /** The layer Generate will send, and its content. */
  inputSource: EditorInputSource;
  synthesisInput: string;
  diacritics: string;
  /** True once the plain text changed after the diacritics were generated. */
  diacriticsStale: boolean;
  setDiacritics: (text: string) => void;
  clearDiacritics: () => void;
  addDiacritics: () => Promise<void>;
  phonemes: string;
  setPhonemes: (phonemes: string) => void;
  convertToPhonemes: () => Promise<void>;
  createVoice: () => void;
};

export function EditorCard({
  busy,
  text,
  setText,
  language,
  isHebrew,
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
  blueVoice,
  blueVoiceIds,
  speed,
  setBlueVoice,
  setSpeed,
}: EditorCardProps) {
  const [selectedTab, setTab] = useState<EditorTab>("text");
  // The Diacritics layer only exists for Hebrew; fall back to the text when the
  // language moves away from it.
  const tab: EditorTab = selectedTab === "diacritics" && !isHebrew ? "text" : selectedTab;
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

  async function runConversion(kind: "diacritics" | "phonemes") {
    setConverting(kind);
    setEditorError("");
    try {
      if (kind === "diacritics") {
        setSelectedLetter(null);
        await addDiacritics();
      } else {
        await convertToPhonemes();
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
    ...(isHebrew ? [{ id: "diacritics" as const, label: "Diacritics" }] : []),
    { id: "phonemes", label: "Phonemes" },
  ];

  return (
    <Card className="relative overflow-hidden p-0 shadow-xl border-none">
      <div role="tablist" aria-label="Editor layer" className="flex items-center gap-2 border-b border-border/10 bg-background/10 px-8 pt-5">
        {tabs.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={tab === item.id}
            onClick={() => selectTab(item.id)}
            title={inputSource === item.id ? "Generate uses this layer" : undefined}
            className={cn("flex items-center gap-1.5 border-b-2 px-3 pb-3 text-[10px] font-bold uppercase tracking-[0.16em] transition-colors", tab === item.id ? "border-primary text-primary" : "border-transparent text-secondary/40 hover:text-primary")}
          >
            {item.label}
            {inputSource === item.id && item.id !== "text" && <span className="h-1.5 w-1.5 rounded-full bg-primary" aria-hidden />}
          </button>
        ))}
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
          className="min-h-[320px] w-full resize-none bg-white p-8 text-start text-lg font-medium leading-relaxed text-primary outline-none placeholder:text-secondary/20"
        />
      ) : tab === "diacritics" ? (
        <div className="bg-white p-8">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-sm font-semibold text-primary">Vocalized Hebrew</p>
              <p className="text-xs text-secondary/55">RenikudPlus adds niqqud to your text. Edit it freely; the vocalized text is what gets spoken.</p>
            </div>
            <div className="flex items-center gap-2">
              {diacritics && (
                <Button variant="ghost" onClick={clearDiacritics} disabled={busy || converting !== null} className="h-9 gap-1.5 px-3 text-xs">
                  <X className="h-3.5 w-3.5" />
                  Clear
                </Button>
              )}
              <Button variant={diacritics ? "outline" : "primary"} onClick={() => void runConversion("diacritics")} disabled={busy || converting !== null || !text.trim()} className="h-9 gap-1.5 px-3 text-xs">
                {converting === "diacritics" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                {converting === "diacritics" ? "Adding…" : diacritics ? "Refresh from text" : "Add diacritics"}
              </Button>
            </div>
          </div>
          {diacriticsStale && (
            <p className="mb-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
              The text changed after these diacritics were added, so Generate uses the plain text. Refresh to vocalize the new text.
            </p>
          )}
          <textarea
            value={diacritics}
            placeholder="Press Add diacritics to vocalize the text…"
            onChange={(event) => setDiacritics(event.currentTarget.value)}
            dir={direction ?? "rtl"}
            onKeyDown={changeDirection}
            lang="he"
            disabled={busy || !diacritics}
            className="min-h-48 w-full resize-y rounded-lg border border-border/50 bg-background/30 p-4 text-lg leading-relaxed text-primary outline-none placeholder:text-sm placeholder:text-secondary/30 focus:border-primary/50"
          />
          {editorError && <p className="mt-3 text-xs text-red-600">{editorError}</p>}
          {diacritics && (
            <div className="mt-5 space-y-3">
              <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-secondary/40">Pick a letter to change its marks</p>
              <div dir="rtl" className="flex flex-wrap items-center gap-3 rounded-lg border border-border/20 bg-background/20 p-3 leading-loose">
                {diacriticWords.map((word, wordIndex) => (
                  <div key={wordIndex} className="flex items-center rounded-lg border border-border/20 bg-white p-1 shadow-sm">
                    {word.map((cluster, index) => "start" in cluster ? (
                      <button key={`${cluster.start}-${index}`} type="button" onClick={() => setSelectedLetter(cluster.start)} title={cluster.value.includes(HATAMA) ? "Stressed" : undefined} className={cn("relative grid h-11 min-w-9 place-items-center rounded-md px-1.5 text-2xl transition-colors", selectedLetter === cluster.start ? "bg-primary text-white" : "text-primary hover:bg-primary/10")}>
                        {cluster.value}
                        {cluster.value.includes(HATAMA) && <span aria-hidden className={cn("absolute inset-x-2 bottom-1 h-0.5 rounded-full", selectedLetter === cluster.start ? "bg-white/70" : "bg-amber-500")} />}
                      </button>
                    ) : <span key={`symbol-${index}`} className="px-0.5 text-xl text-secondary/50">{cluster.value}</span>)}
                  </div>
                ))}
              </div>
              {selectedCharacter && (
                <div className="space-y-3 rounded-lg border border-border/20 bg-background/20 p-3">
                  <p className="text-sm font-semibold text-primary">Editing <span dir="rtl" className="text-2xl">{selectedCluster}</span>{selectedMarks.has(HATAMA) && <span className="ms-2 text-xs font-normal text-amber-600">stressed</span>}</p>
                  {diacriticGroups.map((group, groupIndex) => (
                    <div key={group.label}>
                      <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-secondary/45">{group.label}</p>
                      <div className="flex flex-wrap gap-2">
                        {group.marks.map(([mark, name]) => (
                          <button key={mark} type="button" title={selectedMarks.has(mark) ? `Remove ${name}` : name} aria-pressed={selectedMarks.has(mark)} onClick={() => changeDiacritic(mark, groupIndex === 0)} disabled={busy} className={cn("flex h-12 items-center gap-2 rounded-md border px-3 transition-colors hover:border-primary hover:bg-primary hover:text-white disabled:opacity-50", selectedMarks.has(mark) ? "border-primary bg-primary/10 text-primary" : "border-border/30 bg-white text-primary")}>
                            <span className="text-xl">{selectedCharacter}{mark}</span><span className="text-[10px] opacity-55">{name}</span>
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
        <div className="bg-white p-8">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-sm font-semibold text-primary">Model IPA input</p>
              <p className="text-xs text-secondary/55">
                {isHebrew && diacritics && !diacriticsStale
                  ? "Generated from your vocalized Hebrew. When present, this IPA is sent to BlueTTS as-is."
                  : "IPA for the selected language. When present, it is sent to BlueTTS as-is instead of the text."}
              </p>
            </div>
            <div className="flex items-center gap-2">
              {phonemes && (
                <Button variant="ghost" onClick={() => setPhonemes("")} disabled={busy || converting !== null} className="h-9 gap-1.5 px-3 text-xs">
                  <X className="h-3.5 w-3.5" />
                  Clear
                </Button>
              )}
              <Button variant={phonemes ? "outline" : "primary"} onClick={() => void runConversion("phonemes")} disabled={busy || converting !== null || !text.trim()} className="h-9 gap-1.5 px-3 text-xs">
                {converting === "phonemes" && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                {converting === "phonemes" ? "Converting…" : phonemes ? "Refresh from text" : "Generate phonemes"}
              </Button>
            </div>
          </div>
          <textarea
            ref={phonemeInput}
            value={phonemes}
            placeholder="Press Generate phonemes, or type IPA yourself…"
            onChange={(event) => setPhonemes(event.currentTarget.value)}
            disabled={busy}
            className="min-h-32 w-full resize-y rounded-lg border border-border/50 bg-background/30 p-4 font-mono text-lg leading-relaxed text-primary outline-none placeholder:font-sans placeholder:text-sm placeholder:text-secondary/30 focus:border-primary/50"
          />
          {editorError && <p className="mt-3 text-xs text-red-600">{editorError}</p>}
          {isHebrew && (
          <div className="mt-5">
            <p className="mb-2 text-[10px] font-bold uppercase tracking-[0.16em] text-secondary/40">Hebrew phonemes</p>
            <div className="space-y-3 rounded-lg border border-border/20 bg-background/20 p-3">
              {hebrewPhonemeGroups.map((group) => (
                <div key={group.label}>
                  <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-secondary/45">{group.label}</p>
                  <div className="flex flex-wrap gap-2">
                    {group.items.map((phoneme) => (
                      <button
                        key={phoneme.symbol}
                        type="button"
                        title={`${phoneme.symbol} — ${phoneme.name}`}
                        onClick={() => insertPhoneme(phoneme.symbol)}
                        disabled={busy}
                        className="flex h-12 items-center gap-2 rounded-md border border-border/30 bg-white px-3 font-mono text-xl text-primary transition-colors hover:border-primary hover:bg-primary hover:text-white disabled:opacity-50"
                      >
                        <span>{phoneme.symbol}</span>
                        <span className="font-sans text-[10px] opacity-55">{phoneme.name}</span>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
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
            {inputSource !== "text" && (
              <button
                type="button"
                onClick={() => selectTab(inputSource)}
                title="Generate speaks this layer"
                className="inline-flex items-center gap-1.5 rounded-full border border-border/40 bg-white px-2.5 py-1 text-primary shadow-sm transition-colors hover:border-primary"
              >
                <span className="h-1.5 w-1.5 rounded-full bg-primary" aria-hidden />
                {sourceLabels[inputSource]}
              </button>
            )}
          </div>
          <Button onClick={createVoice} disabled={busy || !synthesisInput.trim()} className="h-12 px-8 text-sm shadow-xl shadow-primary/5 transition-transform hover:scale-[1.01]">
          {busy ? (
            <span className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              Generating...
            </span>
          ) : (
            <span className="flex items-center gap-2">
              <Play className="h-3.5 w-3.5 fill-current" />
              Generate
            </span>
          )}
        </Button>
        </div>
      </div>
    </Card>
  );
}
