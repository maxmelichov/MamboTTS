import { Loader2, Play } from "lucide-react";
import { useRef, useState } from "react";
import { cn } from "../../lib/classNames";
import { Button, Card } from "../../components/ui";

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
  { label: "Pronunciation", marks: [["ֽ", "Meteg"], ["֫", "Hatama"]] },
] as const;

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

type EditorCardProps = {
  busy: boolean;
  text: string;
  setText: (text: string) => void;
  language: string;
  hebrewG2pEngine: string;
  diacritics: string;
  setDiacritics: (text: string) => void;
  addDiacritics: () => Promise<void>;
  advancedMode: boolean;
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
  hebrewG2pEngine,
  diacritics,
  setDiacritics,
  addDiacritics,
  advancedMode,
  phonemes,
  setPhonemes,
  convertToPhonemes,
  createVoice,
}: EditorCardProps) {
  const [tab, setTab] = useState<"text" | "diacritics" | "phonemes">("text");
  const phonemeInput = useRef<HTMLTextAreaElement>(null);
  const [converting, setConverting] = useState(false);
  const [editorError, setEditorError] = useState("");
  const [selectedLetter, setSelectedLetter] = useState<number | null>(null);
  const isHebrew = language === "he" || (language === "auto" && /[\u0590-\u05ff]/.test(text));
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

  async function showRenikudOutput() {
    setConverting(true);
    setEditorError("");
    try {
      await convertToPhonemes();
      setTab("phonemes");
    } catch (error) {
      setEditorError(String(error));
    } finally {
      setConverting(false);
    }
  }

  async function openPhonemes() {
    setTab("phonemes");
    if (!phonemes && text.trim()) await showRenikudOutput();
  }

  async function openDiacritics() {
    setTab("diacritics");
    if (!diacritics && text.trim()) {
      setConverting(true);
      setEditorError("");
      try { await addDiacritics(); } catch (error) { setEditorError(String(error)); } finally { setConverting(false); }
    }
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
    setDiacritics(chars.join(""));
    setSelectedLetter(selectedLetter);
  }

  const selectedCharacter = selectedLetter === null ? "" : Array.from(diacritics)[selectedLetter] ?? "";
  const diacriticLetters = hebrewLetterClusters(diacritics);
  const diacriticWords = groupHebrewWords(diacriticLetters);

  return (
    <Card className="relative overflow-hidden p-0 shadow-xl border-none">
      {advancedMode && (
        <div className="flex items-center gap-2 border-b border-border/10 bg-background/10 px-8 pt-5">
          <button
            type="button"
            onClick={() => setTab("text")}
            className={cn("border-b-2 px-3 pb-3 text-[10px] font-bold uppercase tracking-[0.16em] transition-colors", tab === "text" ? "border-primary text-primary" : "border-transparent text-secondary/40 hover:text-primary")}
          >
            Text
          </button>
          {hebrewG2pEngine === "phonikud" && isHebrew && (
            <button type="button" onClick={() => void openDiacritics()} className={cn("border-b-2 px-3 pb-3 text-[10px] font-bold uppercase tracking-[0.16em] transition-colors", tab === "diacritics" ? "border-primary text-primary" : "border-transparent text-secondary/40 hover:text-primary")}>Diacritics</button>
          )}
          <button
            type="button"
            onClick={() => void openPhonemes()}
            className={cn("border-b-2 px-3 pb-3 text-[10px] font-bold uppercase tracking-[0.16em] transition-colors", tab === "phonemes" ? "border-primary text-primary" : "border-transparent text-secondary/40 hover:text-primary")}
          >
            Phonemes
          </button>
        </div>
      )}
      {tab === "text" || !advancedMode ? (
        <textarea
          id="text"
          value={text}
          placeholder={textPlaceholder}
          dir={isHebrew ? "rtl" : "ltr"}
          lang={isHebrew ? "he" : language === "auto" ? undefined : language}
          onChange={(event) => setText(event.currentTarget.value)}
          disabled={busy}
          className="min-h-[320px] w-full resize-none bg-white p-8 text-left text-lg font-medium leading-relaxed text-primary outline-none placeholder:text-secondary/20"
        />
      ) : tab === "diacritics" ? (
        <div className="bg-white p-8">
          <div className="mb-4 flex items-center justify-between gap-3">
            <div><p className="text-sm font-semibold text-primary">Vocalized Hebrew</p><p className="text-xs text-secondary/55">Phonikud adds niqqud and phonetic marks before IPA conversion.</p></div>
            <Button variant="outline" onClick={openDiacritics} disabled={busy || converting || !text.trim()} className="h-9 px-3 text-xs">{converting ? "Adding…" : "Refresh from text"}</Button>
          </div>
          <textarea value={diacritics} onChange={(event) => setDiacritics(event.currentTarget.value)} dir="rtl" lang="he" disabled={busy} className="min-h-48 w-full resize-y rounded-lg border border-border/50 bg-background/30 p-4 text-lg leading-relaxed text-primary outline-none focus:border-primary/50" />
          {editorError && <p className="mt-3 text-xs text-red-600">{editorError}</p>}
          <div className="mt-5 space-y-3">
            <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-secondary/40">Pick a letter in your vocalized text</p>
            <div dir="rtl" className="flex flex-wrap items-center gap-3 rounded-lg border border-border/20 bg-background/20 p-3 leading-loose">
              {diacriticWords.map((word, wordIndex) => (
                <div key={wordIndex} className="flex items-center rounded-lg border border-border/20 bg-white p-1 shadow-sm">
                  {word.map((cluster, index) => "start" in cluster ? (
                    <button key={`${cluster.start}-${index}`} type="button" onClick={() => setSelectedLetter(cluster.start)} className={cn("grid h-11 min-w-9 place-items-center rounded-md px-1.5 text-2xl transition-colors", selectedLetter === cluster.start ? "bg-primary text-white" : "text-primary hover:bg-primary/10")}>{cluster.value}</button>
                  ) : <span key={`symbol-${index}`} className="px-0.5 text-xl text-secondary/50">{cluster.value}</span>)}
                </div>
              ))}
              {!diacritics && <span className="p-2 text-xs text-secondary/50">Add diacritics from the text first.</span>}
            </div>
            {selectedCharacter && (
              <div className="space-y-3 rounded-lg border border-border/20 bg-background/20 p-3">
                <p className="text-sm font-semibold text-primary">Editing <span className="text-2xl">{selectedCharacter}</span></p>
                {diacriticGroups.map((group, groupIndex) => (
                  <div key={group.label}>
                    <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-secondary/45">{group.label}</p>
                    <div className="flex flex-wrap gap-2">
                      {group.marks.map(([mark, name]) => (
                        <button key={mark} type="button" title={name} onClick={() => changeDiacritic(mark, groupIndex === 0)} className="flex h-12 items-center gap-2 rounded-md border border-border/30 bg-white px-3 text-primary transition-colors hover:border-primary hover:bg-primary hover:text-white">
                          <span className="text-xl">{selectedCharacter}{mark}</span><span className="text-[10px] opacity-55">{name}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="bg-white p-8">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-sm font-semibold text-primary">Model IPA input</p>
              <p className="text-xs text-secondary/55">IPA output for the selected language. It is sent to BlueTTS without re-phonemizing.</p>
            </div>
            <Button variant="outline" onClick={showRenikudOutput} disabled={busy || converting || !text.trim()} className="h-9 px-3 text-xs">
              {converting ? "Converting..." : "Refresh from text"}
            </Button>
          </div>
          <textarea
            ref={phonemeInput}
            value={phonemes}
            placeholder="IPA phonemes appear here..."
            onChange={(event) => setPhonemes(event.currentTarget.value)}
            disabled={busy}
            className="min-h-32 w-full resize-y rounded-lg border border-border/50 bg-background/30 p-4 font-mono text-lg leading-relaxed text-primary outline-none focus:border-primary/50"
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
      <div className="flex items-center justify-between border-t border-border/10 bg-background/10 px-8 py-5">
        <div className="flex items-center gap-4 text-[10px] font-bold uppercase tracking-[0.2em] text-secondary opacity-40">
          <span className={cn("transition-colors", (advancedMode && phonemes ? phonemes : text).length > 500 ? "text-amber-600 opacity-100" : "")}>{(advancedMode && phonemes ? phonemes : text).length} Characters</span>
          {advancedMode && phonemes && <span>IPA input</span>}
        </div>
        <Button onClick={createVoice} disabled={busy || !(advancedMode && phonemes ? phonemes : text).trim()} className="h-12 px-8 text-sm shadow-xl shadow-primary/5 transition-transform hover:scale-[1.01]">
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
    </Card>
  );
}
