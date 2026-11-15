//! Phonemes into chunks, as the BlueTTS site cuts them (`js/bluetts-text.js`,
//! `splitSentences` and `ChunkPacker` in its default `sentence` mode).

use crate::handling::{is_js_space, js_trim};
use crate::ipa::normalize_ipa;
use crate::phonemize::strip_lang_tags;

/// A sentence of fewer words than this is merged into the next one.
pub(crate) const FRAGMENT_WORDS: usize = 3;
/// `chunk_text`'s `max_len`: a chunk's code points, tags not counted.
pub(crate) const MAX_CHUNK_CHARS: usize = 300;
/// `_CUT_MARKS`: where a sentence too long for a chunk is cut first.
const CUT_MARKS: &str = ",;:";
/// A clause cut leaves at least this many code points before it (BlueV3
/// `site_flow.py` `CUT_MIN_CHARS`; NOT the site's).
const CUT_MIN_CHARS: usize = 24;

/// Samples of `seconds` of silence, as the site counts a pause
/// (`Math.trunc(seconds * sampleRate)`).
pub(crate) fn silence_samples(sample_rate: u32, seconds: f64) -> usize {
    (seconds.max(0.0) * f64::from(sample_rate)).trunc() as usize
}

pub(crate) fn is_py_word_char(c: char) -> bool {
    let mut buf = [0u8; 4];
    regex!(r"^[\p{L}\p{N}_]$").is_match(c.encode_utf8(&mut buf))
}

/// The lookbehinds of the site's `SENTENCE_SPLIT_RE` (blue_onnx chunk_text):
/// no break after these abbreviations or after a one-letter initial.
fn ends_with_abbreviation(before: &[char]) -> bool {
    const ABBREVIATIONS: [&str; 19] = [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Ph.D.", "etc.", "e.g.", "i.e.", "vs.",
        "Inc.", "Ltd.", "Co.", "Corp.", "St.", "Ave.", "Blvd.",
    ];
    for abbreviation in ABBREVIATIONS {
        let a: Vec<char> = abbreviation.chars().collect();
        if before.ends_with(&a) {
            return true;
        }
    }
    let n = before.len();
    n >= 2
        && before[n - 1] == '.'
        && before[n - 2].is_ascii_uppercase()
        && (n == 2 || !is_py_word_char(before[n - 3]))
}

/// `splitSentences`: break after `.`, `!` or `?` and the whitespace after it,
/// except after the abbreviations of [`ends_with_abbreviation`].
pub(crate) fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = js_trim(text).chars().collect();
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < chars.len() {
        if i > 0
            && is_js_space(chars[i])
            && matches!(chars[i - 1], '.' | '!' | '?')
            && !ends_with_abbreviation(&chars[..i])
        {
            out.push(chars[start..i].iter().collect::<String>());
            let mut j = i;
            while j < chars.len() && is_js_space(chars[j]) {
                j += 1;
            }
            start = j;
            i = j;
            continue;
        }
        i += 1;
    }
    out.push(chars[start..].iter().collect::<String>());
    out.retain(|s| !s.is_empty());
    out
}

/// Words in `s`, as `s.trim().split(/\s+/).filter(Boolean).length`.
fn words(s: &str) -> usize {
    s.split(is_js_space).filter(|w| !w.is_empty()).count()
}

/// `taggedLength`: the ids `encodeTagged` gives `ipa` (tags, IPA
/// normalisation and final `.` included), without a vocab.
pub(crate) fn tagged_length(ipa: &str, lang: &str) -> usize {
    let code = crate::ipa::tag_code(lang);
    normalize_ipa(&strip_lang_tags(ipa), lang).chars().count() + 2 * code.len() + 5
}

/// `ChunkPacker` in `sentence` mode: one sentence a chunk; a sentence of
/// fewer than [`FRAGMENT_WORDS`] words ("Hi.") joins the next one when both
/// fit. A sentence too long alone is cut as `chunk_text` cuts it
/// ([`ChunkPacker::cut`]).
pub(crate) struct ChunkPacker {
    lang: String,
    max_ids: usize,
    current: String,
}

impl ChunkPacker {
    pub(crate) fn new(lang: &str, max_ids: usize) -> Self {
        Self {
            lang: lang.to_owned(),
            max_ids,
            current: String::new(),
        }
    }

    /// `chunk_text`'s `ok()`: at most [`MAX_CHUNK_CHARS`] code points, and
    /// its ids within the window.
    fn fits(&self, s: &str) -> bool {
        s.chars().count() <= MAX_CHUNK_CHARS && tagged_length(s, &self.lang) <= self.max_ids
    }

    /// A sentence longer than the window, in pieces that fit (`_cut_sentence`,
    /// BlueTTS a6cc157): after the last comma, semicolon or colon whose piece
    /// fits, else at the last space whose piece fits, past a quarter of what
    /// is left; failing that the same before the quarter; failing that at the
    /// last code point that fits. A mark stays with its piece, a space goes.
    ///
    /// An addition for the bot (BlueV3 `site_flow.py`, NOT the site's): a
    /// `,;:` past the first [`CUT_MIN_CHARS`] code points is tried before a
    /// space in the last three quarters, so a sentence whose commas sit in
    /// its first quarter is not cut in the middle of a phrase.
    fn cut(&self, s: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut cps: Vec<char> = js_trim(s).chars().collect();
        let joined = |cps: &[char]| cps.iter().collect::<String>();
        while !cps.is_empty() && !self.fits(&joined(&cps)) {
            let lo = cps.len() / 4;
            let last = |marks: &str, floor: usize| -> Option<usize> {
                for k in (floor + 1..cps.len()).rev() {
                    if !marks.contains(cps[k]) {
                        continue;
                    }
                    let n = if cps[k] == ' ' { k } else { k + 1 };
                    if self.fits(&joined(&cps[..n])) {
                        return Some(n);
                    }
                }
                None
            };
            let found = [
                (CUT_MARKS, lo),
                (CUT_MARKS, lo.min(CUT_MIN_CHARS)),
                (" ", lo),
                (CUT_MARKS, 0),
                (" ", 0),
            ]
                .into_iter()
                .find_map(|(marks, floor)| last(marks, floor));
            let n = match found {
                Some(n) => n,
                None => {
                    let mut n = cps.len() - 1;
                    while n > 1 && !self.fits(&joined(&cps[..n])) {
                        n -= 1;
                    }
                    // A single code point always fits; never cut nothing.
                    n.max(1)
                }
            };
            let piece = joined(&cps[..n]);
            let piece = js_trim(&piece);
            if !piece.is_empty() {
                out.push(piece.to_owned());
            }
            cps = js_trim(&joined(&cps[n..])).chars().collect();
        }
        let rest = joined(&cps);
        let rest = js_trim(&rest);
        if !rest.is_empty() {
            out.push(rest.to_owned());
        }
        out
    }

    fn pieces(&self, sentence: &str) -> Vec<String> {
        if self.fits(sentence) {
            vec![sentence.to_owned()]
        } else {
            self.cut(sentence)
        }
    }

    /// Phonemized text (any number of sentences or paragraphs): the chunks
    /// it completes.
    pub(crate) fn add(&mut self, phonemes: &str) -> Vec<String> {
        let mut done = Vec::new();
        let paragraphs: Vec<&str> = regex!(&format!(
            r"\n{}*\n+",
            crate::handling::JS_S
        ))
        .split(js_trim(phonemes))
        .map(js_trim)
        .filter(|p| !p.is_empty())
        .collect();
        for (i, paragraph) in paragraphs.into_iter().enumerate() {
            if i > 0 && !self.current.is_empty() {
                done.push(js_trim(&self.current).to_owned());
                self.current.clear();
            }
            for sentence in split_sentences(paragraph) {
                let pieces = self.pieces(&sentence);
                for piece in &pieces {
                    let joined = if self.current.is_empty() {
                        piece.clone()
                    } else {
                        format!("{} {piece}", self.current)
                    };
                    // A held fragment joins this piece when both fit.
                    if !self.current.is_empty() && !self.fits(&joined) {
                        done.push(js_trim(&self.current).to_owned());
                        self.current = piece.clone();
                    } else {
                        self.current = joined;
                    }
                    // Hold whole sentences while together they are under
                    // FRAGMENT_WORDS words, for the next one; emit anything else.
                    let fragment = pieces.len() == 1 && words(&self.current) < FRAGMENT_WORDS;
                    if !fragment {
                        done.push(js_trim(&self.current).to_owned());
                        self.current.clear();
                    }
                }
            }
        }
        done
    }

    /// The held rest, if any.
    pub(crate) fn end(&mut self) -> Vec<String> {
        let rest = js_trim(&self.current).to_owned();
        self.current.clear();
        if rest.is_empty() { Vec::new() } else { vec![rest] }
    }
}

/// Phonemes into chunks of at most `max_ids` ids: every unit through one
/// packer, then its rest.
pub(crate) fn pack_chunks<'a>(
    units: impl IntoIterator<Item = &'a str>,
    lang: &str,
    max_ids: usize,
) -> Vec<String> {
    let mut packer = ChunkPacker::new(lang, max_ids);
    let mut chunks = Vec::new();
    for unit in units {
        chunks.extend(packer.add(unit));
    }
    chunks.extend(packer.end());
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentences_break_after_terminal_marks_but_not_abbreviations() {
        assert_eq!(
            split_sentences("Hello there. Dr. Smith is here! Is J. Doe? yes"),
            vec!["Hello there.", "Dr. Smith is here!", "Is J. Doe?", "yes"]
        );
        assert_eq!(split_sentences("  a.  b  "), vec!["a.", "b"]);
        assert_eq!(split_sentences("x...y. z"), vec!["x...y.", "z"]);
    }

    #[test]
    fn a_short_sentence_joins_the_next_one() {
        let chunks = pack_chunks(["hˈaɪ. ðɪs ɪz ə tˈɛst. ænd ə sˈɛkənd wˌʌn hˈɪɹ."], "en", 300);
        assert_eq!(
            chunks,
            vec!["hˈaɪ. ðɪs ɪz ə tˈɛst.", "ænd ə sˈɛkənd wˌʌn hˈɪɹ."]
        );
        // A fragment at the very end is held, then emitted.
        assert_eq!(pack_chunks(["ðɪs ɪz ə tˈɛst. jˈɛs."], "en", 300), vec![
            "ðɪs ɪz ə tˈɛst.",
            "jˈɛs."
        ]);
    }

    #[test]
    fn a_long_sentence_is_cut_to_fit_at_a_comma() {
        let long = format!("{}, {}.", "a ".repeat(100).trim(), "b ".repeat(100).trim());
        let chunks = pack_chunks([long.as_str()], "en", 300);
        assert!(chunks.len() >= 2, "{chunks:?}");
        for chunk in &chunks {
            assert!(tagged_length(chunk, "en") <= 300, "{chunk}");
        }
        assert!(chunks[0].ends_with(','), "{chunks:?}");
    }

    #[test]
    fn a_long_sentence_is_cut_after_a_semicolon_or_colon_past_a_quarter() {
        let long = format!("{}; {}: {}.", "a ".repeat(60).trim(), "b ".repeat(60).trim(), "c ".repeat(60).trim());
        let chunks = pack_chunks([long.as_str()], "en", 300);
        assert_eq!(chunks.len(), 2, "{chunks:?}");
        assert!(chunks[0].ends_with(':'), "{chunks:?}");
        assert!(chunks[1].starts_with('c'), "{chunks:?}");
    }

    #[test]
    fn a_chunk_holds_at_most_300_code_points() {
        // Few ids would fit, but not in 300 code points.
        let long = "a ".repeat(200);
        let chunks = pack_chunks([long.trim()], "en", 10_000);
        assert!(chunks.len() >= 2, "{chunks:?}");
        for chunk in &chunks {
            assert!(chunk.chars().count() <= MAX_CHUNK_CHARS, "{chunk}");
        }
    }

    #[test]
    fn the_budget_counts_tags_and_the_final_period() {
        // "<en>" + "ab" + "." + "</en>": 4 + 3 + 5.
        assert_eq!(tagged_length("ab", "en"), 12);
        assert_eq!(tagged_length("<he>ab.</he>", "en"), 12);
    }
}
