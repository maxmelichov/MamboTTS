//! Text handling around the model: grapheme folding, niqqud stripping, the
//! long-input chunker and the raw-Hebrew backstop.

use crate::pychars;

/// First and last letter of the Hebrew alphabet, alef and tav.
pub(crate) const ALEF: char = '\u{05D0}';
pub(crate) const TAF: char = '\u{05EA}';

/// `max_position_embeddings` (2048) minus CLS + SEP. Measured: 2046 is fine,
/// 2047 raises.
pub const SAFE_CHUNK_CHARS: usize = 2046;

const SENTENCE_ENDS: &str = ".!?;\n\r…\u{05C3}";
const CLAUSE_ENDS: &str = ",:\u{2013}\u{2014}";

pub(crate) fn is_hebrew(c: char) -> bool {
    (ALEF..=TAF).contains(&c)
}

/// Fold the geresh and gershayim variants onto ASCII `'` and `"`.
pub fn normalize_graphemes(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{05F3}' | '\'' | '`' | '\u{00B4}' => '\'',
            '\u{05F4}' | '"' => '"',
            other => other,
        })
        .collect()
}

/// Hebrew points and cantillation marks: niqqud, dagesh, shin/sin dots, te'amim.
///
/// The maqaf (U+05BE), paseq (U+05C0), sof pasuq (U+05C3) and nun hafukha
/// (U+05C6) are punctuation, not marks on a letter, so they are left in place —
/// the decoder passes them through and the chunker splits on the sentence ones.
pub fn is_niqqud(c: char) -> bool {
    matches!(
        c,
        '\u{0591}'
            ..='\u{05BD}'
                | '\u{05BF}'
                | '\u{05C1}'
                | '\u{05C2}'
                | '\u{05C4}'
                | '\u{05C5}'
                | '\u{05C7}'
    )
}

/// Drop niqqud and cantillation, leaving the bare consonantal skeleton.
///
/// The model is trained on unpointed Hebrew: a combining point is not in its
/// character vocabulary, so pointed input both tokenizes to `[UNK]` and leaks
/// the raw mark into the output. Input must be NFD.
pub fn strip_niqqud(text: &str) -> String {
    text.chars().filter(|&c| !is_niqqud(c)).collect()
}

/// ASCII stand-ins to canonical IPA.
pub fn normalize_ipa(text: &str) -> String {
    text.replace('g', "ɡ").replace('x', "χ").replace('r', "ʁ")
}

/// What to do when raw Hebrew letters survive into an IPA string.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum HebrewLeak {
    /// Return the output as it is.
    Ignore,
    /// Report on stderr (the default, as upstream).
    #[default]
    Warn,
    /// Fail.
    Raise,
}

/// Fail loudly when raw Hebrew letters survive into an IPA string.
pub fn check_no_raw_hebrew(
    output: String,
    text: &str,
    mode: HebrewLeak,
    where_: &str,
) -> anyhow::Result<String> {
    if mode == HebrewLeak::Ignore {
        return Ok(output);
    }
    let leaked: Vec<char> = output.chars().filter(|&c| is_hebrew(c)).collect();
    if leaked.is_empty() {
        return Ok(output);
    }
    let mut sample = String::new();
    for c in &leaked {
        if !sample.contains(*c) && sample.chars().count() < 20 {
            sample.push(*c);
        }
    }
    let message = format!(
        "[{where_}] RAW HEBREW IN OUTPUT: {} Hebrew letter(s) ({sample}) survived into what \
         should be an IPA string — the tail of the input was never transcribed. input={} chars, \
         output={} chars.",
        leaked.len(),
        text.chars().count(),
        output.chars().count()
    );
    if mode == HebrewLeak::Raise {
        anyhow::bail!(message);
    }
    eprintln!("WARNING: {message}");
    Ok(output)
}

fn char_len(text: &str) -> usize {
    text.chars().count()
}

/// Split after every run of `terminators`, keeping the terminator and the
/// whitespace that follows it with the piece before them.
fn split_after<'a>(text: &'a str, terminators: &str) -> Vec<&'a str> {
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < text.len() {
        let c = text[i..].chars().next().expect("char boundary");
        if terminators.contains(c) {
            let mut j = i + c.len_utf8();
            while let Some(next) = text[j..].chars().next() {
                if terminators.contains(next) {
                    j += next.len_utf8();
                } else {
                    break;
                }
            }
            while let Some(next) = text[j..].chars().next() {
                if pychars::is_space(next) {
                    j += next.len_utf8();
                } else {
                    break;
                }
            }
            pieces.push(&text[start..j]);
            start = j;
            i = j;
        } else {
            i += c.len_utf8();
        }
    }
    if start < text.len() {
        pieces.push(&text[start..]);
    }
    pieces
}

/// Split after every run of whitespace.
fn split_words(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < text.len() {
        let c = text[i..].chars().next().expect("char boundary");
        if pychars::is_space(c) {
            let mut j = i;
            while let Some(next) = text[j..].chars().next() {
                if pychars::is_space(next) {
                    j += next.len_utf8();
                } else {
                    break;
                }
            }
            pieces.push(&text[start..j]);
            start = j;
            i = j;
        } else {
            i += c.len_utf8();
        }
    }
    if start < text.len() {
        pieces.push(&text[start..]);
    }
    pieces
}

fn pack(pieces: &[&str], limit: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    for piece in pieces {
        if !current.is_empty() && char_len(&current) + char_len(piece) > limit {
            out.push(std::mem::take(&mut current));
            current.push_str(piece);
        } else {
            current.push_str(piece);
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn take_chars(text: &str, start: usize, count: usize) -> String {
    text.chars().skip(start).take(count).collect()
}

/// Split into decode windows of at most `limit` characters, losslessly.
///
/// `split_for_decode(t).concat() == t`, and a text within `limit` comes back as
/// a single window.
pub fn split_for_decode(text: &str, limit: usize, warn_on_hard_cut: bool) -> Vec<String> {
    assert!(limit > 0, "limit must be positive");
    if char_len(text) <= limit {
        return vec![text.to_owned()];
    }

    let mut out: Vec<String> = Vec::new();
    for sentence_chunk in pack(&split_after(text, SENTENCE_ENDS), limit) {
        if char_len(&sentence_chunk) <= limit {
            out.push(sentence_chunk);
            continue;
        }
        for clause_chunk in pack(&split_after(&sentence_chunk, CLAUSE_ENDS), limit) {
            if char_len(&clause_chunk) <= limit {
                out.push(clause_chunk);
                continue;
            }
            for word_chunk in pack(&split_words(&clause_chunk), limit) {
                let length = char_len(&word_chunk);
                if length <= limit {
                    out.push(word_chunk);
                    continue;
                }
                if warn_on_hard_cut {
                    eprintln!(
                        "WARNING: [long-input] a {length}-character run with no whitespace \
                         exceeds the {limit}-character decode window; cutting it mid-token."
                    );
                }
                for start in (0..length).step_by(limit) {
                    out.push(take_chars(&word_chunk, start, limit));
                }
            }
        }
    }

    debug_assert_eq!(out.concat(), text, "split_for_decode is not lossless");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_window() {
        assert_eq!(split_for_decode("שלום עולם", 2046, true), ["שלום עולם"]);
    }

    #[test]
    fn long_text_splits_losslessly_on_sentences() {
        let sentence = "שלום עולם מה שלומך היום. ";
        let text = sentence.repeat(200);
        let windows = split_for_decode(&text, 500, true);
        assert!(windows.len() > 1);
        assert!(windows.iter().all(|w| w.chars().count() <= 500));
        assert_eq!(windows.concat(), text);
    }

    #[test]
    fn a_long_unbroken_run_is_cut_at_the_window() {
        let text = "א".repeat(50);
        let windows = split_for_decode(&text, 20, false);
        assert_eq!(
            windows
                .iter()
                .map(|w| w.chars().count())
                .collect::<Vec<_>>(),
            [20, 20, 10]
        );
        assert_eq!(windows.concat(), text);
    }

    #[test]
    fn graphemes_and_niqqud_fold_as_upstream() {
        assert_eq!(normalize_graphemes("ג׳ירפה צה״ל"), "ג'ירפה צה\"ל");
        // Curly quotes are not part of the upstream mapping.
        assert_eq!(
            normalize_graphemes("\u{201C}a\u{201D}"),
            "\u{201C}a\u{201D}"
        );
        assert_eq!(strip_niqqud("שָׁלוֹם־עוֹלָם׃"), "שלום־עולם׃");
    }

    #[test]
    fn the_leak_backstop_reports_and_raises() {
        assert!(
            check_no_raw_hebrew("ʃalˈom".to_owned(), "שלום", HebrewLeak::Raise, "test").is_ok()
        );
        assert!(check_no_raw_hebrew("שלום".to_owned(), "שלום", HebrewLeak::Raise, "test").is_err());
        assert!(check_no_raw_hebrew("שלום".to_owned(), "שלום", HebrewLeak::Ignore, "test").is_ok());
    }
}
