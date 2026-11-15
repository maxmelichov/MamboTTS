//! The training labels' view of a phoneme string, for BlueV3 release bundles.
//!
//! A port of `blue_onnx/ipa_text.py` as it ships with notmax123/BlueV3-onnx
//! (`bluetts/blue-release.patch`). The model was trained on IPA passed through
//! exactly this normalisation and on ids wrapped in `<lang>` ... `</lang>`
//! (the vocab characters `<`, the code letters, `>` and `/`), sent identically
//! to the duration predictor, the text encoder and the vector field. Keep it in
//! step with the bundle rather than improving it here: a front end that drops
//! the tags renders wrong durations (9.75 s instead of 3.62 s for one sentence
//! in the release's own measurement).

use std::sync::OnceLock;

use regex::Regex;
use unicode_normalization::UnicodeNormalization;

/// Python's `\w` on `str`: letters, numbers and the underscore. The `regex`
/// crate's own `\w` also takes combining marks, which would change where a
/// word-final `ts` ends after NFD.
const WORD: &str = r"\p{L}\p{N}_";

fn emoji() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(concat!(
            "[\u{1f600}-\u{1f64f}\u{1f300}-\u{1f5ff}\u{1f680}-\u{1f6ff}\u{1f700}-\u{1f77f}",
            "\u{1f780}-\u{1f7ff}\u{1f800}-\u{1f8ff}\u{1f900}-\u{1f9ff}\u{1fa00}-\u{1fa6f}",
            "\u{1fa70}-\u{1faff}\u{2600}-\u{26ff}\u{2700}-\u{27bf}\u{1f1e6}-\u{1f1ff}]+",
        ))
        .expect("valid emoji regex")
    })
}

fn symbols() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new("[\u{2665}\u{2606}\u{2661}\u{a9}]").expect("valid regex"))
}

fn affricates() -> &'static [(Regex, &'static str); 4] {
    static RE: OnceLock<[(Regex, &'static str); 4]> = OnceLock::new();
    RE.get_or_init(|| {
        let re = |pattern: &str| Regex::new(pattern).expect("valid affricate regex");
        [
            // `ts(?=[^\w]|$)` in the reference. The lookahead is spelled as a
            // capture that is written back: what follows a word-final `ts` is
            // not a word character, so it can never start the next match.
            (
                re(&format!("(?i)t\u{361}s|ts(?P<after>[^{WORD}]|$)")),
                "\u{2a6}${after}",
            ),
            (re("(?i)t\u{361}\u{283}|t\u{283}"), "\u{2a7}"),
            (re("(?i)d\u{361}\u{292}|d\u{292}"), "\u{2a4}"),
            (re("(?i)d\u{361}z|dz"), "\u{2a3}"),
        ]
    })
}

fn whitespace() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[\s\x1c-\x1f]+").expect("valid regex"))
}

/// Python's `str.isspace`: Unicode whitespace plus the separators
/// U+001C to U+001F, which Rust does not count.
fn is_python_space(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}

fn language_tag() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // The site's `LANG_TAG_RE`, `/<\/?\w+>/g`: JavaScript's `\w` is ASCII.
    RE.get_or_init(|| Regex::new("</?[A-Za-z0-9_]+>").expect("valid tag regex"))
}

/// Marks that already end a chunk, so no period is added after them.
fn is_terminal(character: char) -> bool {
    matches!(
        character,
        '.' | '!'
            | '?'
            | ';'
            | ':'
            | ','
            | '\''
            | '"'
            | ')'
            | ']'
            | '}'
            | '\u{2026}'
            | '\u{3002}'
            | '\\'
            | '\u{300f}'
            | '\u{3011}'
            | '\u{3009}'
            | '\u{300b}'
            | '\u{203a}'
            | '\u{bb}'
    )
}

/// Remove the G2P's inline `<xx>` routing tags; the chunk is wrapped once in
/// its own language instead.
pub(crate) fn strip_language_tags(ipa: &str) -> String {
    language_tag().replace_all(ipa, "").into_owned()
}

/// `normalize_ipa(text, lang)` of the release: NFD, emoji removed,
/// typographic quotes and dashes folded, no space before punctuation,
/// affricate ligatures, and a final `.` when the text has no closing mark.
/// Hebrew writes `r` as `ʁ`, `g` as `ɡ` and keeps /ts/ as two letters.
pub(crate) fn normalize_ipa(text: &str, lang: &str) -> String {
    normalize_ipa_with(text, lang, true)
}

/// [`normalize_ipa`], with the final `.` only when `final_period` (the
/// site's `normalizeIpa(text, lang, finalPeriod)`; an inline span's IPA view
/// is taken without it).
pub(crate) fn normalize_ipa_with(text: &str, lang: &str, final_period: bool) -> String {
    let mut text: String = text.trim_matches(is_python_space).nfd().collect();
    text = emoji().replace_all(&text, "").into_owned();
    for (from, to) in [
        ("\u{201c}", "\""),
        ("\u{201d}", "\""),
        ("\u{2018}", "'"),
        ("\u{2019}", "'"),
        ("\u{b4}", "'"),
        ("`", "'"),
        ("\u{2013}", "-"),
        ("\u{2011}", "-"),
        ("\u{2014}", "-"),
        ("_", " "),
        ("[", " "),
        ("]", " "),
        ("|", " "),
        ("^", " "),
        ("$", " "),
    ] {
        text = text.replace(from, to);
    }
    text = symbols().replace_all(&text, "").into_owned();
    for mark in [",", ".", "!", "?", ";", ":", "'"] {
        text = text.replace(&format!(" {mark}"), mark);
    }
    // The backtick pair of the reference can no longer occur: every backtick
    // became an apostrophe above.
    for (pair, single) in [("\"\"", "\""), ("''", "'")] {
        while text.contains(pair) {
            text = text.replace(pair, single);
        }
    }
    let hebrew = lang == "he";
    if hebrew {
        text = text.replace('r', "\u{281}").replace('g', "\u{261}");
    }
    for (pattern, ligature) in affricates() {
        text = pattern.replace_all(&text, *ligature).into_owned();
    }
    if hebrew {
        text = text.replace('\u{2a6}', "ts");
    }
    text = whitespace()
        .replace_all(&text, " ")
        .trim_matches(is_python_space)
        .to_owned();
    if final_period && !text.chars().next_back().is_some_and(is_terminal) {
        text.push('.');
    }
    text
}

/// The letters of a language code as the tags spell them: `en-US` is `en`.
pub(crate) fn tag_code(lang: &str) -> String {
    lang.to_lowercase()
        .split('-')
        .next()
        .unwrap_or_default()
        .chars()
        .filter(char::is_ascii_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected strings are the output of the reference `normalize_ipa` in
    // `blue_ipa.py` (BlueV3 Space, same file as the bundle's ipa_text.py).
    #[test]
    fn matches_the_reference_normaliser() {
        for (lang, input, expected) in [
            ("en", "həlˈoʊ wˈɜːld", "həlˈoʊ wˈɜːld."),
            ("en", "  həlˈoʊ ,  wˈɜːld !  ", "həlˈoʊ, wˈɜːld!"),
            (
                "en",
                "tʃˈɜːtʃ dʒˈʌdʒ kˈæts ænd dˈɔɡz",
                "ʧˈɜːʧ ʤˈʌʤ kˈæʦ ænd dˈɔɡz.",
            ),
            ("en", "t͡sunˈɑːmi kˈæts.", "ʦunˈɑːmi kˈæʦ."),
            ("de", "tsˈaɪt ʃtˈɛts, zˈats", "tsˈaɪt ʃtˈɛʦ, zˈaʦ."),
            (
                "he",
                "ʃalˈom tsaʁˈiχ gadˈol ramˈa",
                "ʃalˈom tsaʁˈiχ ɡadˈol ʁamˈa.",
            ),
            ("he", "ʔˈets. kˈots", "ʔˈets. kˈots."),
            ("en", "“kwˈoʊt” — ɪts ‘ɪt’", "\"kwˈoʊt\" - ɪʦ'ɪt'"),
            ("en", "a_b [c] d|e", "a b c d e."),
            ("es", "¿kˈe tˈal?", "¿kˈe tˈal?"),
            ("en", "wˈʌn 😀 tˈuː ♥", "wˈʌn tˈuː."),
            ("it", "pˈittsa", "pˈittsa."),
            ("it", "pˈits", "pˈiʦ."),
            ("en", "ɪts ˈoʊkeɪ…", "ɪʦ ˈoʊkeɪ…"),
            ("en", "é", "e\u{301}."),
        ] {
            assert_eq!(normalize_ipa(input, lang), expected, "{lang}: {input}");
        }
    }

    #[test]
    fn a_span_view_has_no_final_period() {
        assert_eq!(normalize_ipa_with("həlˈoʊ ,  wˈɜːld", "en", false), "həlˈoʊ, wˈɜːld");
        assert_eq!(normalize_ipa_with("kˈæts", "he", false), "kˈæts");
    }

    #[test]
    fn information_separators_are_whitespace_as_in_python() {
        assert_eq!(normalize_ipa("\u{1c}a\u{1d}b\u{1f}", "en"), "a b.");
        assert_eq!(normalize_ipa("a \u{1e} ,b", "en"), "a ,b.");
    }

    #[test]
    fn a_stress_mark_after_ts_is_a_letter_as_in_python() {
        // `ˈ` is a modifier letter, so `ts` before it is not word-final.
        assert_eq!(normalize_ipa("ʔatsˈuv", "en"), "ʔatsˈuv.");
        // A combining mark is not a word character in Python's `\w`.
        assert_eq!(normalize_ipa("ats\u{329}a", "en"), "a\u{2a6}\u{329}a.");
    }

    #[test]
    fn routing_tags_are_removed_and_codes_are_letters() {
        assert_eq!(
            strip_language_tags("<he>ʃalˈom</he> <en>həlˈoʊ</en>"),
            "ʃalˈom həlˈoʊ"
        );
        assert_eq!(tag_code("en-US"), "en");
        assert_eq!(tag_code("HE"), "he");
    }
}
