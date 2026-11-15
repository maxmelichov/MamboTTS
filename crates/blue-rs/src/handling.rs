//! Text preparation before G2P, exactly as the BlueTTS site does it
//! (`js/bluetts-text.js`, `prepareText`):
//!
//! - English, German, Spanish and Italian go to polyg2p as written (trimmed):
//!   it reads numbers, dates, money and abbreviations itself, and the model's
//!   labels were made from its reading of raw text.
//! - Hebrew goes through heb-tts-normalizer 0.2.0 at its defaults, and
//!   nothing else.
//! - Russian, which has no release front end, keeps the site's number reading
//!   (`prepareTextForSynthesis`, a port of blue_onnx `text_norm.py`) without
//!   its slow-span markers.
//!
//! The regexes keep the site's JavaScript semantics: `\d` and `\w` are ASCII,
//! `\s` is JavaScript's whitespace set, and Python's `\w` is spelled
//! `[\p{L}\p{N}_]` where the site spells it so.

use std::borrow::Cow;
use std::sync::OnceLock;

/// A `fancy_regex` (lookaround, backreferences) compiled once per process.
macro_rules! fregex {
    ($pattern:expr) => {{
        static RE: std::sync::OnceLock<fancy_regex::Regex> = std::sync::OnceLock::new();
        RE.get_or_init(|| fancy_regex::Regex::new(&$pattern).expect("valid regex"))
    }};
}

/// JavaScript's `\s`: WhiteSpace and LineTerminator (U+FEFF in, U+0085 out).
pub(crate) const JS_S: &str = r"[\t\n\x0B\x0C\r \x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}\x{feff}]";
/// JavaScript's `\S`.
const JS_NS: &str = r"[^\t\n\x0B\x0C\r \x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}\x{feff}]";
/// Python's `\w` on `str`, as the site spells it.
const PY_W: &str = r"\p{L}\p{N}_";

/// JavaScript's `String.prototype.trim` whitespace.
pub(crate) fn is_js_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n'
            | '\u{b}'
            | '\u{c}'
            | '\r'
            | ' '
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
            | '\u{feff}'
    )
}

/// `s.trim()` in JavaScript.
pub(crate) fn js_trim(s: &str) -> &str {
    s.trim_matches(is_js_space)
}

/// `text.replace(/\s+/g, ' ')` in JavaScript.
pub(crate) fn js_collapse_spaces(text: &str) -> String {
    regex!(&format!("{JS_S}+"))
        .replace_all(text, " ")
        .into_owned()
}

/// `canonicalLang`: `ge` is `de`, `en-us` is `en`, `ru-ru` is `ru`, lowercase.
pub fn canonical_lang(lang: &str) -> String {
    let lower = lang.to_lowercase();
    match lower.as_str() {
        "ge" => "de".to_owned(),
        "en-us" => "en".to_owned(),
        "ru-ru" => "ru".to_owned(),
        _ => lower,
    }
}

/// `prepareText(text, lang, heNormalize)`: the text a language's G2P reads.
pub fn prepare_text(text: &str, lang: &str) -> String {
    match canonical_lang(lang).as_str() {
        "ru" => {
            let text = prepare_text_for_synthesis_ru(text).replace(['【', '】'], " ");
            js_trim(&regex!(r"[ \t]+").replace_all(&text, " ")).to_owned()
        }
        "he" => js_trim(&hebrew_normalize(text)).to_owned(),
        _ => js_trim(text).to_owned(),
    }
}

/// The text a language's G2P reads, with or without the bot rules
/// (BlueV3 `site_flow.py`, `SiteFrontEnd.plan`). Without them,
/// [`prepare_text`] alone: the site exactly. With them, [`bot_prepare`].
pub fn prepare(text: &str, lang: &str, bot_rules: bool) -> String {
    if !bot_rules {
        return prepare_text(text, lang);
    }
    bot_prepare(text, lang)
}

/// `bot_prepare`, the golden `prepared` stage: pictographs and invisible
/// marks out, tags flattened and a one-line Hebrew list broken into lines;
/// then [`crate::bot::normalize_before`] on every piece in its own
/// language, [`prepare_text`] with spans prepared in their own language,
/// and [`crate::bot::normalize_after`] on every piece.
pub fn bot_prepare(text: &str, lang: &str) -> String {
    let lang = model_lang(lang).unwrap_or("en");
    let raw = break_inline_lists(
        &flatten_tags(&crate::bot::strip_invisible(&strip_pictographs(text))),
        lang,
    );
    let before = per_span(&raw, lang, crate::bot::normalize_before);
    per_span(&prepare_text_protected(&before, lang), lang, crate::bot::normalize_after)
}

/// `per_span`: `f(piece, lang)` on the text outside inline spans and
/// `f(inner, code)` on each span's content in the span's own language. The
/// spans `f` makes inside a span are kept flat (`<en>a </en><en>x at y</en>`).
fn per_span(text: &str, lang: &str, f: impl Fn(&str, &str) -> String) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for span in inline_lang_spans(text) {
        out.push_str(&f(&text[last..span.start], lang));
        last = span.end;
        let Some(code) = model_lang(&span.name) else {
            out.push_str(&text[span.start..span.end]);
            continue;
        };
        let inner = f(&text[span.content.0..span.content.1], code);
        let name = &span.name;
        let wrap = |piece: &str, out: &mut String| {
            if crate::bot::py_strip(piece).is_empty() {
                out.push_str(piece);
            } else {
                out.push_str(&format!("<{name}>{piece}</{name}>"));
            }
        };
        let mut l2 = 0;
        for nested in inline_lang_spans(&inner) {
            wrap(&inner[l2..nested.start], &mut out);
            out.push_str(&inner[nested.start..nested.end]);
            l2 = nested.end;
        }
        wrap(&inner[l2..], &mut out);
    }
    out.push_str(&f(&text[last..], lang));
    out
}

/// `prepare_text(text, lang, protect_spans=True)`: the text outside inline
/// `<xx>…</xx>` spans prepared in the base language and each span's content
/// in its own (the site's call ran heb-tts-normalizer over `<en>` spans too:
/// "<en>call 5</en>" became "<en>call חמש</en>").
pub fn prepare_text_protected(text: &str, lang: &str) -> String {
    let spans = inline_lang_spans(text);
    if spans.is_empty() {
        return prepare_text(text, lang);
    }
    let code = canonical_lang(lang);
    let keep_ws = |piece: &str, lg: &str| -> String {
        let stripped = crate::bot::py_strip(piece);
        if stripped.is_empty() {
            return piece.to_owned();
        }
        let core = prepare_text(piece, lg);
        let lead = if piece.starts_with(crate::bot::py_space) { " " } else { "" };
        let trail = if piece.ends_with(crate::bot::py_space) { " " } else { "" };
        format!("{lead}{core}{trail}")
    };
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for span in &spans {
        out.push_str(&keep_ws(&text[last..span.start], &code));
        let content = &text[span.content.0..span.content.1];
        let inner = match model_lang(&span.name) {
            Some(inner_lang) => keep_ws(content, inner_lang),
            None => content.to_owned(),
        };
        out.push_str(&format!("<{0}>{inner}</{0}>", span.name));
        last = span.end;
    }
    out.push_str(&keep_ws(&text[last..], &code));
    js_trim(&out).to_owned()
}

// ── the bot rules (NOT the site's; BlueV3 site_flow.py, tests/golden) ───────

/// The model's language for a tag name or request code (`model_lang`),
/// or `None`.
pub(crate) fn model_lang(lang: &str) -> Option<&'static str> {
    match lang.trim().to_lowercase().as_str() {
        "he" => Some("he"),
        "en" | "en-us" => Some("en"),
        "es" => Some("es"),
        "de" | "ge" => Some("de"),
        "it" => Some("it"),
        "ru" | "ru-ru" => Some("ru"),
        _ => None,
    }
}

/// `strip_pictographs`: what the site's `EMOJI_RE` misses. Every emoji
/// presentation character, a pictograph followed by U+FE0F, the variation
/// selectors, the zero-width joiner, the keycap mark and the tag characters
/// (subdivision flags) are removed; a run of them with the spaces and tabs
/// around it becomes one space. The vocab has none of them: each was a pad
/// id in the middle of a chunk.
///
/// `#` or `*` keycaps go whole first (a digit keycap keeps its digit), and
/// © ® ™ count as pictographs. A run becomes nothing instead of a space at
/// the start of the text or a line, at its end, or before a newline or a
/// closing mark (`done 🎉.` is `done.`).
pub fn strip_pictographs(text: &str) -> String {
    let text = regex!(r"[#*]\x{FE0F}?\x{20E3}").replace_all(text, "");
    let re = fregex!({
        let p = r"\p{Emoji_Presentation}|\p{Extended_Pictographic}(?=\x{FE0F})|[\x{FE0E}\x{FE0F}\x{200D}\x{20E3}]|[\x{E0020}-\x{E007F}]|[\x{A9}\x{AE}\x{2122}]";
        format!(r"[ \t]*(?:{p})(?:[ \t]*(?:{p}))*[ \t]*")
    });
    freplace_all(re, &text, |caps| {
        let m = caps.get(0).expect("whole match");
        let before = text[..m.start()].chars().next_back();
        let after = text[m.end()..].chars().next();
        let silent = before.is_none_or(|c| c == '\n')
            || after.is_none_or(|c| ".,!?;:…)]}»\"'”’\n".contains(c));
        if silent { String::new() } else { " ".into() }
    })
}

/// `flatten_tags`: inline tags made flat before anything reads them. A tag
/// that names no language (`<b>`) is dropped; nested language spans
/// (`<en>a <he>b</he> c</en>`) become a flat run of spans, each piece of
/// text inside its innermost language; each tag becomes one space; a `<`
/// or `>` that is not part of a tag (`3 < 5`) becomes a space.
pub fn flatten_tags(text: &str) -> String {
    let mut stack: Vec<&'static str> = Vec::new();
    let mut out = String::with_capacity(text.len());
    let emit = |piece: &str, stack: &[&str], out: &mut String| {
        let piece = piece.replace(['<', '>'], " ");
        if piece.is_empty() {
            return;
        }
        match stack.last() {
            Some(top) if !piece.trim().is_empty() => {
                out.push_str(&format!("<{top}>{piece}</{top}>"));
            }
            _ => out.push_str(&piece),
        }
    };
    let mut last = 0;
    for caps in regex!(r"</?([A-Za-z][A-Za-z0-9_-]*)>").captures_iter(text) {
        let m = caps.get(0).expect("whole match");
        emit(&text[last..m.start()], &stack, &mut out);
        last = m.end();
        let Some(code) = model_lang(&caps[1]) else {
            continue; // not a language: dropped
        };
        if m.as_str().starts_with("</") {
            if let Some(at) = stack.iter().rposition(|&open| open == code) {
                stack.truncate(at);
            } else {
                // A mismatched closer (`<he>…</en>`) closes the open span.
                stack.pop();
            }
        } else {
            stack.push(code);
        }
        out.push(' ');
    }
    emit(&text[last..], &stack, &mut out);
    out
}

/// `break_inline_lists`: a numbered list written on one line in Hebrew
/// (`1. קמח 2. סוכר 3. ביצים`) is put one item a line. heb-tts-normalizer
/// drops a list number at the start of a line but reads the later ones
/// inline, glued to the item before. Only markers numbered exactly 1, 2,
/// 3… (at least two) are touched.
pub fn break_inline_lists(text: &str, lang: &str) -> String {
    if canonical_lang(lang) != "he" {
        return text.to_owned();
    }
    let marks: Vec<(usize, u32)> = fregex!(r"(?m)(?:^|(?<=[ \t]))([0-9]{1,2})\.[ \t]+(?=[א-ת])")
        .captures_iter(text)
        .filter_map(Result::ok)
        .map(|caps| {
            let start = caps.get(0).expect("whole match").start();
            (start, caps[1].parse().unwrap_or(0))
        })
        .collect();
    if marks.len() < 2 || marks.iter().zip(1..).any(|(&(_, n), k)| n != k) {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len() + marks.len());
    let mut last = 0;
    for &(start, _) in &marks {
        let before = text[last..start].trim_end_matches([' ', '\t']);
        out.push_str(before);
        if !before.is_empty() && !before.ends_with('\n') {
            out.push('\n');
        }
        last = start;
    }
    out.push_str(&text[last..]);
    out
}

/// `f` over the text outside `<xx>…</xx>` spans and stray tags.
pub(crate) fn outside_spans(text: &str, mut f: impl FnMut(&str) -> String) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for m in span_or_tag_re().find_iter(text).filter_map(Result::ok) {
        out.push_str(&f(&text[last..m.start()]));
        out.push_str(m.as_str());
        last = m.end();
    }
    out.push_str(&f(&text[last..]));
    out
}

/// `tag_emails`: in Hebrew, English, German, Spanish and Italian an e-mail
/// address becomes its spoken English reading in `<en>…</en>`
/// ([`email_to_spoken_english`]), with a space at every digit/digit and
/// letter/digit boundary so digits are read one by one. Read as written,
/// Hebrew cut the address at the `@` and polyg2p's `.` split the chunk
/// inside it. Russian keeps the site's own e-mail reading.
pub fn tag_emails(text: &str, lang: &str) -> String {
    if !matches!(canonical_lang(lang).as_str(), "he" | "en" | "de" | "es" | "it") {
        return text.to_owned();
    }
    let re = fregex!(
        r"(?<![A-Za-z0-9._%+])(?<![A-Za-z0-9]-)[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![A-Za-z0-9])"
    );
    let spoken = |email: &str| -> String {
        freplace_all(
            fregex!(r"(?<=[0-9])(?=[0-9A-Za-z])|(?<=[A-Za-z])(?=[0-9])"),
            &email_to_spoken_english(email),
            |_| " ".into(),
        )
    };
    outside_spans(text, |t| {
        freplace_all(re, t, |caps| format!("<en>{}</en>", spoken(&caps[0])))
    })
}

/// heb-tts-normalizer 0.2.0 at its default reading style, the site's
/// `heNormalize` (its wasm build of the same crate).
fn hebrew_normalize(text: &str) -> String {
    static CONFIG: OnceLock<heb_tts_normalizer::Config> = OnceLock::new();
    heb_tts_normalizer::normalize(text, CONFIG.get_or_init(Default::default))
}

/// One `<xx>…</xx>` inline span (`/<(\w+)>([\s\S]*?)<\/\1>/g`): the whole
/// match, the tag name and the content, as byte ranges.
pub(crate) struct InlineSpan {
    pub start: usize,
    pub end: usize,
    pub name: String,
    pub content: (usize, usize),
}

pub(crate) fn inline_lang_spans(text: &str) -> Vec<InlineSpan> {
    fregex!(r"<([A-Za-z0-9_]+)>([\s\S]*?)</\1>")
        .captures_iter(text)
        .filter_map(Result::ok)
        .map(|caps| {
            let all = caps.get(0).expect("whole match");
            let content = caps.get(2).expect("content");
            InlineSpan {
                start: all.start(),
                end: all.end(),
                name: caps[1].to_owned(),
                content: (content.start(), content.end()),
            }
        })
        .collect()
}

/// `hasInlineLang`: the text has an inline `<xx>…</xx>` span, which
/// `phonemize` reads span by span.
pub fn has_inline_lang(text: &str) -> bool {
    !inline_lang_spans(text).is_empty()
}

// ── tag_latin_runs (blue_onnx/ipa_text.py, BlueTTS a6cc157) ─────────────────

/// Python's `\d`: any decimal digit.
const PY_D: &str = r"\p{Nd}";
/// Python's `\s` (`str.isspace`), as the site spells it.
const PY_S: &str = r"[\t\n\x0B\x0C\r \x1C-\x1F\x{85}\x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}]";
/// Latin letters, with the Latin-1 and Extended-A/B accented ones.
const LATIN_L: &str = r"A-Za-z\x{c0}-\x{d6}\x{d8}-\x{f6}\x{f8}-\x{24f}";

/// `LATIN_RUN_RE`: a Latin-script word (apostrophes, hyphens and dots
/// inside it, digits glued on), then the words and numbers after it across
/// whitespace.
fn latin_run_re() -> &'static fancy_regex::Regex {
    fregex!({
        let word = format!(
            r"{PY_D}*[{LATIN_L}][{LATIN_L}{PY_D}]*(?:['’.\-][{LATIN_L}{PY_D}]+)*"
        );
        format!(
            r"{word}(?:{PY_S}+(?:{word}|{PY_D}+(?:[.,]{PY_D}+)*(?![{LATIN_L}{PY_D}])))*"
        )
    })
}

/// `SPAN_OR_TAG_RE`: a tag pair (kept whole) or a stray tag.
fn span_or_tag_re() -> &'static fancy_regex::Regex {
    fregex!(format!(r"<([{PY_W}]+)>[\s\S]*?</\1>|</?[{PY_W}]+>"))
}

/// `englishRun`: a Latin run as English polyg2p reads it, accents dropped
/// ("café" as "cafe") and an all-capitals tail after a lower-case letter
/// apart ("BlueTTS" as "Blue TTS", "ChatGPT" as "Chat GPT").
fn english_run(run: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    let decomposed: String = run.nfd().collect();
    let bare = regex!(r"\p{Mn}").replace_all(&decomposed, "");
    fregex!(r"(?<=[a-z])(?=[A-Z]{2,}(?![A-Za-z]))")
        .replace_all(&bare, " ")
        .into_owned()
}

/// `tagLatinRuns`: in Hebrew text each run of Latin-script words goes in
/// `<en>…</en>`, so English polyg2p reads it. Existing spans and stray tags
/// are left alone, and a run is tagged whether or not the text has a Hebrew
/// letter. Other languages: as is.
pub fn tag_latin_runs(text: &str, lang: &str) -> String {
    if canonical_lang(lang) != "he" {
        return text.to_owned();
    }
    let tag = |t: &str| -> String {
        latin_run_re()
            .replace_all(t, |caps: &fancy_regex::Captures| {
                format!("<en>{}</en>", english_run(&caps[0]))
            })
            .into_owned()
    };
    let mut out = String::new();
    let mut last = 0;
    for m in span_or_tag_re().find_iter(text).filter_map(Result::ok) {
        out.push_str(&tag(&text[last..m.start()]));
        out.push_str(m.as_str());
        last = m.end();
    }
    out.push_str(&tag(&text[last..]));
    out
}

// ── prepareTextForSynthesis (text_norm.py), for Russian ─────────────────────

const PERCENT_WORD_RU: &str = "процентов";
const PERCENT_WORD_EN: &str = "percent";
const RATIO_WORD_RU: &str = "к";
const RATIO_WORD_EN: &str = "to";
const PLUS_WORD_RU: &str = "плюс";

/// `EMOJI_RE` of the site.
pub(crate) fn emoji_re() -> &'static regex::Regex {
    regex!(concat!(
        "[\u{1f600}-\u{1f64f}\u{1f300}-\u{1f5ff}\u{1f680}-\u{1f6ff}\u{1f700}-\u{1f77f}",
        "\u{1f780}-\u{1f7ff}\u{1f800}-\u{1f8ff}\u{1f900}-\u{1f9ff}\u{1fa00}-\u{1fa6f}",
        "\u{1fa70}-\u{1faff}\u{2600}-\u{26ff}\u{2700}-\u{27bf}\u{1f1e6}-\u{1f1ff}]+",
    ))
}

/// `stripEmoji`: pictographs out, whitespace collapsed, trimmed.
pub(crate) fn strip_emoji(text: &str) -> String {
    let text = emoji_re().replace_all(text, " ");
    js_trim(&js_collapse_spaces(&text)).to_owned()
}

/// `replace_all` with a `fancy_regex`, a match at a time.
pub(crate) fn freplace_all<F>(re: &fancy_regex::Regex, text: &str, mut f: F) -> String
where
    F: FnMut(&fancy_regex::Captures) -> String,
{
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for caps in re.captures_iter(text).filter_map(Result::ok) {
        let m = caps.get(0).expect("whole match");
        out.push_str(&text[last..m.start()]);
        out.push_str(&f(&caps));
        last = m.end();
    }
    out.push_str(&text[last..]);
    out
}

/// The first match only (a JavaScript regex without the `g` flag).
fn freplace_first(re: &fancy_regex::Regex, text: &str, with: &str) -> String {
    match re.find(text) {
        Ok(Some(m)) => format!("{}{with}{}", &text[..m.start()], &text[m.end()..]),
        _ => text.to_owned(),
    }
}

/// `text.split(re)` with `re` one capture group around the whole pattern:
/// the pieces between matches and the matches, in order, with a flag that
/// says which is which.
fn split_keep<'a>(re: &regex::Regex, text: &'a str) -> Vec<(&'a str, bool)> {
    let mut parts = Vec::new();
    let mut last = 0;
    for m in re.find_iter(text) {
        parts.push((&text[last..m.start()], false));
        parts.push((m.as_str(), true));
        last = m.end();
    }
    parts.push((&text[last..], false));
    parts
}

fn mark_slow_segment(inner: &str) -> String {
    format!("【{inner}】")
}

fn maybe_slow(inner: String, inside_en: bool) -> String {
    if inside_en {
        inner
    } else {
        mark_slow_segment(&inner)
    }
}

/// `_map_en_spans`: `f(segment, lang, inside_en)` over the parts of `text`,
/// `<en>…</en>` spans with their own language.
fn map_en_spans<F>(text: &str, lang: &str, mut f: F) -> String
where
    F: FnMut(&str, &str, bool) -> String,
{
    split_keep(regex!(r"(?i)<en>[\s\S]*?</en>"), text)
        .into_iter()
        .map(|(part, is_en)| {
            if is_en {
                format!("<en>{}</en>", f(&part[4..part.len() - 5], "en", true))
            } else {
                f(part, lang, false)
            }
        })
        .collect()
}

/// num2words for the languages the Russian path reaches: Russian
/// cardinals; `None` where the site's num2words would give a spelling this
/// port does not carry (English inside an `<en>` span of Russian text) or
/// none at all.
fn spoken_number(value: f64, lang: &str, is_float: bool) -> Option<String> {
    if !value.is_finite() || !(0.0..1e15).contains(&value) {
        return None;
    }
    match lang {
        "ru" => ru_cardinal(value, is_float),
        _ => None,
    }
}

fn spoken_digits(digits: &str, lang: &str) -> String {
    digits
        .chars()
        .filter(char::is_ascii_digit)
        .filter_map(|d| spoken_number(f64::from(d.to_digit(10).unwrap_or(0)), lang, false))
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_repeated_punctuation(text: &str) -> String {
    let text = text.replace('…', ",");
    let text = freplace_all(fregex!(r"(?<![0-9])\.{2,}(?![0-9])"), &text, |_| ",".into());
    let text = regex!(r"!+").replace_all(&text, "!").into_owned();
    let text = regex!(r"\?+").replace_all(&text, "?").into_owned();
    let text = regex!(&format!(r",(?:{JS_S}*,)+"))
        .replace_all(&text, ",")
        .into_owned();
    let text = freplace_all(
        fregex!(format!(r"{JS_S}*[,;:]+{JS_S}*(?=[.!?])")),
        &text,
        |_| String::new(),
    );
    freplace_all(
        fregex!(format!(r"(?<=[.!?]){JS_S}*[,;:]+{JS_S}*")),
        &text,
        |_| " ".into(),
    )
}

fn normalize_common_text(text: &str) -> String {
    let text = regex!(&format!(r"(^|{JS_S})#{{1,6}}{JS_S}*"))
        .replace_all(text, "$1")
        .into_owned();
    let text = regex!(&format!(r"{JS_S}*[(\[{{]{JS_S}*"))
        .replace_all(&text, ", ")
        .into_owned();
    let text = regex!(&format!(r"{JS_S}*[)\]}}]{JS_S}*"))
        .replace_all(&text, ", ")
        .into_owned();
    let text = normalize_repeated_punctuation(&text);
    freplace_all(
        fregex!(format!(r"(?i)(?<![{PY_W}])anymore(?![{PY_W}])")),
        &text,
        |caps| {
            let first = caps[0].chars().next().unwrap_or('a');
            if first.to_lowercase().next() != Some(first) {
                "Any more".into()
            } else {
                "any more".into()
            }
        },
    )
}

fn expand_dialogue_quotes(text: &str) -> String {
    let text = freplace_first(
        fregex!(format!(r#"(?<={JS_NS}){JS_S}*["“„”]{JS_S}*$"#)),
        text,
        ".",
    );
    let text = freplace_all(
        fregex!(format!(r#"{JS_S}*:?{JS_S}*["“„”]{JS_S}*(?={JS_NS})"#)),
        &text,
        |_| ", ".into(),
    );
    freplace_all(
        fregex!(format!(r#"(?<={JS_NS}){JS_S}*["“„”]"#)),
        &text,
        |_| ", ".into(),
    )
}

/// `emailToSpokenEnglish`.
pub(crate) fn email_to_spoken_english(email: &str) -> String {
    let (local, domain) = email.split_once('@').unwrap_or((email, ""));
    let local = regex!(r"[._]+").replace_all(local, " dot ");
    let local = regex!(r"-+").replace_all(&local, " dash ");
    let local = regex!(r"\++").replace_all(&local, " plus ");
    let spell = |label: &str| -> String {
        let n = label.encode_utf16().count();
        if n > 0 && n <= 2 && regex!(r"^\p{L}+$").is_match(label) {
            label
                .chars()
                .map(String::from)
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            label.to_owned()
        }
    };
    let parts: Vec<String> = domain
        .split('.')
        .filter(|p| !p.is_empty())
        .map(spell)
        .collect();
    let spoken = format!("{local} at {}", parts.join(" dot "));
    js_trim(&js_collapse_spaces(&spoken)).to_owned()
}

fn expand_emails(text: &str, lang: &str) -> String {
    map_en_spans(text, lang, |seg, _, inside_en| {
        if inside_en {
            return seg.to_owned();
        }
        regex!(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
            .replace_all(seg, |caps: &regex::Captures| {
                format!("<en>{}</en>", email_to_spoken_english(&caps[0]))
            })
            .into_owned()
    })
}

fn should_spell_alphanumeric_token(token: &str) -> bool {
    if token.is_empty() || token.contains('@') {
        return false;
    }
    if !regex!(r"^[A-Za-z0-9][A-Za-z0-9\-_/]*$").is_match(token) {
        return false;
    }
    let letters = token.chars().filter(char::is_ascii_alphabetic).count();
    let digits = token.chars().filter(char::is_ascii_digit).count();
    if letters == 0 || digits == 0 {
        return false;
    }
    if letters < 2 && digits < 2 {
        return false;
    }
    token.chars().filter(|c| !matches!(c, '-' | '_' | '/')).count() >= 3
}

fn spell_alphanumeric_code(code: &str, lang: &str) -> String {
    let mut letters = Vec::new();
    let mut groups = Vec::new();
    for seg in regex!(r"[-_/]+").split(code) {
        if seg.is_empty() {
            continue;
        }
        letters.extend(
            seg.chars()
                .filter(char::is_ascii_alphabetic)
                .map(|c| c.to_ascii_uppercase().to_string()),
        );
        let d: String = seg.chars().filter(char::is_ascii_digit).collect();
        if !d.is_empty() {
            groups.push(d);
        }
    }
    let digit_parts: Vec<String> = groups
        .iter()
        .map(|g| spoken_digits(g, lang))
        .filter(|p| !p.is_empty())
        .collect();
    if letters.is_empty() && digit_parts.is_empty() {
        return String::new();
    }
    let letters_block = if letters.is_empty() {
        String::new()
    } else {
        format!("<en>{}</en>", letters.join(" "))
    };
    let mut digits_block = digit_parts.join(" , ");
    if !digits_block.is_empty() {
        digits_block.push_str(" .");
    }
    let joined = [letters_block, digits_block]
        .into_iter()
        .filter(|b| !b.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    mark_slow_segment(&joined)
}

fn expand_alphanumeric_codes(text: &str, lang: &str) -> String {
    map_en_spans(text, lang, |seg, l, inside_en| {
        if inside_en {
            return seg.to_owned();
        }
        freplace_all(
            fregex!(r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*(?![A-Za-z0-9])"),
            seg,
            |caps| {
                let token = &caps[0];
                if !should_spell_alphanumeric_token(token) {
                    return token.to_owned();
                }
                let spelled = spell_alphanumeric_code(token, l);
                if spelled.is_empty() {
                    token.to_owned()
                } else {
                    spelled
                }
            },
        )
    })
}

fn expand_list_markers(text: &str, lang: &str) -> String {
    map_en_spans(text, lang, |seg, l, inside_en| {
        if inside_en {
            return seg.to_owned();
        }
        freplace_all(
            fregex!(format!(
                r#"(?<![0-9/])([0-9]{{1,2}})\.{JS_S}+(?=[\x{{0590}}-\x{{05ff}}A-Za-z"'(<])"#
            )),
            seg,
            |caps| {
                let k: f64 = caps[1].parse().unwrap_or(0.0);
                match spoken_number(k, l, false) {
                    Some(word) => format!("{word}. "),
                    None => caps[0].to_owned(),
                }
            },
        )
    })
}

fn expand_plus_sign(text: &str, lang: &str) -> String {
    map_en_spans(text, lang, |seg, l, inside_en| {
        if inside_en {
            return seg.to_owned();
        }
        let word = if l == "ru" { PLUS_WORD_RU } else { "plus" };
        regex!(&format!(r"{JS_S}+\+{JS_S}+"))
            .replace_all(seg, format!(" {word} ").as_str())
            .into_owned()
    })
}

fn expand_times(text: &str, lang: &str) -> String {
    map_en_spans(text, lang, |seg, l, inside_en| {
        freplace_all(
            fregex!(r"(?<![0-9])([01]?[0-9]|2[0-3]):([0-5][0-9])(?![0-9])"),
            seg,
            |caps| {
                let h: f64 = caps[1].parse().unwrap_or(0.0);
                let mi: f64 = caps[2].parse().unwrap_or(0.0);
                let Some(hour) = spoken_number(h, l, false) else {
                    return caps[0].to_owned();
                };
                if mi == 0.0 {
                    return maybe_slow(hour, inside_en);
                }
                let Some(minute) = spoken_number(mi, l, false) else {
                    return caps[0].to_owned();
                };
                maybe_slow(format!("{hour} {minute}"), inside_en)
            },
        )
    })
}

fn expand_dates(text: &str, lang: &str) -> String {
    map_en_spans(text, lang, |seg, l, inside_en| {
        freplace_all(
            fregex!(r"(?<![0-9])([0-3]?[0-9])[/.]([01]?[0-9])[/.]([0-9]{2}|[0-9]{4})(?![0-9])"),
            seg,
            |caps| {
                let day: f64 = caps[1].parse().unwrap_or(0.0);
                let month: f64 = caps[2].parse().unwrap_or(0.0);
                if !((1.0..=31.0).contains(&day) && (1.0..=12.0).contains(&month)) {
                    return caps[0].to_owned();
                }
                let mut year: f64 = caps[3].parse().unwrap_or(0.0);
                if caps[3].len() == 2 {
                    year += if year < 70.0 { 2000.0 } else { 1900.0 };
                }
                let Some(year_word) = spoken_number(year, l, false) else {
                    return caps[0].to_owned();
                };
                if l == "ru" {
                    let day_word = spoken_number(day, "ru", false).unwrap_or_default();
                    let month_word = spoken_number(month, "ru", false).unwrap_or_default();
                    return maybe_slow(format!("{day_word} {month_word} {year_word}"), inside_en);
                }
                // Month names and ordinals of the other languages are not
                // reached from the Russian path.
                caps[0].to_owned()
            },
        )
    })
}

fn expand_percent_symbols(text: &str, lang: &str) -> String {
    map_en_spans(text, lang, |seg, l, _| {
        let word = if l == "ru" {
            PERCENT_WORD_RU
        } else {
            PERCENT_WORD_EN
        };
        let seg = regex!(&format!(r"([0-9]+(?:[.,][0-9]+)?){JS_S}*%"))
            .replace_all(seg, format!("${{1}} {word}").as_str())
            .into_owned();
        seg.replace('%', &format!(" {word} "))
    })
}

fn expand_ratios(text: &str, lang: &str) -> String {
    map_en_spans(text, lang, |seg, l, _| {
        let word = if l == "ru" {
            RATIO_WORD_RU
        } else {
            RATIO_WORD_EN
        };
        freplace_all(
            fregex!(format!(
                r"(?<![0-9])([0-9]+){JS_S}*:{JS_S}*([0-9]+)(?![0-9])"
            )),
            seg,
            |caps| format!("{} {word} {}", &caps[1], &caps[2]),
        )
    })
}

fn expand_numbers(text: &str, lang: &str) -> String {
    split_keep(regex!(r"(?i)<en>[\s\S]*?</en>|【[^】]*】"), text)
        .into_iter()
        .map(|(part, protected)| -> Cow<'_, str> {
            if protected {
                return Cow::Borrowed(part);
            }
            let grouped = freplace_all(
                fregex!(format!(
                    r"(?<![{PY_W}.,])[0-9]{{1,3}}(?:,[0-9]{{3}})+(?![{PY_W}.,])"
                )),
                part,
                |caps| {
                    let value: f64 = caps[0].replace(',', "").parse().unwrap_or(f64::NAN);
                    spoken_number(value, lang, false)
                        .filter(|w| !w.is_empty())
                        .unwrap_or_else(|| caps[0].to_owned())
                },
            );
            Cow::Owned(freplace_all(
                fregex!(format!(r"(?<![{PY_W}])[0-9]+(?:[.,][0-9]+)?(?![{PY_W}])")),
                &grouped,
                |caps| {
                    let raw = &caps[0];
                    let is_float = raw.contains(['.', ',']);
                    let value: f64 = raw.replacen(',', ".", 1).parse().unwrap_or(f64::NAN);
                    spoken_number(value, lang, is_float)
                        .filter(|w| !w.is_empty())
                        .unwrap_or_else(|| raw.to_owned())
                },
            ))
        })
        .collect()
}

/// `stripSilentSeparatorTokens`: dashes between scripts and stand-alone
/// dashes and colons become spaces.
pub(crate) fn strip_silent_separator_tokens(text: &str) -> String {
    let text = freplace_all(
        fregex!(r"(?<=[\x{0590}-\x{05ff}])[-–—‑]+(?=[A-Za-z0-9])"),
        text,
        |_| " ".into(),
    );
    let text = freplace_all(
        fregex!(r"(?<=[A-Za-z0-9])[-–—‑]+(?=[\x{0590}-\x{05ff}])"),
        &text,
        |_| " ".into(),
    );
    let text = freplace_all(
        fregex!(format!(r"(?<![A-Za-z]){JS_S}*[-–—‑]+{JS_S}*(?![A-Za-z])")),
        &text,
        |_| " ".into(),
    );
    let text = freplace_all(
        fregex!(format!(r"(?<![0-9]){JS_S}*:+{JS_S}*(?![0-9])")),
        &text,
        |_| " ".into(),
    );
    js_trim(&js_collapse_spaces(&text)).to_owned()
}

/// `prepareTextForSynthesis(text, 'ru')`: the Hebrew-only steps are no-ops
/// for Russian and are left out; slow spans are marked with `【…】`.
pub(crate) fn prepare_text_for_synthesis_ru(text: &str) -> String {
    let lang = "ru";
    let text = regex!(&format!(r"{JS_S}*\[[^\]]*\]{JS_S}*")).replace_all(text, " ");
    let text = strip_emoji(&text);
    let text = normalize_common_text(&text);
    let text = expand_dialogue_quotes(&text);
    let text = expand_emails(&text, lang);
    let text = expand_alphanumeric_codes(&text, lang);
    let text = expand_list_markers(&text, lang);
    let text = expand_plus_sign(&text, lang);
    let text = expand_times(&text, lang);
    let text = expand_dates(&text, lang);
    let text = expand_percent_symbols(&text, lang);
    let text = expand_ratios(&text, lang);
    let text = expand_numbers(&text, lang);
    strip_silent_separator_tokens(&text)
}

// ── num2words, Russian (lang_RU.py, nominative cardinals) ──────────────────

const RU_ONES: [&[&str]; 10] = [
    &["ноль"],
    &["один", "одна"],
    &["два", "две"],
    &["три", "три"],
    &["четыре", "четыре"],
    &["пять"],
    &["шесть"],
    &["семь"],
    &["восемь"],
    &["девять"],
];
const RU_TENS: [&str; 10] = [
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
];
const RU_TWENTIES: [&str; 10] = [
    "",
    "",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
];
const RU_HUNDREDS: [&str; 10] = [
    "",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
];
const RU_THOUSAND_STEMS: [&str; 4] = ["миллион", "миллиард", "триллион", "квадриллион"];
const RU_BITNESS: [(&str, &str); 9] = [
    ("десятая", "десятых"),
    ("сотая", "сотых"),
    ("тысячная", "тысячных"),
    ("десятитысячная", "десятитысячных"),
    ("стотысячная", "стотысячных"),
    ("миллионная", "миллионных"),
    ("десятимиллионная", "десятимиллионных"),
    ("стомиллионная", "стомиллионных"),
    ("миллиардная", "миллиардных"),
];

fn ru_pluralize(n: u64, forms: [&str; 3]) -> String {
    if (11..=14).contains(&(n % 100)) {
        return forms[2].to_owned();
    }
    match n % 10 {
        1 => forms[0].to_owned(),
        2..=4 => forms[1].to_owned(),
        _ => forms[2].to_owned(),
    }
}

fn ru_thousand_forms(i: usize) -> [String; 3] {
    if i == 1 {
        return ["тысяча".into(), "тысячи".into(), "тысяч".into()];
    }
    let stem = RU_THOUSAND_STEMS[i - 2];
    [stem.to_owned(), format!("{stem}а"), format!("{stem}ов")]
}

/// `utils.splitbyx(str(n), 3)`.
fn split_by_3(n: u64) -> Vec<u64> {
    let s = n.to_string();
    if s.len() <= 3 {
        return vec![n];
    }
    let mut out = Vec::new();
    let start = s.len() % 3;
    if start > 0 {
        out.push(s[..start].parse().unwrap_or(0));
    }
    let mut i = start;
    while i < s.len() {
        out.push(s[i..i + 3].parse().unwrap_or(0));
        i += 3;
    }
    out
}

pub(crate) fn ru_int(n: u64, feminine: bool) -> String {
    if n == 0 {
        return "ноль".into();
    }
    let mut words: Vec<String> = Vec::new();
    let chunks = split_by_3(n);
    let mut i = chunks.len();
    for x in chunks {
        i -= 1;
        if x == 0 {
            continue;
        }
        let (ones, tens, hundreds) = ((x % 10) as usize, ((x / 10) % 10) as usize, ((x / 100) % 10) as usize);
        if hundreds > 0 {
            words.push(RU_HUNDREDS[hundreds].into());
        }
        if tens > 1 {
            words.push(RU_TWENTIES[tens].into());
        }
        if tens == 1 {
            words.push(RU_TENS[ones].into());
        } else if ones > 0 {
            // Thousands are feminine.
            let f = if i == 1 { true } else if i == 0 { feminine } else { false };
            let forms = RU_ONES[ones];
            words.push(if f && forms.len() > 1 { forms[1] } else { forms[0] }.into());
        }
        if i > 0 {
            let forms = ru_thousand_forms(i);
            words.push(ru_pluralize(
                x,
                [forms[0].as_str(), forms[1].as_str(), forms[2].as_str()],
            ));
        }
    }
    words.join(" ")
}

fn ru_cardinal(value: f64, is_float: bool) -> Option<String> {
    // JavaScript's String(number) for these magnitudes is Rust's Display.
    let mut n = format!("{value}");
    if is_float && !n.contains('.') {
        n.push_str(".0");
    }
    let Some((left, right)) = n.split_once('.') else {
        return Some(ru_int(n.parse().ok()?, false));
    };
    let (unit, units) = *RU_BITNESS.get(right.len().checked_sub(1)?)?;
    let l: u64 = left.parse().ok()?;
    let bit = if right.ends_with('1') && !right.ends_with("11") {
        unit
    } else {
        units
    };
    Some(format!(
        "{} {} {} {bit}",
        ru_int(l, true),
        ru_pluralize(l, ["целая", "целых", "целых"]),
        ru_int(right.parse().ok()?, true)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latin_languages_are_only_trimmed() {
        assert_eq!(
            prepare_text("  Dr. Smith paid $25 (cash).\n", "en"),
            "Dr. Smith paid $25 (cash)."
        );
    }

    #[test]
    fn russian_numbers_are_read_as_the_site_reads_them() {
        assert_eq!(ru_int(21, false), "двадцать один");
        assert_eq!(ru_int(2_021, false), "две тысячи двадцать один");
        assert_eq!(
            ru_cardinal(3.5, true).unwrap(),
            "три целых пять десятых"
        );
    }

    #[test]
    fn inline_spans_need_a_matching_close() {
        assert!(has_inline_lang("שלום <en>hello</en>"));
        assert!(has_inline_lang("<b>bold</b>"));
        assert!(!has_inline_lang("<en>hello</he>"));
        assert!(!has_inline_lang("a < b > c"));
    }

    #[test]
    fn latin_runs_in_hebrew_are_tagged_as_the_site_tags_them() {
        for (text, expected) in [
            ("אני משתמש ב-ChatGPT כל יום", "אני משתמש ב-<en>Chat GPT</en> כל יום"),
            ("קניתי iPhone 15 Pro חדש", "קניתי <en>iPhone 15 Pro</en> חדש"),
            ("יש Wi-Fi בבית", "יש <en>Wi-Fi</en> בבית"),
            ("שתיתי café הבוקר", "שתיתי <en>cafe</en> הבוקר"),
            ("נסו את BlueTTS.", "נסו את <en>Blue TTS</en>."),
            ("גלשו ל-www.google.com עכשיו", "גלשו ל-<en>www.google.com</en> עכשיו"),
            ("קובץ MP3 ו-GPT-4", "קובץ <en>MP3</en> ו-<en>GPT-4</en>"),
            ("Hello world.", "<en>Hello world</en>."),
            ("גרסה v2 1.5x", "גרסה <en>v2 1</en>.<en>5x</en>"),
            ("הוא אמר <en>hello there</en> ואז OK", "הוא אמר <en>hello there</en> ואז <en>OK</en>"),
            ("<b>bold</b> text", "<b>bold</b> <en>text</en>"),
        ] {
            assert_eq!(tag_latin_runs(text, "he"), expected, "{text}");
        }
        assert_eq!(tag_latin_runs("Hello world.", "en"), "Hello world.");
    }
}
