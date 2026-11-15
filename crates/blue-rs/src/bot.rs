//! The bot's text rules, round 2 (NOT the site's; BlueV3 `site_flow.py`
//! `normalize_before` / `normalize_after`, pinned for every port by
//! `tests/golden/cases.jsonl`): phone numbers in every language, cards,
//! fractions, scores, versions, minus signs, arithmetic, ranges, ISO dates,
//! degrees, units, Russian ordinals, dates, years, currency and spaced
//! thousands, leading zeros, digits glued to Latin letters, "פרק 1",
//! URLs, @/# handles, Markdown and bullets, typographic quotes and Cyrillic
//! inside Hebrew.
//!
//! The regexes are the reference's, copied as written. The reference runs
//! Python's `regex` module, whose `\d`, `\w` and `\s` are Unicode, as the
//! Rust ones are. Where it looks behind over a variable length (which
//! `fancy_regex` cannot), the look-behind is checked by hand at each
//! candidate, which is what the regex engine does: a candidate whose
//! look-behind fails is skipped and the scan goes on one character later.

use crate::handling::{
    freplace_all, js_collapse_spaces, js_trim, outside_spans, ru_int,
};

/// A `fancy_regex` (lookaround, backreferences) compiled once per process.
macro_rules! fregex {
    ($pattern:expr) => {{
        static RE: std::sync::OnceLock<fancy_regex::Regex> = std::sync::OnceLock::new();
        RE.get_or_init(|| fancy_regex::Regex::new(&$pattern).expect("valid regex"))
    }};
}

type Caps<'a> = fancy_regex::Captures<'a>;

/// Python's `str.isspace`.
pub(crate) fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python's `str.strip()`.
pub(crate) fn py_strip(s: &str) -> &str {
    s.trim_matches(py_space)
}

/// `_sub_outside`: `re.sub` over the text outside spans and stray tags.
fn sub_outside(
    re: &fancy_regex::Regex,
    text: &str,
    mut f: impl FnMut(&Caps) -> String,
) -> String {
    outside_spans(text, |piece| freplace_all(re, piece, &mut f))
}

/// `re.sub` with a look-behind the engine cannot express: `behind(piece, at)`
/// says whether the look-behind holds at a candidate's start.
fn sub_behind(
    re: &fancy_regex::Regex,
    piece: &str,
    behind: impl Fn(&str, usize) -> bool,
    mut f: impl FnMut(&Caps) -> String,
) -> String {
    let mut out = String::with_capacity(piece.len());
    let (mut last, mut pos) = (0, 0);
    while pos <= piece.len() {
        let Ok(Some(caps)) = re.captures_from_pos(piece, pos) else {
            break;
        };
        let m = caps.get(0).expect("whole match");
        if behind(piece, m.start()) {
            out.push_str(&piece[last..m.start()]);
            out.push_str(&f(&caps));
            last = m.end();
            pos = if m.end() > m.start() {
                m.end()
            } else {
                next_char(piece, m.end())
            };
        } else {
            pos = next_char(piece, m.start());
        }
    }
    out.push_str(&piece[last..]);
    out
}

fn next_char(s: &str, at: usize) -> usize {
    at + s[at..].chars().next().map_or(1, char::len_utf8)
}

fn prev_char(s: &str, at: usize) -> Option<char> {
    s[..at].chars().next_back()
}

/// Ends `before` with 1..=`max` (at least `min`) letters of `set`, preceded
/// by the start or a character `edge` accepts? (`(?:^|edge)[set]{min,max}`)
fn ends_with_prefix(before: &str, set: &str, min: usize, max: usize, edge: impl Fn(char) -> bool) -> bool {
    let chars: Vec<char> = before.chars().rev().take(max + 1).collect();
    (min..=max).any(|k| {
        chars.len() >= k
            && chars[..k].iter().all(|c| set.contains(*c))
            && chars.get(k).is_none_or(|&c| edge(c))
    })
}

/// `_spaced_digits`: the digits of `s`, one by one.
fn spaced_digits(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_numeric())
        .map(String::from)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Python's `int(s)` for the small numbers these rules compare.
fn int(s: &str) -> u64 {
    s.parse().unwrap_or(u64::MAX)
}

const HE_DIGITS: [&str; 10] = ["אפס", "אחת", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע"];
const EN_NUM: [&str; 10] = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
const PLUS_WORD_RU: &str = "плюс";

fn minus_word(lang: &str) -> &'static str {
    match lang {
        "he" => "מינוס",
        "es" => "menos",
        "it" => "meno",
        "ru" => "минус",
        _ => "minus",
    }
}

fn point_word(lang: &str) -> &'static str {
    match lang {
        "he" => "נקודה",
        "en" => "point",
        "de" => "Punkt",
        "ru" => "точка",
        _ => "punto",
    }
}

fn range_word(lang: &str) -> &'static str {
    match lang {
        "he" => "עד",
        "en" => "to",
        "de" => "bis",
        "ru" => "до",
        _ => "a",
    }
}

fn ratio_word(lang: &str) -> &'static str {
    match lang {
        "he" => "ל-",
        "en" => "to",
        "de" => "zu",
        _ => "a",
    }
}

fn math_word(lang: &str, op: &str) -> &'static str {
    let table: [&str; 5] = match lang {
        "he" => ["פלוס", "מינוס", "כפול", "חלקי", "שווה"],
        "de" => ["plus", "minus", "mal", "geteilt durch", "gleich"],
        "es" => ["más", "menos", "por", "entre", "igual a"],
        "it" => ["più", "meno", "per", "diviso", "uguale a"],
        "ru" => ["плюс", "минус", "умножить на", "разделить на", "равно"],
        _ => ["plus", "minus", "times", "divided by", "equals"],
    };
    match op {
        "+" => table[0],
        "−" => table[1],
        "×" => table[2],
        "÷" => table[3],
        _ => table[4],
    }
}

// ── before prepareText ──────────────────────────────────────────────────────

/// `strip_invisible`: bidi marks and isolates, zero-width spaces and joiners,
/// word joiners, the BOM and the soft hyphen out; C0 controls other than
/// tab, newline and CR (and DEL) become a space.
pub fn strip_invisible(text: &str) -> String {
    let text = regex!(r"[\x{AD}\x{200B}-\x{200F}\x{202A}-\x{202E}\x{2060}-\x{2064}\x{2066}-\x{2069}\x{FEFF}]")
        .replace_all(text, "");
    regex!(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
        .replace_all(&text, " ")
        .into_owned()
}

/// `strip_markdown`: links, emphasis, code, headings and list bullets.
fn strip_markdown(text: &str) -> String {
    let text = freplace_all(fregex!(r"\[([^\]\n]+)\]\([^)\s]*\)"), text, |c| c[1].to_owned());
    let text = freplace_all(fregex!(r"(\*\*|__|`)(?=\S)([^\n]*?\S)\1"), &text, |c| c[2].to_owned());
    let text = freplace_all(
        fregex!(r"(?<![\w*])\*(?=\S)([^*\n]*?\S)\*(?![\w*])"),
        &text,
        |c| c[1].to_owned(),
    );
    let text = regex!(r"(?m)^[ \t]*#{1,6}[ \t]+").replace_all(&text, "");
    regex!(r"(?m)^[ \t]*[•·▪◦●‣∙][ \t]*")
        .replace_all(&text, "")
        .into_owned()
}

/// `spoken_url`: scheme, query and fragment dropped, `www` as `w w w`, `.`
/// as `dot`, `/` as `slash`, `-` as `dash`, labels of one or two letters
/// spelled, digits as English words.
pub fn spoken_url(url: &str) -> String {
    let url = regex!(r"(?i)^https?://").replace(url, "");
    let url = regex!(r"[?#]").splitn(&url, 2).next().unwrap_or("").trim_end_matches('/');
    let (host, path) = url.split_once('/').unwrap_or((url, ""));
    let spell = |label: &str| -> String {
        if label.to_lowercase() == "www" {
            "w w w".to_owned()
        } else if (1..=2).contains(&label.chars().count()) && regex!(r"^\p{L}+$").is_match(label) {
            label.chars().map(String::from).collect::<Vec<_>>().join(" ")
        } else {
            label.replace('-', " dash ")
        }
    };
    let mut words = host
        .split('.')
        .filter(|p| !p.is_empty())
        .map(spell)
        .collect::<Vec<_>>()
        .join(" dot ");
    for seg in path.split('/').filter(|s| !s.is_empty()) {
        let seg = regex!(r"[._]+").replace_all(seg, " dot ");
        let seg = regex!(r"-+").replace_all(&seg, " dash ");
        words.push_str(&format!(" slash {seg}"));
    }
    let words = freplace_all(
        fregex!(r"(?<=[0-9])(?=[0-9A-Za-z])|(?<=[A-Za-z])(?=[0-9])"),
        js_trim(&js_collapse_spaces(&words)),
        |_| " ".into(),
    );
    let words = regex!(r"[0-9]").replace_all(&words, |c: &regex::Captures| {
        format!(" {} ", EN_NUM[c[0].parse::<usize>().unwrap_or(0)])
    });
    py_strip(&js_collapse_spaces(&words)).to_owned()
}

/// `tag_urls`: a URL becomes `<en>` + its spoken reading, trailing
/// punctuation kept after the span.
fn tag_urls(text: &str) -> String {
    let re = fregex!(concat!(
        r"(?i)(?<![\w@./])(?:(?:https?://|www\.)[^\s<>]+|[A-Za-z0-9][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)*\.",
        r"(?:com|org|net|io|ai|co|il|ru|de|es|it|uk|fr|eu|info|app|dev|me|tv|edu|gov)",
        r"(?:\.[a-z]{2})?(?:/[^\s<>]*)?)(?![\w@])"
    ));
    sub_outside(re, text, |c| {
        let whole = &c[0];
        let url = whole.trim_end_matches(|ch| ".,;:!?)]}»\"'”’".contains(ch));
        let tail = &whole[url.len()..];
        if !url.contains('.') && !url.to_lowercase().starts_with("http") {
            return whole.to_owned();
        }
        let spoken = spoken_url(url);
        if spoken.is_empty() {
            tail.to_owned()
        } else {
            format!("<en>{spoken}</en>{tail}")
        }
    })
}

/// `strip_handles`: `@name` is read as the name, `#tag` as the tag.
fn strip_handles(text: &str) -> String {
    let text = sub_outside(
        fregex!(r"(?<![\w@.])@([A-Za-z0-9_]{2,})(?![\w@])"),
        text,
        |c| c[1].to_owned(),
    );
    sub_outside(fregex!(r"(?<![\w&#])#(?=\p{L})"), &text, |_| String::new())
}

/// `_PHONE_RE`: 7 to 15 digits in groups split by `-`, spaces or `( )`.
fn phone_re() -> &'static fancy_regex::Regex {
    fregex!(concat!(
        r"(?<![\w+.,/:-])(\+[ ]?)?(?:\([0-9]{1,5}\)[ -]?|[0-9]{1,5}[ -])(?:\([0-9]{1,5}\)[ -]?|[0-9]{1,5}[ -]){0,5}",
        r"[0-9]{2,8}(?![\w/:-]|[.,][0-9])"
    ))
}

/// `_phone_groups`: whether a phone-shaped match is a phone number, and its
/// leading `+` and digit groups. After a `+`, a `(0)` trunk is dropped.
fn phone_groups(raw: &str) -> Option<(bool, Vec<String>)> {
    let (plus, paren) = (raw.starts_with('+'), raw.contains('('));
    let raw = if plus {
        regex!(r"\(0\)[ -]?").replace_all(raw, "").into_owned()
    } else {
        raw.to_owned()
    };
    let groups: Vec<String> = regex!(r"[0-9]+").find_iter(&raw).map(|m| m.as_str().to_owned()).collect();
    let digits: usize = groups.iter().map(String::len).sum();
    if !(7..=15).contains(&digits) {
        return None;
    }
    if !(plus || paren || groups.len() >= 3 || groups[0].starts_with('0')) {
        return None;
    }
    if !plus
        && !paren
        && !raw.contains('-')
        && digits < 9
        && groups[0].len() <= 3
        && groups[1..].iter().all(|g| g.len() == 3)
    {
        return None; // "1 000 000": thousands ("612 345 678" is a phone)
    }
    if regex!(r"^\d{4}-\d{2}-\d{2}$").is_match(py_strip(&raw)) {
        return None; // an ISO date
    }
    Some((plus, groups))
}

/// `spell_phone_numbers`: en/de/es/it phone numbers digit by digit, groups
/// joined with `", "`, a leading `+` as `plus`/`más`/`più`.
pub fn spell_phone_numbers(text: &str, lang: &str) -> String {
    let plus_word = match lang {
        "en" | "de" => "plus",
        "es" => "más",
        "it" => "più",
        _ => return text.to_owned(),
    };
    sub_outside(phone_re(), text, |c| {
        let Some((plus, groups)) = phone_groups(&c[0]) else {
            return c[0].to_owned();
        };
        let spoken = groups.iter().map(|g| spaced_digits(g)).collect::<Vec<_>>().join(", ");
        if plus {
            format!("{plus_word} {spoken}")
        } else {
            spoken
        }
    })
}

/// `spell_he_ru_phones`: Hebrew digit by digit in Hebrew; Russian group by
/// group (a group with a leading zero digit by digit).
fn spell_he_ru_phones(text: &str, lang: &str) -> String {
    sub_outside(phone_re(), text, |c| {
        let Some((plus, groups)) = phone_groups(&c[0]) else {
            return c[0].to_owned();
        };
        if lang == "he" {
            let spoken = groups
                .iter()
                .map(|g| g.chars().map(|d| HE_DIGITS[d as usize - '0' as usize]).collect::<Vec<_>>().join(" "))
                .collect::<Vec<_>>()
                .join(", ");
            return if plus { format!("פלוס {spoken}") } else { spoken };
        }
        let spoken = groups
            .iter()
            .map(|g| if g.starts_with('0') && g.len() > 1 { spaced_digits(g) } else { g.clone() })
            .collect::<Vec<_>>()
            .join(", ");
        if plus {
            format!("{PLUS_WORD_RU} {spoken}")
        } else {
            spoken
        }
    })
}

fn expand_time_ranges(text: &str, lang: &str) -> String {
    sub_outside(
        fregex!(r"(?<![\d:])(\d{1,2}:\d{2})[ \t]*[-–—][ \t]*(\d{1,2}:\d{2})(?![\d:])"),
        text,
        |c| format!("{} {} {}", &c[1], range_word(lang), &c[2]),
    )
}

fn expand_scores(text: &str, lang: &str) -> String {
    let re = fregex!(r"(?<![\d:.,])(\d{1,2})[ \t]*:[ \t]*(\d)(?![\d:])");
    if lang == "he" {
        return sub_outside(re, text, |c| format!("{} ל-{}", &c[1], &c[2]));
    }
    sub_outside(re, text, |c| format!("{} {} {}", &c[1], ratio_word(lang), &c[2]))
}

fn looks_like_date(parts: &[&str]) -> bool {
    parts.len() == 3
        && (1..=31).contains(&int(parts[0]))
        && (1..=12).contains(&int(parts[1]))
        && matches!(parts[2].chars().count(), 2 | 4)
}

fn expand_versions(text: &str, lang: &str) -> String {
    sub_outside(fregex!(r"(?<![\d.,])\d+(?:\.\d+){2,}(?![\d]|\.\d)"), text, |c| {
        let parts: Vec<&str> = c[0].split('.').collect();
        if looks_like_date(&parts) {
            return c[0].to_owned();
        }
        parts.join(&format!(" {} ", point_word(lang)))
    })
}

fn expand_math(text: &str, lang: &str) -> String {
    sub_outside(fregex!(r"(?<=\d)[ \t]*([+=×÷−])[ \t]*(?=[\d(−])"), text, |c| {
        format!(" {} ", math_word(lang, &c[1]))
    })
}

const HE_PREFIX_LETTERS: &str = "והבלמשכ";

fn expand_unary_minus(text: &str, lang: &str) -> String {
    // (?:(?<=^)|(?<=[\s(\[])|(?<=(?:^|\s)[והבלמשכ]{1,3}-))[-−](?=\d), re.M
    let behind = |piece: &str, at: usize| match prev_char(piece, at) {
        None => true,
        Some(c) if py_space(c) || c == '(' || c == '[' => true,
        Some('-') => ends_with_prefix(&piece[..at - 1], HE_PREFIX_LETTERS, 1, 3, py_space),
        Some(_) => false,
    };
    let word = minus_word(lang);
    outside_spans(text, |piece| {
        sub_behind(fregex!(r"[-−](?=\d)"), piece, behind, |_| format!("{word} "))
    })
}

const EN_DEN: [&str; 11] = ["", "", "half", "third", "quarter", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"];
const DE_NUM: [&str; 10] = ["null", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"];
const DE_DEN: [&str; 11] = ["", "", "halb", "Drittel", "Viertel", "Fünftel", "Sechstel", "Siebtel", "Achtel", "Neuntel", "Zehntel"];
const ES_NUM: [&str; 10] = ["cero", "un", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"];
const ES_DEN: [&str; 11] = ["", "", "medio", "tercio", "cuarto", "quinto", "sexto", "séptimo", "octavo", "noveno", "décimo"];
const IT_NUM: [&str; 10] = ["zero", "un", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove"];
const IT_DEN: [(&str, &str); 11] = [
    ("", ""),
    ("", ""),
    ("mezzo", "mezzi"),
    ("terzo", "terzi"),
    ("quarto", "quarti"),
    ("quinto", "quinti"),
    ("sesto", "sesti"),
    ("settimo", "settimi"),
    ("ottavo", "ottavi"),
    ("nono", "noni"),
    ("decimo", "decimi"),
];
const RU_NUM_F: [&str; 10] = ["ноль", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"];
const RU_DEN: [(&str, &str); 11] = [
    ("", ""),
    ("", ""),
    ("вторая", "вторых"),
    ("третья", "третьих"),
    ("четвёртая", "четвёртых"),
    ("пятая", "пятых"),
    ("шестая", "шестых"),
    ("седьмая", "седьмых"),
    ("восьмая", "восьмых"),
    ("девятая", "девятых"),
    ("десятая", "десятых"),
];

/// `fraction_words`: `n/d` (1 ≤ n < d ≤ 10) in words.
pub fn fraction_words(n: usize, d: usize, lang: &str) -> Option<String> {
    if !(1 <= n && n < d && d <= 10) {
        return None;
    }
    Some(match lang {
        "he" => match (n, d) {
            (1, 2) => "חצי".to_owned(),
            (1, 3) => "שליש".to_owned(),
            (2, 3) => "שני שלישים".to_owned(),
            (1, 4) => "רבע".to_owned(),
            (3, 4) => "שלושה רבעים".to_owned(),
            _ => format!("{n} חלקי {d}"),
        },
        "en" => {
            let mut den = EN_DEN[d].to_owned();
            if n > 1 {
                den = if d == 2 { "halves".to_owned() } else { den + "s" };
            }
            format!("{} {den}", EN_NUM[n])
        }
        "de" if (n, d) == (1, 2) => "ein halb".to_owned(),
        "de" => format!("{} {}", DE_NUM[n], DE_DEN[d]),
        "es" => format!("{} {}{}", ES_NUM[n], ES_DEN[d], if n > 1 { "s" } else { "" }),
        "it" => format!("{} {}", IT_NUM[n], if n == 1 { IT_DEN[d].0 } else { IT_DEN[d].1 }),
        "ru" => format!("{} {}", RU_NUM_F[n], if n == 1 { RU_DEN[d].0 } else { RU_DEN[d].1 }),
        _ => return None,
    })
}

fn expand_fractions(text: &str, lang: &str) -> String {
    sub_outside(
        fregex!(r"(?<![\d/.,:])([1-9])/([1-9]|10)(?!\d|/|[.,:]\d)"),
        text,
        |c| fraction_words(int(&c[1]) as usize, int(&c[2]) as usize, lang).unwrap_or_else(|| c[0].to_owned()),
    )
}

fn spell_cards(text: &str, lang: &str) -> String {
    sub_outside(fregex!(r"(?<![\d\-])\d{4}([ \-])\d{4}\1\d{4}\1\d{4}(?![\d\-])"), text, |c| {
        let groups: Vec<&str> = regex!(r"\d{4}").find_iter(&c[0]).map(|m| m.as_str()).collect();
        groups
            .iter()
            .map(|g| {
                if lang == "he" {
                    g.chars()
                        .map(|d| HE_DIGITS[d.to_digit(10).unwrap_or(0) as usize])
                        .collect::<Vec<_>>()
                        .join(" ")
                } else {
                    g.chars().map(String::from).collect::<Vec<_>>().join(" ")
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    })
}

const HE_MONTHS: [&str; 12] = [
    "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר",
];

fn expand_he_day_month(text: &str) -> String {
    sub_outside(fregex!(r"(?<![\d/.,:])(\d{1,2})/(\d{1,2})(?![\d/]|[.,]\d)"), text, |c| {
        let (d, mo) = (int(&c[1]), int(&c[2]));
        if !((1..=31).contains(&d) && (1..=12).contains(&mo)) {
            return c[0].to_owned();
        }
        format!("{d} ב{}", HE_MONTHS[mo as usize - 1])
    })
}

fn he_postal_codes(text: &str) -> String {
    // (?<=מיקוד:?[ \t])(\d{5,7})(?!\d)
    let behind = |piece: &str, at: usize| {
        matches!(prev_char(piece, at), Some(' ' | '\t'))
            && (piece[..at - 1].ends_with("מיקוד") || piece[..at - 1].ends_with("מיקוד:"))
    };
    outside_spans(text, |piece| {
        sub_behind(fregex!(r"(\d{5,7})(?!\d)"), piece, behind, |c| spaced_digits(&c[1]))
    })
}

const HE_MASC_LABELS: [&str; 22] = [
    "פרק", "עמוד", "סעיף", "שלב", "חדר", "תרגיל", "שיעור", "פסוק", "ערוץ", "אוטובוס", "קו", "רציף", "שער", "חלק",
    "כרך", "דגם", "ספר", "מקום", "טור", "סיבוב", "משחק", "מספר",
];

/// "1" after a masculine label noun is "אחד" ("פרק 1" as "פרק אחד").
fn he_masculine_one(text: &str) -> String {
    // (?<=(?:^|[\s("'])[והבלמש]{0,2}(?:פרק|…)[ \t])1(?!\d|[.,:/]\d|%), re.M
    let behind = |piece: &str, at: usize| {
        if !matches!(prev_char(piece, at), Some(' ' | '\t')) {
            return false;
        }
        let before = &piece[..at - 1];
        HE_MASC_LABELS.iter().any(|label| {
            before.strip_suffix(label).is_some_and(|rest| {
                ends_with_prefix(rest, "והבלמש", 0, 2, |c| py_space(c) || matches!(c, '(' | '"' | '\''))
            })
        })
    };
    outside_spans(text, |piece| {
        sub_behind(fregex!(r"1(?!\d|[.,:/]\d|%)"), piece, behind, |_| "אחד".to_owned())
    })
}

fn spell_leading_zeros(text: &str) -> String {
    sub_outside(
        fregex!(r"(?<![\d.,:/\-+])0\d+(?!\d|[.,:/]\d|[\-–]\d)"),
        text,
        |c| spaced_digits(&c[0]),
    )
}

fn split_digits_from_latin(text: &str) -> String {
    sub_outside(fregex!(r"(?<=\d)(?=[A-Za-z])"), text, |_| " ".to_owned())
}

/// A prefix hyphen before a Hebrew word is removed (`ו-שלושה` as `ושלושה`).
fn glue_he_prefix_hyphens(text: &str) -> String {
    // (?<=(?:^|[\s(])[והבלמשכ]{1,3})-(?=[א-ת]), re.M
    let behind = |piece: &str, at: usize| {
        ends_with_prefix(&piece[..at], HE_PREFIX_LETTERS, 1, 3, |c| py_space(c) || c == '(')
    };
    outside_spans(text, |piece| sub_behind(fregex!(r"-(?=[א-ת])"), piece, behind, |_| String::new()))
}

const EN_MONTHS: [&str; 12] = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
    "December",
];

fn expand_iso_dates(text: &str, lang: &str) -> String {
    sub_outside(fregex!(r"(?<![\d.\-/])(\d{4})-(\d{2})-(\d{2})(?![\d\-])"), text, |c| {
        let (y, mo, d) = (int(&c[1]), int(&c[2]), int(&c[3]));
        if !((1..=12).contains(&mo) && (1..=31).contains(&d)) {
            return c[0].to_owned();
        }
        match lang {
            "en" => format!("{} {d}, {y}", EN_MONTHS[mo as usize - 1]),
            "de" | "ru" => format!("{d:02}.{mo:02}.{y}"),
            _ => format!("{d}/{mo}/{y}"),
        }
    })
}

/// `_ru_count_form`: the Russian form for a count written as `num`.
fn ru_count_form<'a>(num: &str, forms: [&'a str; 3]) -> &'a str {
    if num.contains([',', '.']) {
        return forms[1];
    }
    // _ru_pluralize(int(num)): only the last two digits count.
    let tail: String = num.chars().rev().take(2).collect::<Vec<_>>().into_iter().rev().collect();
    let n: u64 = tail.parse().unwrap_or(0);
    if (11..=14).contains(&(n % 100)) {
        return forms[2];
    }
    match n % 10 {
        1 => forms[0],
        2..=4 => forms[1],
        _ => forms[2],
    }
}

fn expand_degrees(text: &str, lang: &str) -> String {
    sub_outside(fregex!(r"(\d+(?:[.,]\d+)?)[ \t]?°[ \t]?([CF](?![A-Za-z]))?"), text, |c| {
        let num = &c[1];
        let scale = c.get(2).map_or("", |m| m.as_str());
        let one = num == "1";
        let latin = match scale {
            "C" => " Celsius",
            "F" => " Fahrenheit",
            _ => "",
        };
        let (unit, name) = match lang {
            "en" => (if one { "degree" } else { "degrees" }, latin),
            "de" => ("Grad", latin),
            "es" => (if one { "grado" } else { "grados" }, latin),
            "it" => (if one { "grado" } else { "gradi" }, latin),
            _ => (
                ru_count_form(num, ["градус", "градуса", "градусов"]),
                match scale {
                    "C" => " Цельсия",
                    "F" => " по Фаренгейту",
                    _ => "",
                },
            ),
        };
        format!("{num} {unit}{name}")
    })
}

fn unit_words(lang: &str, unit: &str) -> (&'static str, &'static str) {
    match (lang, unit) {
        ("en", "km/h") => ("kilometer per hour", "kilometers per hour"),
        ("en", "km") => ("kilometer", "kilometers"),
        ("en", "kg") => ("kilogram", "kilograms"),
        ("en", "cm") => ("centimeter", "centimeters"),
        ("en", "mm") => ("millimeter", "millimeters"),
        ("en", "mg") => ("milligram", "milligrams"),
        ("en", "ml") => ("milliliter", "milliliters"),
        ("en", "m") => ("meter", "meters"),
        ("en", "g") => ("gram", "grams"),
        ("en", _) => ("liter", "liters"),
        ("de", "km/h") => ("Kilometer pro Stunde", "Kilometer pro Stunde"),
        ("de", "km") => ("Kilometer", "Kilometer"),
        ("de", "kg") => ("Kilogramm", "Kilogramm"),
        ("de", "cm") => ("Zentimeter", "Zentimeter"),
        ("de", "mm") => ("Millimeter", "Millimeter"),
        ("de", "mg") => ("Milligramm", "Milligramm"),
        ("de", "ml") => ("Milliliter", "Milliliter"),
        ("de", "m") => ("Meter", "Meter"),
        ("de", "g") => ("Gramm", "Gramm"),
        ("de", _) => ("Liter", "Liter"),
        ("es", "km/h") => ("kilómetro por hora", "kilómetros por hora"),
        ("es", "km") => ("kilómetro", "kilómetros"),
        ("es", "kg") => ("kilogramo", "kilogramos"),
        ("es", "cm") => ("centímetro", "centímetros"),
        ("es", "mm") => ("milímetro", "milímetros"),
        ("es", "mg") => ("miligramo", "miligramos"),
        ("es", "ml") => ("mililitro", "mililitros"),
        ("es", "m") => ("metro", "metros"),
        ("es", "g") => ("gramo", "gramos"),
        ("es", _) => ("litro", "litros"),
        (_, "km/h") => ("chilometro orario", "chilometri orari"),
        (_, "km") => ("chilometro", "chilometri"),
        (_, "kg") => ("chilogrammo", "chilogrammi"),
        (_, "cm") => ("centimetro", "centimetri"),
        (_, "mm") => ("millimetro", "millimetri"),
        (_, "mg") => ("milligrammo", "milligrammi"),
        (_, "ml") => ("millilitro", "millilitri"),
        (_, "m") => ("metro", "metri"),
        (_, "g") => ("grammo", "grammi"),
        (_, _) => ("litro", "litri"),
    }
}

fn ru_unit_forms(unit: &str) -> [&'static str; 3] {
    match unit {
        "км/ч" => ["километр в час", "километра в час", "километров в час"],
        "км" => ["километр", "километра", "километров"],
        "кг" => ["килограмм", "килограмма", "килограммов"],
        "см" => ["сантиметр", "сантиметра", "сантиметров"],
        "мм" => ["миллиметр", "миллиметра", "миллиметров"],
        "мг" => ["миллиграмм", "миллиграмма", "миллиграммов"],
        "мл" => ["миллилитр", "миллилитра", "миллилитров"],
        "м" => ["метр", "метра", "метров"],
        _ => ["литр", "литра", "литров"],
    }
}

fn expand_units(text: &str, lang: &str) -> String {
    if lang == "ru" {
        return sub_outside(
            fregex!(r"(\d+(?:[.,]\d+)?)[ \t]?(км/ч|км|кг|см|мм|мг|мл|м|л)(?![\w/])"),
            text,
            |c| format!("{} {}", &c[1], ru_count_form(&c[1], ru_unit_forms(&c[2]))),
        );
    }
    sub_outside(
        fregex!(r"(\d+(?:[.,]\d+)?)[ \t]?(km/h|km|kg|cm|mm|mg|ml|m|g|l)(?![\w/])"),
        text,
        |c| {
            let (one, many) = unit_words(lang, &c[2]);
            format!("{} {}", &c[1], if &c[1] == "1" { one } else { many })
        },
    )
}

fn ru_currency_forms(sign: &str) -> [&'static str; 3] {
    match sign {
        "₽" => ["рубль", "рубля", "рублей"],
        "$" => ["доллар", "доллара", "долларов"],
        "€" => ["евро", "евро", "евро"],
        _ => ["фунт", "фунта", "фунтов"],
    }
}

fn expand_ru_currency(text: &str) -> String {
    let text = sub_outside(fregex!(r"([₽$€£])[ \t]?(\d+(?:[.,]\d+)?)"), text, |c| {
        format!("{} {}", &c[2], ru_count_form(&c[2], ru_currency_forms(&c[1])))
    });
    sub_outside(fregex!(r"(\d+(?:[.,]\d+)?)[ \t]?([₽$€£])"), &text, |c| {
        format!("{} {}", &c[1], ru_count_form(&c[1], ru_currency_forms(&c[2])))
    })
}

fn ru_ord_stem(word: &str) -> Option<&'static str> {
    Some(match word {
        "один" => "перв",
        "два" => "втор",
        "три" => "трет",
        "четыре" => "четвёрт",
        "пять" => "пят",
        "шесть" => "шест",
        "семь" => "седьм",
        "восемь" => "восьм",
        "девять" => "девят",
        "десять" => "десят",
        "одиннадцать" => "одиннадцат",
        "двенадцать" => "двенадцат",
        "тринадцать" => "тринадцат",
        "четырнадцать" => "четырнадцат",
        "пятнадцать" => "пятнадцат",
        "шестнадцать" => "шестнадцат",
        "семнадцать" => "семнадцат",
        "восемнадцать" => "восемнадцат",
        "девятнадцать" => "девятнадцат",
        "двадцать" => "двадцат",
        "тридцать" => "тридцат",
        "сорок" => "сороков",
        "пятьдесят" => "пятидесят",
        "шестьдесят" => "шестидесят",
        "семьдесят" => "семидесят",
        "восемьдесят" => "восьмидесят",
        "девяносто" => "девяност",
        "сто" => "сот",
        "двести" => "двухсот",
        "триста" => "трёхсот",
        "четыреста" => "четырёхсот",
        "пятьсот" => "пятисот",
        "шестьсот" => "шестисот",
        "семьсот" => "семисот",
        "восемьсот" => "восьмисот",
        "девятьсот" => "девятисот",
        "тысяча" | "тысячи" | "тысяч" => "тысячн",
        _ => return None,
    })
}

/// `_RU_ORD_CASE`: the case a hyphenated ordinal suffix asks for.
fn ru_ord_case(suffix: &str) -> char {
    match suffix {
        "й" | "ый" | "ий" | "ой" => 'm',
        "я" | "ая" => 'f',
        "е" | "ое" => 'n',
        "го" | "ого" => 'g',
        "му" | "ому" => 'd',
        "м" | "ом" => 'p',
        "х" | "ых" => 'x',
        "ю" | "ую" => 'a',
        "ми" | "ыми" => 'i',
        _ => 's', // "ым"
    }
}

fn ru_ord_end(case: char, third: bool) -> &'static str {
    match (case, third) {
        ('m', false) => "ый",
        ('f', false) => "ая",
        ('n', false) => "ое",
        ('g', false) => "ого",
        ('d', false) => "ому",
        ('p', false) => "ом",
        ('x', false) => "ых",
        ('a', false) => "ую",
        ('i', false) => "ыми",
        ('s', false) => "ым",
        ('m', true) => "ий",
        ('f', true) => "ья",
        ('n', true) => "ье",
        ('g', true) => "ьего",
        ('d', true) => "ьему",
        ('p', true) => "ьем",
        ('x', true) => "ьих",
        ('a', true) => "ью",
        ('i', true) => "ьими",
        _ => "ьим",
    }
}

/// `ru_ordinal`: the Russian ordinal of `n` in `case` (m f n g d p x a i s),
/// or `None` (0, a million or more, a round thousand above 5000).
pub fn ru_ordinal(n: u64, case: char) -> Option<String> {
    if n == 0 || n >= 1_000_000 {
        return None;
    }
    if n % 1000 == 0 {
        let prefix = match n / 1000 {
            1 => "",
            2 => "двух",
            3 => "трёх",
            4 => "четырёх",
            5 => "пяти",
            _ => return None,
        };
        return Some(format!("{prefix}тысячн{}", ru_ord_end(case, false)));
    }
    let spoken = ru_int(n, false);
    let mut words: Vec<&str> = spoken.split(' ').collect();
    if words.len() >= 2 && words[0] == "одна" && words[1] == "тысяча" {
        words.remove(0);
    }
    let last = words.pop()?;
    let stem = ru_ord_stem(last)?;
    let end = if stem == "трет" {
        ru_ord_end(case, true)
    } else if case == 'm' && matches!(stem, "втор" | "шест" | "седьм" | "восьм" | "сороков") {
        "ой"
    } else {
        ru_ord_end(case, false)
    };
    words.push("");
    let head = words[..words.len() - 1].join(" ");
    Some(if head.is_empty() {
        format!("{stem}{end}")
    } else {
        format!("{head} {stem}{end}")
    })
}

const RU_MONTHS_GEN: [&str; 12] = [
    "января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября",
    "декабря",
];

fn expand_ru_ordinals_dates(text: &str) -> String {
    let text = sub_outside(
        fregex!(r"(?<![\d.,])(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})(?![\d]|\.\d)(?:[ \t]?(?:г\.|года))?"),
        text,
        |c| {
            let (d, mo, y) = (int(&c[1]), int(&c[2]), &c[3]);
            if !((1..=31).contains(&d) && (1..=12).contains(&mo)) {
                return c[0].to_owned();
            }
            let year = if y.len() == 4 { int(y) } else { 2000 + int(y) };
            // Python formats a missing ordinal as "None"; so does this.
            let none = || "None".to_owned();
            format!(
                "{} {} {} года",
                ru_ordinal(d, 'n').unwrap_or_else(none),
                RU_MONTHS_GEN[mo as usize - 1],
                ru_ordinal(year, 'g').unwrap_or_else(none)
            )
        },
    );
    let text = outside_spans(&text, |piece| {
        freplace_all(
            fregex!(r"(?<![\w.,])(\d{4})[ \t](годах|годов|году|года|годом|год|г\.)(?!\w)"),
            piece,
            |c| {
                let n = int(&c[1]);
                let mut word = c[2].to_owned();
                if word == "г." {
                    let before = &piece[..c.get(0).expect("whole match").start()];
                    let prep = regex!(r"(?:^|\s)[вВ]\s*$").is_match(before);
                    word = if prep { "году" } else { "года" }.to_owned();
                }
                let case = match word.as_str() {
                    "году" => 'p',
                    "года" => 'g',
                    "годом" => 's',
                    "год" => 'm',
                    _ => 'x', // годах, годов
                };
                match ru_ordinal(n, case) {
                    Some(o) => format!("{o} {word}"),
                    None => c[0].to_owned(),
                }
            },
        )
    });
    // only a hyphenated suffix makes an ordinal ("3-й", "5-го", "1990-х")
    sub_outside(
        fregex!(r"(?<![\d.,])(\d{1,6})[-‑](ыми|ого|ому|ый|ий|ой|ая|ое|ых|ую|ым|го|му|ми|й|я|е|м|х|ю)(?![\w])"),
        &text,
        |c| ru_ordinal(int(&c[1]), ru_ord_case(&c[2])).unwrap_or_else(|| c[1].to_owned()),
    )
}

fn join_spaced_thousands(text: &str) -> String {
    sub_outside(
        fregex!(r"(?<![\d.,])\d{1,3}(?:[ \x{a0}\x{202f}]\d{3})+(?![\d]|[.,]\d)"),
        text,
        |c| c[0].replace([' ', '\u{a0}', '\u{202f}'], ""),
    )
}

fn split_word_codes(text: &str) -> String {
    sub_outside(
        fregex!(r"(?<![A-Za-z0-9])([A-Z]?[a-z]{2,})([A-Z]{1,3})(\d+)(?![A-Za-z0-9])"),
        text,
        |c| format!("{} {} {}", &c[1], &c[2], &c[3]),
    )
}

fn expand_number_ranges(text: &str, lang: &str) -> String {
    sub_outside(
        fregex!(r"(?<![\d+\-–])(?<!\d[.,:/])(?<!\d[ \t])(\d{1,4})[ \t]*[-–][ \t]*(\d{1,4})(?!\d|[.,:/]\d|[ \t]?[-–]\d)"),
        text,
        |c| format!("{} {} {}", &c[1], range_word(lang), &c[2]),
    )
}

const ES_ORD: [&str; 10] = ["primer", "segund", "tercer", "cuart", "quint", "sext", "séptim", "octav", "noven", "décim"];
const IT_ORD: [&str; 10] = ["prim", "second", "terz", "quart", "quint", "sest", "settim", "ottav", "non", "decim"];

fn expand_ordinal_indicators(text: &str, lang: &str) -> String {
    let stems = if lang == "es" { ES_ORD } else { IT_ORD };
    sub_outside(fregex!(r"(?<![\d.,])(10|[1-9])\.?([ºª])"), text, |c| {
        format!("{}{}", stems[int(&c[1]) as usize - 1], if &c[2] == "º" { "o" } else { "a" })
    })
}

/// `normalize_before`: the bot rules on the raw text (tags flat), before
/// `prepareText`, for one piece in its language.
pub fn normalize_before(text: &str, lang: &str) -> String {
    let text = outside_spans(text, strip_markdown);
    let text = tag_urls(&text);
    let mut text = strip_handles(&text);
    if matches!(lang, "en" | "de" | "es" | "it") {
        text = spell_phone_numbers(&text, lang);
    }
    let text = expand_time_ranges(&text, lang);
    let text = if lang != "ru" { expand_scores(&text, lang) } else { text };
    let text = expand_versions(&text, lang);
    let text = expand_math(&text, lang);
    let text = expand_unary_minus(&text, lang);
    let text = expand_fractions(&text, lang);
    let text = spell_cards(&text, lang);
    if lang == "he" {
        // maqaf: RenikudPlus passed it through (a pad id)
        let text = text.replace('־', "-");
        let text = spell_he_ru_phones(&text, lang);
        let text = expand_he_day_month(&text);
        let text = he_postal_codes(&text);
        let text = he_masculine_one(&text);
        let text = spell_leading_zeros(&text);
        let text = split_digits_from_latin(&text);
        return glue_he_prefix_hyphens(&text);
    }
    let text = expand_iso_dates(&text, lang);
    let text = expand_degrees(&text, lang);
    if lang == "ru" {
        let text = spell_he_ru_phones(&text, lang);
        let text = join_spaced_thousands(&text);
        let text = expand_ru_currency(&text);
        let text = expand_units(&text, lang);
        let text = expand_ru_ordinals_dates(&text);
        let text = split_word_codes(&text);
        let text = spell_leading_zeros(&text);
        return split_digits_from_latin(&text);
    }
    let text = expand_number_ranges(&text, lang);
    let mut text = expand_units(&text, lang);
    if matches!(lang, "es" | "it") {
        text = expand_ordinal_indicators(&text, lang);
    }
    if lang == "en" {
        text = sub_outside(fregex!(r"(?i)\bext\.?[ \t]*(?=\d)"), &text, |_| "extension ".to_owned());
    }
    text
}

/// `normalize_after`: the bot rules on the prepared text, for one piece in
/// its language.
pub fn normalize_after(text: &str, lang: &str) -> String {
    if lang == "ru" {
        return freplace_all(fregex!(r"[ \t]+\.(?=[ \t]*[.!?…])"), text, |_| String::new());
    }
    let text = crate::handling::tag_emails(text, lang);
    if lang != "he" {
        return text;
    }
    // Typographic quotes came back from RenikudPlus as pad ids; Cyrillic in
    // Hebrew text is read by the Russian G2P.
    let text = outside_spans(&text, |t| regex!(r"[„“”‟«»‹›]").replace_all(t, "").into_owned());
    sub_outside(
        fregex!(r"[\x{400}-\x{4FF}]+(?:[ \t\-]+[\x{400}-\x{4FF}]+)*"),
        &text,
        |c| format!("<ru>{}</ru>", &c[0]),
    )
}

/// `split_long_words`: polyg2p's English model stops after about 20 letters
/// of an unknown word, so a Latin word of 21 or more letters is read in
/// equal pieces of at most 14.
pub fn split_long_words(text: &str) -> String {
    regex!(r"[A-Za-z]{21,}")
        .replace_all(text, |c: &regex::Captures| {
            let w = &c[0];
            let n = w.len().div_ceil(14);
            let size = w.len().div_ceil(n);
            w.as_bytes()
                .chunks(size)
                .map(|piece| std::str::from_utf8(piece).expect("ASCII"))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinals_and_fractions_read_as_the_reference_reads_them() {
        assert_eq!(ru_ordinal(3, 'm').unwrap(), "третий");
        assert_eq!(ru_ordinal(2, 'm').unwrap(), "второй");
        assert_eq!(ru_ordinal(2024, 'p').unwrap(), "две тысячи двадцать четвёртом");
        assert_eq!(ru_ordinal(1001, 'm').unwrap(), "тысяча первый");
        assert_eq!(ru_ordinal(3000, 'g').unwrap(), "трёхтысячного");
        assert_eq!(ru_ordinal(6000, 'g'), None);
        assert_eq!(fraction_words(3, 4, "en").unwrap(), "three quarters");
        assert_eq!(fraction_words(1, 2, "de").unwrap(), "ein halb");
        assert_eq!(fraction_words(2, 3, "es").unwrap(), "dos tercios");
        assert_eq!(fraction_words(4, 3, "en"), None);
    }

    #[test]
    fn urls_and_long_words() {
        assert_eq!(
            spoken_url("https://www.example.co.il/path-a?x=1"),
            "w w w dot example dot c o dot i l slash path dash a"
        );
        assert_eq!(spoken_url("v2.app"), "v two dot app");
        assert_eq!(
            split_long_words("Pneumonoultramicroscopicsilicovolcanoconiosis"),
            "Pneumonoultr amicroscopic silicovolcan oconiosis"
        );
    }

    #[test]
    fn hand_written_look_behinds() {
        assert_eq!(normalize_before("פרק 1 ועמוד 1", "he"), "פרק אחד ועמוד אחד");
        assert_eq!(normalize_before("מיקוד 6100101", "he"), "מיקוד 6 1 0 0 1 0 1");
        assert_eq!(normalize_before("ו-שלושה", "he"), "ושלושה");
        assert_eq!(normalize_before("It was -5 today", "en"), "It was minus 5 today");
    }
}
