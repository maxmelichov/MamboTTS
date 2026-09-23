//! Model-independent handling for difficult Hebrew TTS input.
//!
//! Niqqud the writer typed is kept through normalization; RenikudPlus reads it
//! directly during G2P.

use regex::{Captures, Regex};

use crate::hebrew_numbers::normalize_hebrew_numbers;

/// Internal boundaries for segments that need slower, clearer synthesis.
pub const REF_CODE_MARK_OPEN: char = '【';
pub const REF_CODE_MARK_CLOSE: char = '】';

/// Text that is ready for G2P, split by its requested synthesis pacing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedSegment {
    pub text: String,
    pub is_reference_code: bool,
}

/// Strip Hebrew niqqud / cantillation while keeping letters and geresh.
pub fn strip_nikud(text: &str) -> String {
    text.chars()
        .filter(|character| !is_nikud(*character))
        .collect()
}

/// True when `text` contains any Hebrew niqqud / cantillation mark.
pub fn contains_nikud(text: &str) -> bool {
    text.chars().any(is_nikud)
}

/// Remove internal slow-segment markers before displaying prepared text.
pub fn strip_reference_code_markers(text: &str) -> String {
    text.replace(REF_CODE_MARK_OPEN, "")
        .replace(REF_CODE_MARK_CLOSE, "")
}

/// Split prepared text into ordinary and slow reference-code segments.
///
/// Punctuation-only fragments are retained with the preceding segment, matching
/// the Python pipeline's protection against a stray final phoneme.
pub fn split_prepared_by_reference_codes(text: &str) -> Vec<PreparedSegment> {
    let mut segments = Vec::new();
    let mut ordinary = String::new();
    let mut slow = String::new();
    let mut in_slow = false;

    let push = |text: &mut String, is_reference_code, out: &mut Vec<PreparedSegment>| {
        let value = text.trim();
        if !value.is_empty() {
            out.push(PreparedSegment {
                text: value.to_owned(),
                is_reference_code,
            });
        }
        text.clear();
    };

    for character in text.chars() {
        match character {
            REF_CODE_MARK_OPEN if !in_slow => {
                push(&mut ordinary, false, &mut segments);
                in_slow = true;
            }
            REF_CODE_MARK_CLOSE if in_slow => {
                push(&mut slow, true, &mut segments);
                in_slow = false;
            }
            _ if in_slow => slow.push(character),
            _ => ordinary.push(character),
        }
    }
    if in_slow {
        ordinary.push(REF_CODE_MARK_OPEN);
        ordinary.push_str(&slow);
    }
    push(&mut ordinary, false, &mut segments);

    let mut merged: Vec<PreparedSegment> = Vec::new();
    for segment in segments {
        if segment
            .text
            .chars()
            .all(|c| matches!(c, '.' | '!' | '?' | ',' | ';' | ':' | '…'))
            && !merged.is_empty()
        {
            merged
                .last_mut()
                .expect("not empty")
                .text
                .push_str(&segment.text);
        } else {
            merged.push(segment);
        }
    }
    if merged.is_empty() {
        merged.push(PreparedSegment {
            text: strip_reference_code_markers(text).trim().to_owned(),
            is_reference_code: false,
        });
    }
    merged
}

/// Normalize structured text while preserving ordinary Hebrew words **and**
/// their niqqud.
pub fn normalize_for_speech(text: &str) -> String {
    let heading = Regex::new(r"(?m)^\s{0,3}#{1,6}\s+").expect("valid heading regex");
    let list = Regex::new(r"(?m)(^|\n)\s*(\d{1,2})[.)]\s+").expect("valid list regex");
    let email =
        Regex::new(r"(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}").expect("valid email regex");
    let phone = Regex::new(r"(?:\*\d{2,}|0\d{1,2}-\d{6,8})").expect("valid phone regex");
    let date = Regex::new(r"([0-3]?\d)[/.]([01]?\d)[/.](\d{2}|\d{4})").expect("valid date regex");
    let time = Regex::new(r"([01]?\d|2[0-3]):([0-5]\d)").expect("valid time regex");
    let identifier =
        Regex::new(r"[A-Za-z]+(?:[-_/]?[A-Za-z0-9]+)+").expect("valid identifier regex");

    let mut text = heading.replace_all(text, "").into_owned();
    text = text
        .replace('–', "-")
        .replace('—', "-")
        .replace('‑', "-")
        .replace('[', " ")
        .replace(']', " ")
        .replace('{', " ")
        .replace('}', " ")
        .replace('(', ",")
        .replace(')', ",");
    text = normalize_hebrew_punctuation(&text);
    text = list
        .replace_all(&text, |caps: &Captures| {
            format!(
                "{}{}. ",
                &caps[1],
                list_number(caps[2].parse().unwrap_or(0))
            )
        })
        .into_owned();
    text = email
        .replace_all(&text, |caps: &Captures| expand_email(&caps[0]))
        .into_owned();
    text = phone
        .replace_all(&text, |caps: &Captures| expand_phone(&caps[0]))
        .into_owned();
    text = date
        .replace_all(&text, |caps: &Captures| expand_date(caps))
        .into_owned();
    text = time
        .replace_all(&text, |caps: &Captures| expand_time(caps))
        .into_owned();
    text = identifier
        .replace_all(&text, |caps: &Captures| expand_identifier(&caps[0]))
        .into_owned();
    normalize_punctuation(&text)
}

/// Apply BlueTTS's text normalization before G2P.
///
/// This intentionally keeps `<en>…</en>` spans and `【…】` pacing spans in the
/// output. The G2P and synthesis layers consume those markers later.
pub fn prepare_text_for_synthesis(text: &str, lang: &str) -> String {
    let lang = canonical_lang(lang);
    let mut text = strip_helper_markup(text);
    text = normalize_common_text(&text);
    if lang == "he" {
        text = normalize_hebrew_punctuation(&text);
        text = expand_letter_labels(&text);
        text = expand_geresh_loanwords(&text);
        text = expand_dialogue_quotes(&text);
        text = expand_lamed_before_latin(&text);
    }
    text = expand_alphanumeric_codes(&text, &lang);
    text = expand_list_markers(&text, &lang);
    text = expand_plus_sign(&text, &lang);
    if lang == "he" {
        text = expand_phone_numbers(&text);
        text = expand_times(&text);
        text = expand_dates(&text);
        // Hebrew number words depend on gender, on the noun that follows and
        // on whether the digits are a year, an ordinal or a range, so Hebrew
        // gets its own pass and leaves nothing for the generic ones below.
        text = normalize_hebrew_numbers(&text);
    }
    text = expand_percent_symbols(&text, &lang);
    text = expand_ratios(&text, &lang);
    text = expand_numbers(&text, &lang);
    strip_silent_separator_tokens(&text.replace(':', ". "))
}

fn canonical_lang(lang: &str) -> String {
    match lang.to_ascii_lowercase().as_str() {
        "en-us" => "en".to_owned(),
        "ge" => "de".to_owned(),
        value => value.to_owned(),
    }
}

fn strip_helper_markup(text: &str) -> String {
    let block = Regex::new(r"(?is)<lang_list\b[^>]*>.*?</lang_list>").expect("valid regex");
    let tag = Regex::new(r"(?i)</?lang_list\b[^>]*>").expect("valid regex");
    tag.replace_all(&block.replace_all(text, " "), " ")
        .into_owned()
}

fn normalize_common_text(text: &str) -> String {
    let heading = Regex::new(r"(?m)(^|\s)#{1,6}\s*").expect("valid regex");
    let mut text = heading.replace_all(text, "$1").into_owned();
    text = text
        .replace(['(', '[', '{'], ", ")
        .replace([')', ']', '}'], ", ");
    text = normalize_punctuation(&text);
    Regex::new(r"(?i)\banymore\b")
        .expect("valid regex")
        .replace_all(&text, |caps: &Captures| {
            if caps[0].starts_with('A') {
                "Any more"
            } else {
                "any more"
            }
        })
        .into_owned()
}

fn mark_slow_segment(text: impl AsRef<str>) -> String {
    format!("{REF_CODE_MARK_OPEN}{}{REF_CODE_MARK_CLOSE}", text.as_ref())
}

fn expand_letter_labels(text: &str) -> String {
    // A lone letter with a geresh is a letter name or an ordinal (דגניה ב׳,
    // כיתה ג׳, סעיף א׳). A letter right after the geresh makes it a loanword
    // instead (ג׳אז, צ׳יפס), which stays intact. Niqqud may sit on the letter.
    let chars: Vec<char> = text.chars().collect();
    let is_quote = |c: char| matches!(c, '׳' | '\'' | '’' | '"' | '״');
    let is_word = |c: char| c.is_alphanumeric() || is_nikud(c);
    let mut output = String::with_capacity(text.len());
    let mut index = 0;
    while index < chars.len() {
        let character = chars[index];
        if let Some(name) = letter_name(character) {
            // An opening quote may precede the letter, but not a geresh or
            // gershayim that belongs to the word before it.
            let starts_word = match index.checked_sub(1).map(|i| chars[i]) {
                None => true,
                Some(c) if is_quote(c) => index
                    .checked_sub(2)
                    .is_none_or(|i| chars[i].is_whitespace()),
                Some(c) => !is_word(c),
            };
            let mut end = index + 1;
            while chars.get(end).copied().is_some_and(is_nikud) {
                end += 1;
            }
            let geresh = matches!(chars.get(end), Some('׳' | '\'' | '’'));
            // A closing quote may follow, but a second quote mark inside a
            // word (ט''ו) is gershayim typed twice, not a label.
            let ends_word = |at: usize| chars.get(at).is_none_or(|&c| !is_word(c));
            let ends_word = match chars.get(end + 1) {
                Some(&c) if is_quote(c) => ends_word(end + 2),
                _ => ends_word(end + 1),
            };
            if starts_word && geresh && ends_word {
                output.push_str(name);
                index = end + 1;
                continue;
            }
        }
        output.push(character);
        index += 1;
    }
    output
}

fn letter_name(letter: char) -> Option<&'static str> {
    Some(match letter {
        'א' => "אָלֶף",
        'ב' => "בֵּת",
        'ג' => "גִּימֶל",
        'ד' => "דָּלֶת",
        'ה' => "הֵא",
        'ו' => "וָו",
        'ז' => "זַיִן",
        'ח' => "חֵית",
        'ט' => "טֵית",
        'י' => "יוּד",
        'כ' => "כַּף",
        'ל' => "לָמֶד",
        'מ' => "מֵם",
        'נ' => "נוּן",
        'ס' => "סָמֶךְ",
        'ע' => "עַיִן",
        'פ' => "פֵּא",
        'צ' => "צָדִי",
        'ק' => "קוֹף",
        'ר' => "רֵישׁ",
        'ש' => "שִׁין",
        'ת' => "תָּו",
        _ => return None,
    })
}

fn expand_geresh_loanwords(text: &str) -> String {
    text.replace("ג'מיני", "<en>Gemini</en>")
        .replace("ג׳מיני", "<en>Gemini</en>")
        .replace("מנג'ר", "<en>Manager</en>")
        .replace("מנג׳ר", "<en>Manager</en>")
}

fn expand_dialogue_quotes(text: &str) -> String {
    let terminal = Regex::new(r#"(?u)(\S)\s*["“„”]\s*$"#).expect("valid regex");
    let opening = Regex::new(r#"(?u)\s*:?\s*["“„”]\s*"#).expect("valid regex");
    let closing = Regex::new(r#"(?u)(\S)\s*["“„”]"#).expect("valid regex");
    let text = terminal.replace_all(text, "$1.").into_owned();
    let text = opening.replace_all(&text, ", ").into_owned();
    closing.replace_all(&text, "$1, ").into_owned()
}

fn expand_lamed_before_latin(text: &str) -> String {
    Regex::new(r"(?u)(^|[^\u{0590}-\u{05ff}])ל[\u{0591}-\u{05C7}]*\s*[-–—‑]?\s*([A-Za-z0-9])")
        .expect("valid regex")
        .replace_all(text, "$1אל $2")
        .into_owned()
}

fn expand_alphanumeric_codes(text: &str, lang: &str) -> String {
    let token = Regex::new(r"(?i)[a-z0-9]+(?:[-_/][a-z0-9]+)*").expect("valid regex");
    token
        .replace_all(text, |caps: &Captures| {
            let value = &caps[0];
            let letters = value.chars().filter(|c| c.is_ascii_alphabetic()).count();
            let digits = value.chars().filter(|c| c.is_ascii_digit()).count();
            if letters == 0 || digits == 0 || (letters < 2 && digits < 2) {
                return value.to_owned();
            }
            let letter_block = value
                .chars()
                .filter(|c| c.is_ascii_alphabetic())
                .map(|c| c.to_ascii_uppercase().to_string())
                .collect::<Vec<_>>()
                .join(" ");
            let digits = value
                .split(['-', '_', '/'])
                .filter_map(|part| {
                    let digits: String = part.chars().filter(|c| c.is_ascii_digit()).collect();
                    (!digits.is_empty()).then(|| spell_digits_lang(&digits, lang))
                })
                .collect::<Vec<_>>()
                .join(" , ");
            let letters = if letter_block.is_empty() {
                String::new()
            } else {
                format!("<en>{letter_block}</en>")
            };
            mark_slow_segment(format!("{} {} .", letters, digits).trim())
        })
        .into_owned()
}

fn expand_list_markers(text: &str, lang: &str) -> String {
    let list = Regex::new(r"(?m)(^|[^\d/])(\d{1,2})\.\s+").expect("valid regex");
    list.replace_all(text, |caps: &Captures| {
        let n: u8 = caps[2].parse().unwrap_or_default();
        let word = if lang == "he" {
            list_number(n)
        } else {
            number_to_words(u16::from(n), lang)
        };
        format!("{}{}. ", &caps[1], word)
    })
    .into_owned()
}

fn expand_plus_sign(text: &str, lang: &str) -> String {
    let word = match lang {
        "he" => "פלוס",
        "es" => "más",
        "it" => "più",
        _ => "plus",
    };
    Regex::new(r"\s+\+\s+")
        .expect("valid regex")
        .replace_all(text, format!(" {word} "))
        .into_owned()
}

fn expand_phone_numbers(text: &str) -> String {
    let phone = Regex::new(r"(?x)(?:\*\d{2,}|0\d{0,2}-\d{6,8})").expect("valid regex");
    phone
        .replace_all(text, |caps: &Captures| {
            let raw = &caps[0];
            let prefix = if raw.starts_with('*') {
                "כוכבית "
            } else {
                ""
            };
            mark_slow_segment(format!("{prefix}{}", spell_digits(raw)))
        })
        .into_owned()
}

fn expand_times(text: &str) -> String {
    let time = Regex::new(r"([01]?\d|2[0-3]):([0-5]\d)").expect("valid regex");
    time.replace_all(text, |caps: &Captures| {
        let hour = caps[1].parse::<u16>().unwrap_or_default();
        let minute = caps[2].parse::<u16>().unwrap_or_default();
        let spoken = if minute == 0 {
            number_to_words(hour, "he")
        } else {
            format!(
                "{} ו{}",
                number_to_words(hour, "he"),
                number_to_words(minute, "he")
            )
        };
        mark_slow_segment(spoken)
    })
    .into_owned()
}

fn expand_dates(text: &str) -> String {
    let date = Regex::new(r"([0-3]?\d)[/.]([01]?\d)[/.](\d{2}|\d{4})").expect("valid regex");
    date.replace_all(text, |caps: &Captures| {
        let day = caps[1].parse::<u16>().unwrap_or_default();
        let month = caps[2].parse::<usize>().unwrap_or_default();
        let mut year = caps[3].parse::<u16>().unwrap_or_default();
        if caps[3].len() == 2 {
            year += if year < 70 { 2000 } else { 1900 };
        }
        let ordinals = [
            "",
            "לראשון",
            "לשני",
            "לשלישי",
            "לרביעי",
            "לחמישי",
            "לשישי",
            "לשביעי",
            "לשמיני",
            "לתשיעי",
            "לעשירי",
            "לאחד עשר",
            "לשנים עשר",
        ];
        if !(1..=31).contains(&day) || month == 0 || month >= ordinals.len() {
            caps[0].to_owned()
        } else {
            mark_slow_segment(format!(
                "{} {} {}",
                number_to_words(day, "he"),
                ordinals[month],
                number_to_words(year, "he")
            ))
        }
    })
    .into_owned()
}

fn expand_percent_symbols(text: &str, lang: &str) -> String {
    let word = match lang {
        "he" => "אחוז",
        "es" => "por ciento",
        "de" => "Prozent",
        "it" => "per cento",
        _ => "percent",
    };
    Regex::new(r"(\d+(?:[.,]\d+)?)\s*%")
        .expect("valid regex")
        .replace_all(text, format!("$1 {word}"))
        .into_owned()
}

fn expand_ratios(text: &str, lang: &str) -> String {
    let word = match lang {
        "he" => "ל",
        "de" => "zu",
        _ => "to",
    };
    Regex::new(r"(\d+)\s*:\s*(\d+)")
        .expect("valid regex")
        .replace_all(text, format!("$1 {word} $2"))
        .into_owned()
}

fn expand_numbers(text: &str, lang: &str) -> String {
    let numbers = Regex::new(r"\d+(?:[.,]\d+)?").expect("valid regex");
    numbers
        .replace_all(text, |caps: &Captures| {
            let raw = &caps[0];
            if raw.contains(['.', ',']) {
                let separator = if lang == "he" {
                    " נקודה "
                } else {
                    " point "
                };
                raw.replace(['.', ','], separator)
                    .split_whitespace()
                    .map(|piece| {
                        piece
                            .parse::<u16>()
                            .ok()
                            .map_or_else(|| piece.to_owned(), |n| number_to_words(n, lang))
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                raw.parse::<u16>()
                    .map(|n| number_to_words(n, lang))
                    .unwrap_or_else(|_| spell_digits_lang(raw, lang))
            }
        })
        .into_owned()
}

fn strip_silent_separator_tokens(text: &str) -> String {
    let text = Regex::new(r"(?u)([\u{0590}-\u{05ff}])[-–—‑]+([A-Za-z0-9])")
        .expect("valid regex")
        .replace_all(text, "$1 $2")
        .into_owned();
    let text = Regex::new(r"(?u)([A-Za-z0-9])[-–—‑]+([\u{0590}-\u{05ff}])")
        .expect("valid regex")
        .replace_all(&text, "$1 $2")
        .into_owned();
    let text = Regex::new(r"\s*[-–—‑]+\s*")
        .expect("valid regex")
        .replace_all(&text, " ")
        .into_owned();
    let text = Regex::new(r"\s*:+\s*")
        .expect("valid regex")
        .replace_all(&text, " ")
        .into_owned();
    Regex::new(r"\s+")
        .expect("valid regex")
        .replace_all(&text, " ")
        .trim()
        .to_owned()
}

fn is_nikud(character: char) -> bool {
    // Hebrew points + cantillation, including the hatama (ole) and meteg.
    ('\u{0591}'..='\u{05c7}').contains(&character)
}

fn normalize_hebrew_punctuation(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut output = String::with_capacity(text.len());
    for (index, character) in chars.iter().copied().enumerate() {
        // Niqqud sits between a letter and its geresh or gershayim.
        let previous = chars[..index].iter().rev().copied().find(|&c| !is_nikud(c));
        let next = chars.get(index + 1).copied();
        if matches!(character, '"' | '״')
            && previous.is_some_and(is_hebrew_letter)
            && next.is_some_and(is_hebrew_letter)
        {
            continue;
        }
        if character == '\''
            && matches!(previous, Some('ג' | 'צ' | 'ז'))
            && next.is_some_and(is_hebrew_letter)
        {
            output.push('׳');
            continue;
        }
        if character == '-'
            && previous.is_some_and(is_hebrew_letter)
            && next.is_some_and(is_hebrew_letter)
        {
            continue;
        }
        output.push(character);
    }
    output
}

fn normalize_punctuation(text: &str) -> String {
    let text = text.replace('…', ",");
    let dots = Regex::new(r"\.{2,}").expect("valid dot regex");
    let bangs = Regex::new(r"!{2,}").expect("valid exclamation regex");
    let questions = Regex::new(r"\?{2,}").expect("valid question regex");
    let commas = Regex::new(r",{2,}").expect("valid comma regex");
    let whitespace = Regex::new(r"\s+").expect("valid whitespace regex");
    let text = dots.replace_all(&text, ",");
    let text = bangs.replace_all(&text, "!");
    let text = questions.replace_all(&text, "?");
    let text = commas.replace_all(&text, ",");
    whitespace.replace_all(&text, " ").trim().to_owned()
}

fn expand_email(email: &str) -> String {
    let (local, domain) = email.split_once('@').expect("email regex contains @");
    let local = local.replace(['.', '_'], " dot ").replace('-', " dash ");
    let domain = domain
        .split('.')
        .map(|part| {
            if part.len() <= 2 && part.chars().all(|c| c.is_ascii_alphabetic()) {
                part.chars()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                part.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(" dot ");
    format!("{local} at {domain}")
}

fn expand_phone(phone: &str) -> String {
    let prefix = if phone.starts_with('*') {
        "כוכבית "
    } else {
        ""
    };
    format!("{prefix}{}.", spell_digits(phone))
}

fn expand_time(caps: &Captures) -> String {
    format!(
        "שעה {} ו{}",
        spell_digits(&format!("{:0>2}", &caps[1])),
        spell_digits(&caps[2])
    )
}

fn expand_date(caps: &Captures) -> String {
    let day = caps[1].parse::<u8>().unwrap_or(0);
    let month = caps[2].parse::<u8>().unwrap_or(0);
    let Some(month_name) = month_name(month) else {
        return caps[0].to_owned();
    };
    if !(1..=31).contains(&day) {
        return caps[0].to_owned();
    }
    format!(
        "{} ב{month_name} {}",
        spell_digits(&caps[1]),
        spell_digits(&caps[3])
    )
}

fn expand_identifier(token: &str) -> String {
    let letters: Vec<_> = token
        .chars()
        .filter(|character| character.is_ascii_alphabetic())
        .map(|character| character.to_ascii_uppercase())
        .collect();
    if letters.is_empty() || !token.chars().any(|character| character.is_ascii_digit()) {
        return token.to_owned();
    }
    let digit_groups = token
        .split(['-', '_', '/'])
        .filter(|part| part.chars().any(|character| character.is_ascii_digit()))
        .map(spell_digits)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    let letters = letters
        .into_iter()
        .map(|letter| letter.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    format!("{letters} {}.", digit_groups.join(", "))
}

fn spell_digits(text: &str) -> String {
    spell_digits_lang(text, "he")
}

fn spell_digits_lang(text: &str, lang: &str) -> String {
    let words = match lang {
        "he" => [
            "אפס",
            "אחת",
            "שתיים",
            "שלוש",
            "ארבע",
            "חמש",
            "שש",
            "שבע",
            "שמונה",
            "תשע",
        ],
        _ => [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        ],
    };
    text.chars()
        .filter_map(|digit| digit.to_digit(10).map(|digit| words[digit as usize]))
        .collect::<Vec<_>>()
        .join(" ")
}

fn number_to_words(number: u16, lang: &str) -> String {
    if lang != "he" {
        return match number {
            0 => "zero".to_owned(),
            1 => "one".to_owned(),
            2 => "two".to_owned(),
            3 => "three".to_owned(),
            4 => "four".to_owned(),
            5 => "five".to_owned(),
            6 => "six".to_owned(),
            7 => "seven".to_owned(),
            8 => "eight".to_owned(),
            9 => "nine".to_owned(),
            10 => "ten".to_owned(),
            11 => "eleven".to_owned(),
            12 => "twelve".to_owned(),
            13 => "thirteen".to_owned(),
            14 => "fourteen".to_owned(),
            15 => "fifteen".to_owned(),
            16 => "sixteen".to_owned(),
            17 => "seventeen".to_owned(),
            18 => "eighteen".to_owned(),
            19 => "nineteen".to_owned(),
            20 => "twenty".to_owned(),
            _ => spell_digits_lang(&number.to_string(), lang),
        };
    }
    let units = [
        "אפס",
        "אחת",
        "שתיים",
        "שלוש",
        "ארבע",
        "חמש",
        "שש",
        "שבע",
        "שמונה",
        "תשע",
        "עשר",
        "אחת עשרה",
        "שתים עשרה",
        "שלוש עשרה",
        "ארבע עשרה",
        "חמש עשרה",
        "שש עשרה",
        "שבע עשרה",
        "שמונה עשרה",
        "תשע עשרה",
    ];
    if number < 20 {
        return units[number as usize].to_owned();
    }
    let tens = [
        "",
        "",
        "עשרים",
        "שלושים",
        "ארבעים",
        "חמישים",
        "שישים",
        "שבעים",
        "שמונים",
        "תשעים",
    ];
    if number < 100 {
        let ten = number / 10;
        let remainder = number % 10;
        return if remainder == 0 {
            tens[ten as usize].to_owned()
        } else {
            format!("{} ו{}", tens[ten as usize], units[remainder as usize])
        };
    }
    if number < 1_000 {
        let hundred = number / 100;
        let remainder = number % 100;
        let head = match hundred {
            1 => "מאה".to_owned(),
            2 => "מאתיים".to_owned(),
            n => format!("{} מאות", number_to_words(n, "he")),
        };
        return if remainder == 0 {
            head
        } else {
            format!("{head} {}", number_to_words(remainder, "he"))
        };
    }
    if number < 10_000 {
        let thousand = number / 1_000;
        let remainder = number % 1_000;
        let head = if thousand == 1 {
            "אלף".to_owned()
        } else {
            format!("{} אלף", number_to_words(thousand, "he"))
        };
        return if remainder == 0 {
            head
        } else {
            format!("{head} {}", number_to_words(remainder, "he"))
        };
    }
    spell_digits_lang(&number.to_string(), "he")
}

fn list_number(number: u8) -> String {
    match number {
        1 => "אחד".to_owned(),
        2 => "שתיים".to_owned(),
        3 => "שלוש".to_owned(),
        4 => "ארבע".to_owned(),
        5 => "חמש".to_owned(),
        6 => "שש".to_owned(),
        7 => "שבע".to_owned(),
        8 => "שמונה".to_owned(),
        9 => "תשע".to_owned(),
        10 => "עשר".to_owned(),
        _ => spell_digits(&number.to_string()),
    }
}

fn month_name(month: u8) -> Option<&'static str> {
    Some(match month {
        1 => "ינואר",
        2 => "פברואר",
        3 => "מרץ",
        4 => "אפריל",
        5 => "מאי",
        6 => "יוני",
        7 => "יולי",
        8 => "אוגוסט",
        9 => "ספטמבר",
        10 => "אוקטובר",
        11 => "נובמבר",
        12 => "דצמבר",
        _ => return None,
    })
}

fn is_hebrew_letter(character: char) -> bool {
    ('\u{05d0}'..='\u{05ea}').contains(&character)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colon_pauses_and_answer_labels_keep_explicit_vowels() {
        assert_eq!(
            prepare_text_for_synthesis("לכן תשובה ב׳: אֶן קֶלְוִין", "he"),
            "לכן תשובה בֵּת. אֶן קֶלְוִין"
        );
        for quote in ["'", "׳", "’"] {
            assert_eq!(
                prepare_text_for_synthesis(&format!("תשובה ב{quote}"), "he"),
                "תשובה בֵּת"
            );
        }
        assert!(!prepare_text_for_synthesis("בשעה 12:30", "he").contains('.'));
        assert_eq!(prepare_text_for_synthesis("ג׳אז", "he"), "ג׳אז");
    }

    #[test]
    fn lone_letters_with_geresh_read_as_letter_names() {
        for (text, expected) in [
            ("דגניה ב׳ היא קיבוץ", "דגניה בֵּת היא קיבוץ"),
            ("דגניה ב׳, היא", "דגניה בֵּת, היא"),
            ("כיתה ג׳.", "כיתה גִּימֶל."),
            ("יום ב' וחלק ד’", "יום בֵּת וחלק דָּלֶת"),
            ("א׳ ב׳ ג׳", "אָלֶף בֵּת גִּימֶל"),
            ("פרק ת׳", "פרק תָּו"),
            ("\"ק׳\"", "\"קוֹף\""),
        ] {
            assert_eq!(expand_letter_labels(text), expected, "{text}");
        }
        // A letter after the geresh makes it a loanword, and a letter before
        // the lone letter makes it part of a longer word.
        for text in ["ג׳אז", "צ׳יפס", "ז׳קט", "ת׳ירטי", "מנג׳ר", "צה״ל", "ט''ו"]
        {
            assert_eq!(expand_letter_labels(text), text);
        }
        assert_eq!(prepare_text_for_synthesis("ג׳אז", "he"), "ג׳אז");
        assert_eq!(prepare_text_for_synthesis("צ׳יפס", "he"), "צ׳יפס");
        assert_eq!(
            prepare_text_for_synthesis("דגניה ב׳ היא קיבוץ.", "he"),
            "דגניה בֵּת היא קיבוץ."
        );
    }

    #[test]
    fn normalization_keeps_typed_nikud() {
        let text = normalize_for_speech("הַמְּנוֹרָה מאירה.");
        assert!(contains_nikud(&text));
        assert_eq!(text, "הַמְּנוֹרָה מאירה.");
        assert_eq!(
            prepare_text_for_synthesis("שָׁל֫וֹם", "he"),
            "שָׁל֫וֹם",
            "the hatama survives for RenikudPlus to read"
        );
    }

    #[test]
    fn diacritized_text_prepares_like_plain_text() {
        // Niqqud between a letter and its gershayim, geresh or hyphen must not
        // change how the text is prepared.
        assert_eq!(prepare_text_for_synthesis("צַהַ\"ל", "he"), "צַהַל");
        assert_eq!(prepare_text_for_synthesis("תשובה בַּ׳", "he"), "תשובה בֵּת");
        assert_eq!(
            prepare_text_for_synthesis("לַ-GPU", "he"),
            prepare_text_for_synthesis("ל-GPU", "he")
        );
    }

    #[test]
    fn strip_nikud_preserves_letters() {
        assert_eq!(strip_nikud("הַמְּנוֹרָה"), "המנורה");
    }

    #[test]
    fn normalizes_reference_codes_and_phone_numbers() {
        let text = normalize_for_speech("מספר TKT-90254, טלפון 03-5551234.");
        assert!(text.contains("T K T"));
        assert!(text.contains("אפס שלוש חמש חמש חמש"));
    }

    #[test]
    fn prepares_hebrew_reference_segments_and_structured_values() {
        let text = prepare_text_for_synthesis(
            "הג'מיני הגיע ל-GPU ב-08:15, קוד TKT-90254 הוא 50% הצלחה.",
            "he",
        );
        assert!(text.contains("<en>Gemini</en>"));
        assert!(text.contains("אל GPU"));
        assert!(text.contains(REF_CODE_MARK_OPEN));
        assert!(text.contains("אחוז"));
    }

    #[test]
    fn splits_slow_reference_segments() {
        let segments = split_prepared_by_reference_codes("רגיל 【אחת שתיים】.");
        assert_eq!(segments.len(), 2);
        assert!(!segments[0].is_reference_code);
        assert!(segments[1].is_reference_code);
        assert_eq!(segments[1].text, "אחת שתיים.");
    }

    #[test]
    fn expands_dialogue_and_list_markers() {
        let text = prepare_text_for_synthesis("1. אמר \"שלום\"", "he");
        assert!(
            text.starts_with("אחד."),
            "unexpected prepared text: {text:?}"
        );
        assert!(
            text.contains("אמר, שלום."),
            "unexpected prepared text: {text:?}"
        );
    }
}
