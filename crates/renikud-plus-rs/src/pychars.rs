//! The handful of Python character predicates the upstream package relies on.
//!
//! The port has to agree with CPython character by character, and Rust's own
//! predicates do not: [`char::is_alphabetic`] also accepts combining marks and
//! letter-like numerals, and Rust has no equivalent of `str.isdigit()` at all.
//! The tables below were generated from CPython 3.12 (Unicode 15.0.0), which is
//! the interpreter the upstream package targets.

/// `str.isspace()`: bidi class WS/B/S or category Zs.
///
/// Rust's `char::is_whitespace` differs only in the C1-adjacent separators
/// U+001C..U+001F, which Python counts as whitespace and Unicode does not.
pub fn is_space(c: char) -> bool {
    matches!(c, '\u{1C}'..='\u{1F}') || c.is_whitespace()
}

/// `str.isalpha()`: general category L*.
pub fn is_alpha(c: char) -> bool {
    use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};
    c.general_category_group() == GeneralCategoryGroup::Letter
}

/// The code point of the zero of every decimal-digit run (category Nd).
///
/// Every Nd block is a contiguous run of ten code points in ascending value, so
/// a digit's value is its distance from the zero of its run.
const DECIMAL_ZEROS: [u32; 68] = [
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66,
    0xDE6, 0xE50, 0xED0, 0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90,
    0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10,
    0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650,
    0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16A60, 0x16AC0,
    0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E950,
    0x1FBF0,
];

/// Digit characters outside category Nd: `str.isdigit()` accepts these (they
/// carry Numeric_Type=Digit), while the regex `\d` and `int()` do not.
const NON_DECIMAL_DIGITS: [(u32, u32); 20] = [
    (0xB2, 0xB3),
    (0xB9, 0xB9),
    (0x1369, 0x1371),
    (0x19DA, 0x19DA),
    (0x2070, 0x2070),
    (0x2074, 0x2079),
    (0x2080, 0x2089),
    (0x2460, 0x2468),
    (0x2474, 0x247C),
    (0x2488, 0x2490),
    (0x24EA, 0x24EA),
    (0x24F5, 0x24FD),
    (0x24FF, 0x24FF),
    (0x2776, 0x277E),
    (0x2780, 0x2788),
    (0x278A, 0x2792),
    (0x10A40, 0x10A43),
    (0x10E60, 0x10E68),
    (0x11052, 0x1105A),
    (0x1F100, 0x1F10A),
];

/// The value of a decimal digit (category Nd), as `int()` reads it.
pub fn decimal_value(c: char) -> Option<u32> {
    let cp = c as u32;
    if c.is_ascii_digit() {
        return Some(cp - 0x30);
    }
    DECIMAL_ZEROS
        .iter()
        .find(|&&zero| (zero..zero + 10).contains(&cp))
        .map(|&zero| cp - zero)
}

/// The regex `\d` for `str` patterns: category Nd.
pub fn is_decimal(c: char) -> bool {
    decimal_value(c).is_some()
}

/// `str.isdigit()`: Numeric_Type of Decimal or Digit.
pub fn is_digit(c: char) -> bool {
    let cp = c as u32;
    is_decimal(c)
        || NON_DECIMAL_DIGITS
            .iter()
            .any(|&(lo, hi)| (lo..=hi).contains(&cp))
}

/// `str.split()`: split on runs of whitespace, dropping empty pieces.
pub fn split_whitespace(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        if is_space(c) {
            if let Some(s) = start.take() {
                out.push(&text[s..i]);
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        out.push(&text[s..]);
    }
    out
}

/// `str.strip()`.
pub fn strip(text: &str) -> &str {
    text.trim_matches(is_space)
}

/// `str.strip(chars)`.
pub fn strip_chars<'a>(text: &'a str, chars: &str) -> &'a str {
    text.trim_matches(|c| chars.contains(c))
}

/// `str.lstrip(chars)`.
pub fn lstrip_chars<'a>(text: &'a str, chars: &str) -> &'a str {
    text.trim_start_matches(|c| chars.contains(c))
}

/// `str.rstrip(chars)`.
pub fn rstrip_chars<'a>(text: &'a str, chars: &str) -> &'a str {
    text.trim_end_matches(|c| chars.contains(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_digit_classes() {
        assert_eq!(decimal_value('7'), Some(7));
        // Arabic-Indic digits are decimals too, and `int()` reads their value.
        assert_eq!(decimal_value('\u{663}'), Some(3));
        assert!(is_digit('\u{B2}') && !is_decimal('\u{B2}')); // superscript two
        assert!(!is_digit('\u{2160}')); // Roman numeral one is a numeral, not a digit
    }

    #[test]
    fn python_letter_and_space_classes() {
        assert!(is_alpha('a') && is_alpha('\u{5D0}'));
        // Combining marks are alphabetic to Rust but not to Python.
        assert!(!is_alpha('\u{5B8}') && '\u{5B8}'.is_alphabetic());
        assert!(!is_alpha('\u{2160}')); // Nl
        assert!(is_space('\u{1C}') && !'\u{1C}'.is_whitespace());
    }

    #[test]
    fn splitting_matches_python() {
        assert_eq!(split_whitespace("  a\u{1C}b  c "), vec!["a", "b", "c"]);
        assert_eq!(split_whitespace("   "), Vec::<&str>::new());
    }
}
