//! The aligner and the force-only runtime lexicon.
//!
//! The lexicon is off unless one is passed; where it matches it always wins.
//! Every entry is validated with [`align_word`] at load time, so a surface that
//! cannot possibly be read as its IPA is reported rather than shipped.

use std::collections::HashMap;
use std::path::Path;

use unicode_normalization::UnicodeNormalization;

use crate::pychars::{lstrip_chars, rstrip_chars, strip, strip_chars};
use crate::text::{normalize_graphemes, normalize_ipa};

pub const STRESS_MARK: char = 'ˈ';

/// Punctuation stripped off an IPA token before comparing it to a replacement.
pub const PUNCT_STRIP: &str = ".,?!:;\"'()-—…«»";

/// Characters stripped from the ENDS of a Hebrew token when looking it up.
///
/// Deliberately excludes `'` `"` `-`: those are part of the surface.
pub const LEX_EDGE_STRIP: &str = ".,?!:;()[]{}…«»־–—";

const ALIGN_VOWELS: [&str; 5] = ["a", "e", "i", "o", "u"];
const ALIGN_SPECIAL_CHARS: [char; 2] = ['\'', '"'];
const FURTIVE_PATAH_LETTER: char = 'ח';
const FINAL_HE_LETTER: char = 'ה';
const GERESH: char = '\'';

fn align_letter_consonants(letter: char) -> Option<&'static [&'static str]> {
    Some(match letter {
        'א' => &["ʔ", ""],
        'ב' => &["b", "v"],
        'ג' => &["ɡ", "dʒ"],
        'ד' => &["d"],
        'ה' => &["h", ""],
        'ו' => &["v", "w", ""],
        'ז' => &["z", "ʒ"],
        'ח' => &["χ"],
        'ט' => &["t"],
        'י' => &["j", ""],
        'כ' | 'ך' => &["k", "χ"],
        'ל' => &["l", ""],
        'מ' | 'ם' => &["m"],
        'נ' | 'ן' => &["n"],
        'ס' => &["s"],
        'ע' => &["ʔ", ""],
        'פ' | 'ף' => &["p", "f"],
        'צ' | 'ץ' => &["ts", "tʃ"],
        'ק' => &["k"],
        'ר' => &["ʁ"],
        'ש' => &["ʃ", "s", ""],
        'ת' => &["t"],
        _ => return None,
    })
}

fn ctx_free(_word: &[char], _i: usize) -> bool {
    true
}

fn ctx_geresh_after(word: &[char], i: usize) -> bool {
    word.get(i + 1) == Some(&GERESH)
}

fn ctx_dalet_gimel_geresh(word: &[char], i: usize) -> bool {
    word.get(i + 1) == Some(&'ג') && word.get(i + 2) == Some(&GERESH)
}

fn ctx_final_vav_after_alef(word: &[char], i: usize) -> bool {
    if i == 0 || word[i - 1] != 'א' {
        return false;
    }
    word[i + 1..]
        .iter()
        .all(|c| ALIGN_SPECIAL_CHARS.contains(c))
}

/// One optional extra aligner cell: a consonant a letter may take in a context
/// the standard aligner does not allow.
#[derive(Clone, Copy)]
pub struct AlignCell {
    pub letter: char,
    pub consonant: &'static str,
    pub predicate: fn(&[char], usize) -> bool,
    pub name: &'static str,
}

/// The optional cells, off by default; `align_cells = []` is the standard
/// aligner.
pub const EXTRA_CELLS: [AlignCell; 4] = [
    AlignCell {
        letter: 'ס',
        consonant: "z",
        predicate: ctx_free,
        name: "samz",
    },
    AlignCell {
        letter: 'ת',
        consonant: "s",
        predicate: ctx_geresh_after,
        name: "tafs",
    },
    AlignCell {
        letter: 'ד',
        consonant: "",
        predicate: ctx_dalet_gimel_geresh,
        name: "daled0",
    },
    AlignCell {
        letter: 'ו',
        consonant: "j",
        predicate: ctx_final_vav_after_alef,
        name: "vavj",
    },
];

/// `""`/nothing → standard; `"extended"` → all four cells; or a comma- or
/// space-separated list of cell names.
pub fn resolve_align_cells(spec: &str) -> anyhow::Result<Vec<AlignCell>> {
    if spec.is_empty() {
        return Ok(Vec::new());
    }
    if matches!(spec.to_lowercase().as_str(), "extended" | "w4") {
        return Ok(EXTRA_CELLS.to_vec());
    }
    spec.split([',', ' ', '\t', '\n'])
        .filter(|name| !name.is_empty())
        .map(|name| {
            EXTRA_CELLS
                .iter()
                .find(|cell| cell.name == name)
                .copied()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "unknown aligner cell `{name}`; known: daled0, samz, tafs, vavj"
                    )
                })
        })
        .collect()
}

/// DP alignment of a Hebrew surface to an IPA string, or `None` if illegal.
///
/// `cells` empty is the standard aligner; a non-empty `cells` widens it.
pub fn align_word(
    heb_word: &str,
    ipa_word: &str,
    cells: &[AlignCell],
) -> Option<Vec<(char, String)>> {
    let heb: Vec<char> = heb_word.chars().collect();
    let ipa: Vec<char> = ipa_word.chars().collect();
    if ipa.is_empty() || ipa.iter().filter(|&&c| c == STRESS_MARK).count() > 1 {
        return None;
    }
    let mut memo: HashMap<(usize, usize), Option<Vec<(char, String)>>> = HashMap::new();
    search(&heb, &ipa, cells, 0, 0, &mut memo)
}

fn starts_with(rest: &[char], prefix: &str) -> bool {
    let mut chars = prefix.chars();
    let mut i = 0;
    for c in &mut chars {
        if rest.get(i) != Some(&c) {
            return false;
        }
        i += 1;
    }
    true
}

fn search(
    heb: &[char],
    ipa: &[char],
    cells: &[AlignCell],
    h_idx: usize,
    i_idx: usize,
    memo: &mut HashMap<(usize, usize), Option<Vec<(char, String)>>>,
) -> Option<Vec<(char, String)>> {
    if let Some(cached) = memo.get(&(h_idx, i_idx)) {
        return cached.clone();
    }
    let result = search_uncached(heb, ipa, cells, h_idx, i_idx, memo);
    memo.insert((h_idx, i_idx), result.clone());
    result
}

fn search_uncached(
    heb: &[char],
    ipa: &[char],
    cells: &[AlignCell],
    h_idx: usize,
    i_idx: usize,
    memo: &mut HashMap<(usize, usize), Option<Vec<(char, String)>>>,
) -> Option<Vec<(char, String)>> {
    if h_idx == heb.len() && i_idx == ipa.len() {
        return Some(Vec::new());
    }
    if h_idx == heb.len() {
        return None;
    }
    let char_ = heb[h_idx];
    if ALIGN_SPECIAL_CHARS.contains(&char_) {
        let rest = search(heb, ipa, cells, h_idx + 1, i_idx, memo)?;
        let mut out = vec![(char_, String::new())];
        out.extend(rest);
        return Some(out);
    }
    let base = align_letter_consonants(char_)?;
    let mut allowed_cons: Vec<&str> = base.to_vec();
    for cell in cells.iter().filter(|cell| cell.letter == char_) {
        if !allowed_cons.contains(&cell.consonant) && (cell.predicate)(heb, h_idx) {
            allowed_cons.push(cell.consonant);
        }
    }
    let rest_ipa = &ipa[i_idx..];
    let is_final_he = char_ == FINAL_HE_LETTER
        && h_idx > 0
        && heb[h_idx + 1..]
            .iter()
            .all(|c| ALIGN_SPECIAL_CHARS.contains(c));

    for cons in &allowed_cons {
        if !cons.is_empty() && !starts_with(rest_ipa, cons) {
            continue;
        }
        let c_len = cons.chars().count();
        for has_stress in [true, false] {
            if has_stress && is_final_he {
                continue;
            }
            let mut s_len = 0;
            if has_stress {
                if c_len < rest_ipa.len() && rest_ipa[c_len] == STRESS_MARK {
                    s_len = 1;
                } else {
                    continue;
                }
            }
            for vowel in ALIGN_VOWELS.iter().copied().chain(std::iter::once("")) {
                if has_stress && vowel.is_empty() {
                    continue;
                }
                if !vowel.is_empty() && is_final_he && cons.is_empty() {
                    continue;
                }
                let v_start = c_len + s_len;
                let v_len = if !vowel.is_empty()
                    && starts_with(&rest_ipa[v_start.min(rest_ipa.len())..], vowel)
                {
                    vowel.chars().count()
                } else if vowel.is_empty() {
                    0
                } else {
                    continue;
                };
                let total_step = v_start + v_len;
                if let Some(rest) = search(heb, ipa, cells, h_idx + 1, i_idx + total_step, memo) {
                    let mut out = vec![(char_, rest_ipa[..total_step].iter().collect::<String>())];
                    out.extend(rest);
                    return Some(out);
                }
            }
        }
    }
    if char_ == FURTIVE_PATAH_LETTER {
        for cons in &allowed_cons {
            if cons.is_empty() {
                continue;
            }
            for has_stress in [true, false] {
                let prefix = if has_stress {
                    format!("{STRESS_MARK}a{cons}")
                } else {
                    format!("a{cons}")
                };
                if starts_with(rest_ipa, &prefix) {
                    let step = prefix.chars().count();
                    if let Some(rest) = search(heb, ipa, cells, h_idx + 1, i_idx + step, memo) {
                        let mut out = vec![(char_, rest_ipa[..step].iter().collect::<String>())];
                        out.extend(rest);
                        return Some(out);
                    }
                }
            }
        }
    }
    None
}

/// What to do with a lexicon entry that fails validation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OnInvalidEntry {
    /// Fail with the full list (the default, as upstream).
    #[default]
    Raise,
    /// Keep the entry out of the map and collect it on `rejected`.
    Report,
}

/// A user-supplied `surface -> IPA` map that ALWAYS wins where it matches.
///
/// It is keyed on the RAW surface **including separators** (gershayim, geresh,
/// hyphen), normalized with [`normalize_graphemes`].
pub struct ForceLexicon {
    entries: HashMap<String, String>,
    /// `(surface, ipa, reason)` for every entry that failed validation.
    pub rejected: Vec<(String, String, String)>,
}

impl ForceLexicon {
    /// Load from `surface -> IPA` pairs.
    pub fn new(
        pairs: impl IntoIterator<Item = (String, String)>,
        align_cells: &[AlignCell],
        on_invalid: OnInvalidEntry,
    ) -> anyhow::Result<Self> {
        let mut entries = HashMap::new();
        let mut rejected = Vec::new();
        for (raw_surface, raw_ipa) in pairs {
            let surface =
                strip(&normalize_graphemes(&raw_surface.nfd().collect::<String>())).to_owned();
            let ipa = normalize_ipa(strip(&raw_ipa));
            if surface.is_empty() || ipa.is_empty() {
                rejected.push((raw_surface, raw_ipa, "empty surface or IPA".to_owned()));
                continue;
            }
            match validate(&surface, &ipa, align_cells) {
                Some(reason) => rejected.push((raw_surface, raw_ipa, reason)),
                None => {
                    entries.insert(surface, ipa);
                }
            }
        }
        if !rejected.is_empty() && on_invalid == OnInvalidEntry::Raise {
            let lines = rejected
                .iter()
                .map(|(surface, ipa, why)| format!("  {surface:?} -> {ipa:?}: {why}"))
                .collect::<Vec<_>>()
                .join("\n");
            anyhow::bail!(
                "{} force-lexicon entr{} failed validation (use OnInvalidEntry::Report to \
                 collect instead of failing):\n{lines}",
                rejected.len(),
                if rejected.len() == 1 { "y" } else { "ies" }
            );
        }
        Ok(Self { entries, rejected })
    }

    /// Load from a `surface<TAB>IPA` file; blank lines and `#` comments are
    /// skipped.
    pub fn from_tsv(
        path: impl AsRef<Path>,
        align_cells: &[AlignCell],
        on_invalid: OnInvalidEntry,
    ) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|err| anyhow::anyhow!("read lexicon {}: {err}", path.display()))?;
        let mut pairs = Vec::new();
        for (lineno, line) in text.trim_start_matches('\u{FEFF}').lines().enumerate() {
            let line = line.trim_end_matches('\r');
            if strip(line).is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let mut parts = line.split('\t');
            let (Some(surface), Some(ipa)) = (parts.next(), parts.next()) else {
                anyhow::bail!(
                    "{}:{}: expected 'surface<TAB>IPA', got {line:?}",
                    path.display(),
                    lineno + 1
                );
            };
            pairs.push((surface.to_owned(), ipa.to_owned()));
        }
        Self::new(pairs, align_cells, on_invalid)
    }

    /// Match the raw whitespace-delimited token, then the token with edge
    /// punctuation stripped. Separators inside the surface are preserved.
    pub fn lookup(&self, token: &str) -> Option<&str> {
        if let Some(ipa) = self.entries.get(token) {
            return Some(ipa);
        }
        let stripped = strip_chars(token, LEX_EDGE_STRIP);
        if stripped.is_empty() {
            return None;
        }
        self.entries.get(stripped).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// `None` when the entry aligns; otherwise the reason it does not.
fn validate(surface: &str, ipa: &str, align_cells: &[AlignCell]) -> Option<String> {
    if align_word(surface, ipa, align_cells).is_some() {
        return None;
    }
    if surface.contains('-') {
        // A hyphen is a real key character but the aligner has no rule for it:
        // validate the compound segment by segment, which requires the IPA to
        // carry the same number of separators.
        let hs: Vec<&str> = surface.split('-').collect();
        let ps: Vec<&str> = ipa
            .split(|c: char| c == '-' || crate::pychars::is_space(c))
            .collect();
        if hs.len() != ps.len() {
            return Some(format!(
                "hyphenated surface has {} segments but the IPA has {} — split the IPA on '-' \
                 or a space to validate it",
                hs.len(),
                ps.len()
            ));
        }
        for (h, p) in hs.iter().zip(&ps) {
            if align_word(h, p, align_cells).is_none() {
                return Some(format!("segment {h:?} does not align to {p:?}"));
            }
        }
        return None;
    }
    let unknown: Vec<char> = {
        let mut seen: Vec<char> = surface
            .chars()
            .filter(|c| align_letter_consonants(*c).is_none() && !ALIGN_SPECIAL_CHARS.contains(c))
            .collect();
        seen.sort_unstable();
        seen.dedup();
        seen
    };
    if !unknown.is_empty() {
        return Some(format!(
            "surface contains character(s) the aligner has no rule for: {:?}",
            unknown.into_iter().collect::<String>()
        ));
    }
    Some("IPA does not align to the surface under align_word".to_owned())
}

/// One word the lexicon overrode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Forced {
    pub word_index: usize,
    pub surface: String,
    pub from: String,
    pub to: String,
}

/// Overwrite every output word whose Hebrew token is in `lex`.
///
/// Word alignment is positional (whitespace-delimited tokens of the Hebrew vs
/// of the IPA); if the two counts disagree the override is skipped for that
/// string rather than guessed. Leading/trailing IPA punctuation on the replaced
/// token is preserved.
pub fn apply_force_lexicon(text: &str, ipa: &str, lex: &ForceLexicon) -> (String, Vec<Forced>) {
    let heb_tokens = crate::pychars::split_whitespace(text);
    let mut words: Vec<String> = crate::pychars::split_whitespace(ipa)
        .into_iter()
        .map(str::to_owned)
        .collect();
    let mut forced = Vec::new();
    if heb_tokens.len() != words.len() {
        return (ipa.to_owned(), forced);
    }
    for (wi, tok_heb) in heb_tokens.iter().enumerate() {
        let Some(repl) = lex.lookup(tok_heb) else {
            continue;
        };
        let token = words[wi].clone();
        let lead = token.chars().count() - lstrip_chars(&token, PUNCT_STRIP).chars().count();
        let trail = token.chars().count() - rstrip_chars(&token, PUNCT_STRIP).chars().count();
        let core: String = token
            .chars()
            .skip(lead)
            .take(token.chars().count() - lead - trail)
            .collect();
        if core == repl {
            continue;
        }
        let head: String = token.chars().take(lead).collect();
        let tail: String = token
            .chars()
            .skip(token.chars().count() - trail)
            .collect::<String>();
        words[wi] = format!("{head}{repl}{tail}");
        forced.push(Forced {
            word_index: wi,
            surface: (*tok_heb).to_owned(),
            from: core,
            to: repl.to_owned(),
        });
    }
    (words.join(" "), forced)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_aligner_accepts_a_real_reading_and_rejects_a_wrong_one() {
        assert!(align_word("סבתא", "sˈavta", &[]).is_some());
        assert!(align_word("סבתא", "ɡˈavta", &[]).is_none());
        // Two stress marks are never legal.
        assert!(align_word("סבתא", "sˈavtˈa", &[]).is_none());
        // Furtive patah under a final het.
        assert!(align_word("רוח", "ʁˈuaχ", &[]).is_some());
    }

    #[test]
    fn extra_cells_widen_the_aligner() {
        assert!(align_word("ת'", "s", &[]).is_none());
        let cells = resolve_align_cells("tafs").unwrap();
        assert!(align_word("ת'", "s", &cells).is_some());
        assert_eq!(resolve_align_cells("extended").unwrap().len(), 4);
        assert!(resolve_align_cells("nope").is_err());
    }

    #[test]
    fn a_lexicon_validates_its_entries() {
        let good = ForceLexicon::new(
            [("סבתא".to_owned(), "sˈavta".to_owned())],
            &[],
            OnInvalidEntry::Raise,
        )
        .unwrap();
        assert_eq!(good.lookup("סבתא,"), Some("sˈavta"));
        assert_eq!(good.lookup("סבא"), None);
        assert!(
            ForceLexicon::new(
                [("סבתא".to_owned(), "ɡˈavta".to_owned())],
                &[],
                OnInvalidEntry::Raise
            )
            .is_err()
        );
        let reported = ForceLexicon::new(
            [("סבתא".to_owned(), "ɡˈavta".to_owned())],
            &[],
            OnInvalidEntry::Report,
        )
        .unwrap();
        assert_eq!(reported.len(), 0);
        assert_eq!(reported.rejected.len(), 1);
    }

    #[test]
    fn forcing_replaces_the_word_and_keeps_its_punctuation() {
        let lex = ForceLexicon::new(
            [("סבתא".to_owned(), "sˈavta".to_owned())],
            &[],
            OnInvalidEntry::Raise,
        )
        .unwrap();
        let (out, forced) = apply_force_lexicon("שלום סבתא.", "ʃalˈom savtˈa.", &lex);
        assert_eq!(out, "ʃalˈom sˈavta.");
        assert_eq!(forced.len(), 1);
        // A word-count mismatch skips the override rather than guessing.
        let (out, forced) = apply_force_lexicon("שלום סבתא.", "ʃalˈom", &lex);
        assert_eq!(out, "ʃalˈom");
        assert!(forced.is_empty());
    }
}
