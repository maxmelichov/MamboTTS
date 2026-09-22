//! Reading niqqud off the input, and writing it back out.
//!
//! [`parse_niqqud`] turns each vowel sign, dagesh qal and shin/sin dot into a
//! hard constraint on the decode, so pointed input disambiguates what the
//! consonantal skeleton cannot. The point tables at the bottom are the other
//! direction: how `vocalize` renders a predicted (consonant, vowel) pair.

use std::collections::BTreeMap;

use crate::pychars;
use crate::text::is_hebrew;

pub const PATAH: char = '\u{05B7}';
pub const SEGOL: char = '\u{05B6}';
pub const TSERE: char = '\u{05B5}';
pub const HIRIQ: char = '\u{05B4}';
pub const HOLAM: char = '\u{05B9}';
pub const QUBUTS: char = '\u{05BB}';
pub const QAMATS: char = '\u{05B8}';
pub const SHEVA: char = '\u{05B0}';
pub const HATAF_SEGOL: char = '\u{05B1}';
pub const HATAF_PATAH: char = '\u{05B2}';
pub const HATAF_QAMATS: char = '\u{05B3}';
pub const HOLAM_HASER_FOR_VAV: char = '\u{05BA}';
pub const QAMATS_QATAN: char = '\u{05C7}';
pub const DAGESH: char = '\u{05BC}';
pub const SHIN_DOT: char = '\u{05C1}';
pub const SIN_DOT: char = '\u{05C2}';

/// Hebrew accent "ole" (U+05AB). Niqqud has no stress mark, so MamboTTS writes
/// the predicted stress with this one and reads it back here; upstream
/// RenikudPlus neither writes nor reads it.
pub const HATAMA: char = '\u{05AB}';

/// The label the model uses for "no consonant" / "no vowel".
pub const NONE: &str = "∅";

/// What the points on one letter allow the decoder to pick.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Constraint {
    /// Consonant labels this letter may take, or `None` where the pointing says
    /// nothing.
    pub consonants: Option<Vec<&'static str>>,
    /// Vowel labels this letter may take, or `None`.
    pub vowels: Option<Vec<&'static str>>,
    /// The letter carries the hatama: this is the stressed letter of its word.
    pub stressed: bool,
}

/// Constraints keyed by character index into the bare (unpointed) text.
pub type Constraints = BTreeMap<usize, Constraint>;

/// The vowel classes a sign admits.
///
/// Two signs stay ambiguous on purpose and are left to the model: qamats,
/// because qamats qatan (/o/, as in כָּל) is written with the same sign in most
/// pointed text, and shva, because shva na is /e/ while shva nah is nothing.
fn point_vowels(c: char) -> Option<&'static [&'static str]> {
    Some(match c {
        PATAH | HATAF_PATAH => &["a"],
        QAMATS => &["a", "o"],
        QAMATS_QATAN | HATAF_QAMATS => &["o"],
        SEGOL | TSERE | HATAF_SEGOL => &["e"],
        HIRIQ => &["i"],
        HOLAM | HOLAM_HASER_FOR_VAV => &["o"],
        QUBUTS => &["u"],
        SHEVA => &[NONE, "e"],
        _ => return None,
    })
}

/// Dagesh qal: present → stop, absent on an otherwise pointed letter →
/// fricative.
fn dagesh_hard(letter: char) -> Option<&'static str> {
    Some(match letter {
        'ב' => "b",
        'כ' | 'ך' => "k",
        'פ' | 'ף' => "p",
        _ => return None,
    })
}

fn dagesh_soft(letter: char) -> Option<&'static str> {
    Some(match letter {
        'ב' => "v",
        'כ' | 'ך' => "χ",
        'פ' | 'ף' => "f",
        _ => return None,
    })
}

struct Pointed {
    /// Index of the letter in the bare text, in characters.
    index: usize,
    letter: char,
    marks: Vec<char>,
}

/// Split pointed text into its skeleton and the readings its points force.
///
/// Returns `(bare_text, constraints)`, where `constraints` maps a character
/// index into `bare_text` to the labels that index is allowed. Input must be
/// NFD.
///
/// Matres lectionis are resolved to the model's own label convention: it hangs
/// the vowel on the consonant and leaves the vowel letter silent, so a holam
/// male or shuruk vav (וֹ / וּ) pushes its /o/ or /u/ back onto the preceding
/// letter and takes ∅/∅ itself — the inverse of the fixup `vocalize` applies.
///
/// `read_hatama` additionally reads U+05AB as "this letter is the stressed one
/// of its word" (a MamboTTS extension; upstream ignores the mark).
pub fn parse_niqqud(text: &str, read_hatama: bool) -> (String, Constraints) {
    let mut bare: Vec<char> = Vec::new();
    let mut letters: Vec<Pointed> = Vec::new();
    for c in text.chars() {
        if crate::text::is_niqqud(c) {
            if let Some(last) = letters.last_mut()
                && last.index + 1 == bare.len()
            {
                last.marks.push(c);
            }
            continue;
        }
        if is_hebrew(c) {
            letters.push(Pointed {
                index: bare.len(),
                letter: c,
                marks: Vec::new(),
            });
        }
        bare.push(c);
    }

    // A word counts as pointed once any of its letters carries a vowel sign;
    // only there is an UNMARKED letter informative (in pointed text a vowel vav
    // always carries holam or dagesh, so a bare vav is the consonant).
    let mut word_of: Vec<usize> = Vec::with_capacity(bare.len());
    let mut word = 0usize;
    for &c in &bare {
        if pychars::is_space(c) {
            word += 1;
        }
        word_of.push(word);
    }
    let mut pointed_words: Vec<usize> = Vec::new();
    for entry in &letters {
        if entry.marks.iter().any(|&m| point_vowels(m).is_some()) {
            pointed_words.push(word_of[entry.index]);
        }
    }
    let word_pointed = |index: usize| pointed_words.contains(&word_of[index]);

    let mut cons: BTreeMap<usize, Vec<&'static str>> = BTreeMap::new();
    let mut vows: BTreeMap<usize, Vec<&'static str>> = BTreeMap::new();
    let mut stressed: Vec<usize> = Vec::new();
    // (record index, "o" | "u")
    let mut mater: Vec<(usize, &'static str)> = Vec::new();

    for (rec_i, entry) in letters.iter().enumerate() {
        let idx = entry.index;
        let letter = entry.letter;
        let signs: Vec<char> = entry
            .marks
            .iter()
            .copied()
            .filter(|&m| point_vowels(m).is_some())
            .collect();
        let dagesh = entry.marks.contains(&DAGESH);
        if read_hatama && entry.marks.contains(&HATAMA) {
            stressed.push(idx);
        }

        if letter == 'ש' {
            if entry.marks.contains(&SHIN_DOT) {
                cons.insert(idx, vec!["ʃ"]);
            } else if entry.marks.contains(&SIN_DOT) {
                cons.insert(idx, vec!["s"]);
            }
        } else if letter == 'ה' && dagesh {
            cons.insert(idx, vec!["h"]); // mapiq: consonantal he
        } else if let Some(hard) = dagesh_hard(letter) {
            if dagesh {
                cons.insert(idx, vec![hard]);
            } else if !signs.is_empty() {
                // Pointed, yet no dagesh: in pointed text that is the fricative.
                cons.insert(idx, vec![dagesh_soft(letter).expect("paired table")]);
            }
        }

        if letter == 'ו' && signs.is_empty() && dagesh {
            mater.push((rec_i, "u")); // shuruk
            continue;
        }
        if letter == 'ו' && signs.len() == 1 && point_vowels(signs[0]) == Some(&["o"][..]) {
            mater.push((rec_i, "o")); // holam male
            continue;
        }
        if let Some(&first) = signs.first() {
            let mut allowed: Vec<&'static str> = point_vowels(first).expect("vowel sign").to_vec();
            for sign in &signs[1..] {
                let next = point_vowels(*sign).expect("vowel sign");
                allowed.retain(|label| next.contains(label));
            }
            if !allowed.is_empty() {
                allowed.sort_unstable();
                vows.insert(idx, allowed);
            }
        }
    }

    // Unmarked letters inside a pointed word.
    for (rec_i, entry) in letters.iter().enumerate() {
        let idx = entry.index;
        if !entry.marks.is_empty() || !word_pointed(idx) {
            continue;
        }
        let previous = rec_i.checked_sub(1).map(|i| &letters[i]);
        let next = letters.get(rec_i + 1);
        let same_word = next.is_some_and(|next| word_of[next.index] == word_of[idx]);
        if entry.letter == 'ו' {
            // A vowel vav carries a point; this one does not, so it is a consonant.
            cons.insert(idx, vec!["v", "w"]);
        } else if entry.letter == 'י'
            && previous.is_some_and(|previous| {
                previous.index + 1 == idx
                    && matches!(
                        vows.get(&previous.index).map(Vec::as_slice),
                        Some(["i"]) | Some(["e"])
                    )
            })
        {
            cons.insert(idx, vec![NONE]); // hiriq / tsere male
            vows.insert(idx, vec![NONE]);
        } else if matches!(entry.letter, 'א' | 'ה') && !same_word {
            cons.insert(idx, vec![NONE]); // silent word-final alef / he (no mapiq)
        }
    }

    for &(rec_i, quality) in &mater {
        let idx = letters[rec_i].index;
        let previous = rec_i.checked_sub(1).map(|i| letters[i].index);
        // The vowel lands on the letter before, when that letter is adjacent and
        // has no vowel of its own; otherwise the vav keeps it.
        match previous {
            Some(previous) if previous + 1 == idx && !vows.contains_key(&previous) => {
                vows.insert(previous, vec![quality]);
                vows.insert(idx, vec![NONE]);
                if let Some(position) = stressed.iter().position(|&s| s == idx) {
                    // The vowel moved, and the stress on it moves along.
                    stressed[position] = previous;
                }
            }
            _ => {
                vows.insert(idx, vec![quality]);
            }
        }
        cons.insert(idx, vec![NONE]);
    }

    let mut constraints = Constraints::new();
    for idx in cons
        .keys()
        .chain(vows.keys())
        .chain(stressed.iter())
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
    {
        constraints.insert(
            idx,
            Constraint {
                consonants: cons.get(&idx).cloned(),
                vowels: vows.get(&idx).cloned(),
                stressed: stressed.contains(&idx),
            },
        );
    }
    (bare.into_iter().collect(), constraints)
}

// --------------------------------------------------------------------------
// Niqqud rendering (`G2P::vocalize`)
// --------------------------------------------------------------------------

/// Each of the model's five predicted vowel qualities as one representative
/// sign; signs that share a sound (patah/qamats, tsere/segol) collapse to one,
/// which is lossless for pronunciation.
pub(crate) fn niqqud_vowel(vowel: &str) -> Option<char> {
    Some(match vowel {
        "a" => PATAH,
        "e" => SEGOL,
        "i" => HIRIQ,
        "o" => HOLAM,
        "u" => QUBUTS,
        _ => return None,
    })
}

/// Dagesh or shin/sin dot implied by the consonant the model chose for `letter`.
pub(crate) fn consonant_point(letter: char, consonant: &str) -> Option<char> {
    if letter == 'ש' {
        return Some(if consonant == "ʃ" { SHIN_DOT } else { SIN_DOT });
    }
    match (letter, consonant) {
        ('ב', "b") | ('כ' | 'ך', "k") | ('פ' | 'ף', "p") => Some(DAGESH),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constraint(text: &str, letter: char) -> Constraint {
        let nfd: String = unicode_normalization::UnicodeNormalization::nfd(text).collect();
        let (bare, constraints) = parse_niqqud(&nfd, false);
        let index = bare.chars().position(|c| c == letter).expect("letter");
        constraints.get(&index).cloned().unwrap_or_default()
    }

    #[test]
    fn the_four_sefer_pointings_constrain_what_they_should() {
        // סֵפֶר: tsere + segol, no dagesh -> /e/, /e/.
        assert_eq!(constraint("סֵפֶר", 'ס').vowels, Some(vec!["e"]));
        assert_eq!(constraint("סֵפֶר", 'פ').consonants, Some(vec!["f"]));
        // סַפָּר: dagesh in the pe -> /p/; qamats stays ambiguous.
        assert_eq!(constraint("סַפָּר", 'פ').consonants, Some(vec!["p"]));
        assert_eq!(constraint("סַפָּר", 'פ').vowels, Some(vec!["a", "o"]));
        assert_eq!(constraint("סָפַר", 'ס').vowels, Some(vec!["a", "o"]));
        assert_eq!(constraint("סִפֵּר", 'ס').vowels, Some(vec!["i"]));
    }

    #[test]
    fn a_holam_male_vav_pushes_its_vowel_back() {
        let nfd: String = unicode_normalization::UnicodeNormalization::nfd("שָׁלוֹם").collect();
        let (bare, constraints) = parse_niqqud(&nfd, false);
        assert_eq!(bare, "שלום");
        let lamed = bare.chars().position(|c| c == 'ל').expect("lamed");
        assert_eq!(constraints[&lamed].vowels, Some(vec!["o"]));
        assert_eq!(constraints[&(lamed + 1)].vowels, Some(vec![NONE]));
        assert_eq!(constraints[&(lamed + 1)].consonants, Some(vec![NONE]));
        assert_eq!(constraints[&0].consonants, Some(vec!["ʃ"]));
    }

    #[test]
    fn the_hatama_is_read_as_stress_only_when_asked() {
        let nfd: String =
            unicode_normalization::UnicodeNormalization::nfd("שָׁלוֹ\u{05AB}ם").collect();
        let (_, with) = parse_niqqud(&nfd, true);
        let (_, without) = parse_niqqud(&nfd, false);
        // The hatama sits on the vav; the vowel moved back to the lamed, and so
        // does the stress.
        assert!(with[&1].stressed);
        assert!(without.values().all(|c| !c.stressed));
    }

    #[test]
    fn a_bare_vav_in_a_pointed_word_is_a_consonant() {
        assert_eq!(constraint("מִצְוה", 'ו').consonants, Some(vec!["v", "w"]));
        // A word nobody pointed keeps every letter open.
        assert_eq!(constraint("מצוה", 'ו').consonants, None);
    }
}
