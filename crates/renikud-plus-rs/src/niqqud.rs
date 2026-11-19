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
/// Rafe: the mark that says a letter takes no dagesh — and, on a vowel letter,
/// that it is not a consonant.
pub const RAFE: char = '\u{05BF}';
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

/// Meteg (U+05BD): in phonikud's pointing, a vocal shva (shva na, /e/).
pub const METEG: char = '\u{05BD}';
/// Nikud haser (U+05AF, Masora circle): phonikud's mark for a silent letter.
pub const NIKUD_HASER: char = '\u{05AF}';
/// Phonikud's prefix bar: `וְ|הַ|יֶּלֶד` separates the prefix letters from the
/// stem. It is not read aloud.
pub const PREFIX_BAR: char = '|';
/// Prefix letters whose shva is vocal (/e/) when the prefix bar follows them.
const PREFIX_SHVA_LETTERS: [char; 4] = ['ו', 'כ', 'ל', 'ב'];

struct Pointed {
    /// Index of the letter in the bare text, in characters.
    index: usize,
    letter: char,
    marks: Vec<char>,
    /// A prefix bar followed this letter (and was dropped from the bare text).
    prefix: bool,
}

fn is_hebrew_letter(c: char) -> bool {
    ('א'..='ת').contains(&c)
}

fn has_vowel_sign(entry: &Pointed) -> bool {
    entry.marks.iter().any(|&m| point_vowels(m).is_some())
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
/// With `phonikud` off this is upstream RenikudPlus's `parse_niqqud`
/// (renikud_onnx 0.6.0) exactly. With it on it is the BlueV3 reference's
/// `site_flow.parse_points` (tests/golden/README.md, "Pointed Hebrew"), which
/// adds phonikud's marks:
///
/// 1. hatama U+05AB: the letter is its word's stressed letter. It follows the
///    vowel: a letter left with consonant ∅ and vowel ∅ (or none) — the
///    holam-male/shuruk vav, the hiriq/tsere-male yod, a silent final
///    alef/he — passes it to the adjacent letter before in the word; a
///    nikud-haser letter passes it to the adjacent letter after.
/// 2. meteg U+05BD: vowel `e` when the letter has no vowel sign or its signs
///    admit /e/ (shva); ignored next to any other vowel.
/// 3. the prefix bar `|` right after a letter carrying any mark and right
///    before a Hebrew letter is dropped; if that letter starts its word, is one
///    of וכלב and has shva: vowel `e`.
/// 4. nikud haser U+05AF: consonant ∅, vowel ∅.
/// 5. vav with holam haser for vav (U+05BA) as its only sign, or with holam as
///    its only sign right after an adjacent letter of the same word that has a
///    vowel sign: consonant `v`, vowel `o` (not a mater).
/// 6. qamats right before an adjacent letter with hataf qamats: vowel `o`.
/// 7. a bare vav in a pointed word is the consonant (`v`/`w`) only right after
///    an adjacent letter with a vowel sign (upstream: any bare vav, which reads
///    partially pointed שָׁלום as ʃalˈovm).
///
/// Only vowel signs make a word pointed, and a letter whose only mark is the
/// hatama counts as unmarked.
pub fn parse_niqqud(text: &str, phonikud: bool) -> (String, Constraints) {
    let chars: Vec<char> = text.chars().collect();
    let mut bare: Vec<char> = Vec::new();
    let mut letters: Vec<Pointed> = Vec::new();
    for (pos, &c) in chars.iter().enumerate() {
        let attached = letters
            .last()
            .is_some_and(|last| last.index + 1 == bare.len());
        if crate::text::is_niqqud(c) {
            if attached {
                letters.last_mut().expect("attached").marks.push(c);
            }
            continue;
        }
        if phonikud
            && c == PREFIX_BAR
            && attached
            && !letters.last().expect("attached").marks.is_empty()
            && chars
                .get(pos + 1)
                .is_some_and(|&next| is_hebrew_letter(next))
        {
            letters.last_mut().expect("attached").prefix = true;
            continue;
        }
        if is_hebrew(c) {
            letters.push(Pointed {
                index: bare.len(),
                letter: c,
                marks: Vec::new(),
                prefix: false,
            });
        }
        bare.push(c);
    }

    // A word counts as pointed once any of its letters carries a vowel sign;
    // only there is an UNMARKED letter informative.
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
        if has_vowel_sign(entry) {
            pointed_words.push(word_of[entry.index]);
        }
    }
    let word_pointed = |index: usize| pointed_words.contains(&word_of[index]);
    // The letter right before (`-1`) or after (`+1`): adjacent in the bare
    // text and in the same word.
    let neighbour = |rec_i: usize, step: isize| -> Option<usize> {
        let j = rec_i.checked_add_signed(step)?;
        let (here, there) = (letters[rec_i].index, letters.get(j)?.index);
        (there.checked_add_signed(-step) == Some(here) && word_of[there] == word_of[here])
            .then_some(j)
    };

    let mut cons: BTreeMap<usize, Vec<&'static str>> = BTreeMap::new();
    let mut vows: BTreeMap<usize, Vec<&'static str>> = BTreeMap::new();
    // Record indices of the letters carrying the hatama.
    let mut hatama: Vec<usize> = Vec::new();
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
        if phonikud && entry.marks.contains(&HATAMA) {
            hatama.push(rec_i);
        }
        if phonikud && entry.marks.contains(&NIKUD_HASER) {
            cons.insert(idx, vec![NONE]);
            vows.insert(idx, vec![NONE]);
            continue;
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
        if phonikud && letter == 'ו' && signs == [HOLAM_HASER_FOR_VAV] {
            cons.insert(idx, vec!["v"]);
            vows.insert(idx, vec!["o"]);
            continue;
        }
        if letter == 'ו' && signs.len() == 1 && point_vowels(signs[0]) == Some(&["o"][..]) {
            if phonikud && neighbour(rec_i, -1).is_some_and(|prev| has_vowel_sign(&letters[prev])) {
                cons.insert(idx, vec!["v"]);
                vows.insert(idx, vec!["o"]);
            } else {
                mater.push((rec_i, "o")); // holam male
            }
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
        if !phonikud {
            continue;
        }
        if entry.marks.contains(&METEG)
            && (signs.is_empty() || vows.get(&idx).is_some_and(|v| v.contains(&"e")))
        {
            vows.insert(idx, vec!["e"]);
        }
        if entry.prefix
            && PREFIX_SHVA_LETTERS.contains(&letter)
            && signs.contains(&SHEVA)
            && (rec_i == 0 || word_of[letters[rec_i - 1].index] != word_of[idx])
        {
            vows.insert(idx, vec!["e"]);
        }
        if signs.contains(&QAMATS)
            && neighbour(rec_i, 1).is_some_and(|next| letters[next].marks.contains(&HATAF_QAMATS))
        {
            vows.insert(idx, vec!["o"]);
        }
    }

    // Unmarked letters inside a pointed word.
    for (rec_i, entry) in letters.iter().enumerate() {
        let idx = entry.index;
        let unmarked = if phonikud {
            entry.marks.iter().all(|&m| m == HATAMA)
        } else {
            entry.marks.is_empty()
        };
        if !unmarked || !word_pointed(idx) {
            continue;
        }
        let previous = rec_i.checked_sub(1).map(|i| &letters[i]);
        let next = letters.get(rec_i + 1);
        let same_word = next.is_some_and(|next| word_of[next.index] == word_of[idx]);
        if entry.letter == 'ו' {
            // Upstream: a vowel vav carries a point, so a bare one is the
            // consonant. Phonikud: only after a letter with a vowel sign of its
            // own; after a bare one the vav may be that letter's vowel in
            // partially pointed text (שָׁלום).
            if !phonikud
                || previous
                    .is_some_and(|previous| previous.index + 1 == idx && has_vowel_sign(previous))
            {
                cons.insert(idx, vec!["v", "w"]);
            }
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
            }
            _ => {
                vows.insert(idx, vec![quality]);
            }
        }
        cons.insert(idx, vec![NONE]);
    }

    // The stress goes where the vowel is.
    let mut stressed: Vec<usize> = Vec::new();
    for &rec_i in &hatama {
        let mut rec_i = rec_i;
        if letters[rec_i].marks.contains(&NIKUD_HASER) {
            match neighbour(rec_i, 1) {
                Some(next) => rec_i = next,
                None => continue,
            }
        }
        let mut idx = letters[rec_i].index;
        let silent = cons.get(&idx).map(Vec::as_slice) == Some(&[NONE][..])
            && vows.get(&idx).is_none_or(|v| v.as_slice() == [NONE]);
        if silent && let Some(prev) = neighbour(rec_i, -1) {
            idx = letters[prev].index;
        }
        if !stressed.contains(&idx) {
            stressed.push(idx);
        }
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
    fn the_hatama_is_read_as_stress_only_with_phonikud_marks() {
        let nfd: String =
            unicode_normalization::UnicodeNormalization::nfd("שָׁלוֹ\u{05AB}ם").collect();
        let (_, with) = parse_niqqud(&nfd, true);
        let (_, without) = parse_niqqud(&nfd, false);
        // The hatama sits on the vav; the vowel moved back to the lamed, and so
        // does the stress.
        assert!(with[&1].stressed);
        assert!(without.values().all(|c| !c.stressed));
    }

    fn phonikud(text: &str) -> (Vec<char>, Constraints) {
        let nfd: String = unicode_normalization::UnicodeNormalization::nfd(text).collect();
        let (bare, constraints) = parse_niqqud(&nfd, true);
        (bare.chars().collect(), constraints)
    }

    fn at(constraints: &Constraints, index: usize) -> Constraint {
        constraints.get(&index).cloned().unwrap_or_default()
    }

    #[test]
    fn the_hatama_follows_the_vowel() {
        // Off a hiriq-male yod onto the letter before: שִׁי֫ר.
        let (_, c) = phonikud("שִׁי\u{05AB}ר");
        assert!(at(&c, 0).stressed && !at(&c, 1).stressed);
        // Off a silent final he (its only mark, the hatama, leaves it
        // unmarked, so it is silent): יַלְדָּה֫.
        let (_, c) = phonikud("יַלְדָּה\u{05AB}");
        assert!(at(&c, 2).stressed && !at(&c, 3).stressed);
        // Off a nikud-haser letter onto the letter after.
        let (_, c) = phonikud("אַ\u{05AF}\u{05AB}בָּא");
        assert_eq!(at(&c, 0).consonants, Some(vec![NONE]));
        assert_eq!(at(&c, 0).vowels, Some(vec![NONE]));
        assert!(at(&c, 1).stressed && !at(&c, 0).stressed);
    }

    #[test]
    fn meteg_prefix_bar_and_qamats_read_as_phonikud_writes_them() {
        // Meteg on a shva: vocal, /e/.
        let (_, c) = phonikud("שְׁ\u{05BD}מוֹ");
        assert_eq!(at(&c, 0).vowels, Some(vec!["e"]));
        // Prefix bar: dropped from the bare text; וְ| reads /e/.
        let (bare, c) = phonikud("וְ|הַיֶּלֶד");
        assert_eq!(bare.iter().collect::<String>(), "והילד");
        assert_eq!(at(&c, 0).vowels, Some(vec!["e"]));
        // Without the bar the shva stays open.
        let (_, c) = phonikud("וְהַיֶּלֶד");
        assert_eq!(at(&c, 0).vowels, Some(vec!["e", NONE]));
        // Qamats before hataf qamats: צָהֳרַיִם.
        let (_, c) = phonikud("צָהֳרַיִם");
        assert_eq!(at(&c, 0).vowels, Some(vec!["o"]));
    }

    #[test]
    fn the_vav_rules() {
        // Holam after a vowelled letter: consonantal vav, /vo/ (מִצְוֹת).
        let (_, c) = phonikud("מִצְוֹת");
        assert_eq!(at(&c, 2).consonants, Some(vec!["v"]));
        assert_eq!(at(&c, 2).vowels, Some(vec!["o"]));
        // Upstream reads it as holam male.
        let nfd: String = unicode_normalization::UnicodeNormalization::nfd("מִצְוֹת").collect();
        let (_, up) = parse_niqqud(&nfd, false);
        assert_eq!(up[&2].consonants, Some(vec![NONE]));
        // Holam haser for vav alone: consonantal too.
        let (_, c) = phonikud("עָו\u{05BA}ן");
        assert_eq!(at(&c, 1).consonants, Some(vec!["v"]));
        // Partial pointing: a bare vav after a bare letter is left open.
        let (_, c) = phonikud("שָׁלום");
        assert_eq!(at(&c, 2).consonants, None);
        let nfd: String = unicode_normalization::UnicodeNormalization::nfd("שָׁלום").collect();
        let (_, up) = parse_niqqud(&nfd, false);
        assert_eq!(up[&2].consonants, Some(vec!["v", "w"]));
    }

    #[test]
    fn a_bare_vav_in_a_pointed_word_is_a_consonant() {
        assert_eq!(constraint("מִצְוה", 'ו').consonants, Some(vec!["v", "w"]));
        // A word nobody pointed keeps every letter open.
        assert_eq!(constraint("מצוה", 'ו').consonants, None);
    }
}
