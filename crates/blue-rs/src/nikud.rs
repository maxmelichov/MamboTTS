//! Deterministic Modern Hebrew nikud → IPA.
//!
//! RenikudPlus infers vowels for unvocalized text and, on ambiguous words,
//! sometimes infers the wrong word. The one tool a writer has to correct that
//! is to vocalize the word, and until now those marks were stripped before
//! phonemization. This module reads them instead. It is rule-based, so a
//! vocalized word always comes out the way it was written, offline, with no
//! model in the loop.
//!
//! The rules follow Israeli Hebrew as it is actually spoken: no vowel length,
//! ע and א both as a glottal stop, ר as ʁ, ו as v. Three things are
//! conventions rather than rules, and each has an escape hatch:
//!
//! - Shva. Whether a shva is spoken (`e`) or silent is only partly
//!   predictable from spelling. The rules below cover the common cases; a
//!   writer who wants a shva spoken can write a segol instead.
//! - Kamatz. Ordinary kamatz (U+05B8) is read as `a`. The `o` reading, kamatz
//!   katan, is not recoverable from the mark alone, so write it as the
//!   dedicated kamatz katan (U+05C7) where it is meant.
//! - Stress. Hebrew is mostly ultimate, with a penultimate class (segolates,
//!   furtive patach, the בַּיִת pattern). A meteg (U+05BD) under a letter marks
//!   that syllable as stressed explicitly and overrides the heuristic.
//!
//! Output uses the same conventions RenikudPlus does, so the two sources can
//! sit side by side in one sentence: `ˈ` immediately before the stressed
//! vowel, `ts` for צ, `χ` for ח and כ, `ʔ` for א and ע.

use anyhow::Result;

use crate::handling::{contains_nikud, is_hebrew_letter};

// Points. Names follow Unicode.
const SHVA: char = '\u{05B0}';
const HATAF_SEGOL: char = '\u{05B1}';
const HATAF_PATACH: char = '\u{05B2}';
const HATAF_KAMATZ: char = '\u{05B3}';
const HIRIQ: char = '\u{05B4}';
const TSERE: char = '\u{05B5}';
const SEGOL: char = '\u{05B6}';
const PATACH: char = '\u{05B7}';
const KAMATZ: char = '\u{05B8}';
const HOLAM: char = '\u{05B9}';
const HOLAM_HASER: char = '\u{05BA}';
const KUBUTZ: char = '\u{05BB}';
const DAGESH: char = '\u{05BC}';
const METEG: char = '\u{05BD}';
const RAFE: char = '\u{05BF}';
const SHIN_DOT: char = '\u{05C1}';
const SIN_DOT: char = '\u{05C2}';
const KAMATZ_KATAN: char = '\u{05C7}';
const GERESH: char = '\u{05F3}';

const STRESS: char = 'ˈ';

/// One letter with everything written on it.
#[derive(Debug, Default, Clone)]
struct Letter {
    base: char,
    vowel: Option<char>,
    dagesh: bool,
    sin_dot: bool,
    meteg: bool,
    geresh: bool,
}

/// A piece of the IPA output. Vowels are kept apart so stress can be placed
/// in front of the right one once the whole word is known.
#[derive(Debug, Clone, PartialEq)]
enum Seg {
    Consonant(&'static str),
    Vowel(char),
    Literal(char),
}

/// Phonemize a run of text that contains nikud. Whitespace-separated tokens
/// are handled independently; tokens without Hebrew letters pass through
/// untouched, as does any punctuation stuck to a word.
pub fn phonemize_vocalized(text: &str) -> Result<String> {
    let words: Vec<String> = text.split_whitespace().map(word_to_ipa).collect();
    Ok(words.join(" "))
}

/// True when the token has at least one Hebrew letter carrying a point, so it
/// should be read from its marks rather than inferred.
pub fn is_vocalized(token: &str) -> bool {
    contains_nikud(token) && token.chars().any(is_hebrew_letter)
}

/// Phonemize one token. Only Hebrew letters and their marks are interpreted;
/// everything else is copied through in place.
pub fn word_to_ipa(token: &str) -> String {
    if !token.chars().any(is_hebrew_letter) {
        return token.to_owned();
    }
    let mut letters = parse(token);
    vocalize_bare_prefix(&mut letters);
    let count = letters.iter().filter(|l| is_hebrew_letter(l.base)).count();
    let mut segs: Vec<Seg> = Vec::with_capacity(letters.len() * 2);
    let mut prev_vowel: Option<char> = None;
    let mut index = 0usize;

    for (i, letter) in letters.iter().enumerate() {
        if !is_hebrew_letter(letter.base) {
            segs.push(Seg::Literal(letter.base));
            continue;
        }
        let first = index == 0;
        let last = index + 1 == count;
        index += 1;
        let next_has_shva = letters.get(i + 1).is_some_and(|n| n.vowel == Some(SHVA));

        // Vowel letters: ו and י carrying no vowel of their own act as the
        // vowel of the previous letter rather than as consonants.
        if letter.base == 'ו' && !letter.geresh {
            if letter.dagesh && letter.vowel.is_none() {
                segs.push(Seg::Vowel('u'));
                prev_vowel = Some('u');
                continue;
            }
            if matches!(letter.vowel, Some(HOLAM) | Some(HOLAM_HASER)) && !letter.dagesh && !first {
                segs.push(Seg::Vowel('o'));
                prev_vowel = Some('o');
                continue;
            }
        }
        if letter.base == 'י' && letter.vowel.is_none() && !first {
            if matches!(prev_vowel, Some('i') | Some('e')) {
                // הִיא, אֵין: the yod lengthens the vowel and is not spoken.
                continue;
            }
            // דַּי, גּוֹי: a glide closes the syllable.
            segs.push(Seg::Consonant("j"));
            continue;
        }

        // Furtive patach: a patach under a final ח, ע or הּ is spoken before
        // the consonant, so רוּחַ is ʁuaχ, not ʁuχa.
        if last
            && letter.vowel == Some(PATACH)
            && matches!(letter.base, 'ח' | 'ע' | 'ה')
            && prev_vowel.is_some()
        {
            segs.push(Seg::Vowel('a'));
            if let Some(c) = consonant(letter, first, last) {
                segs.push(Seg::Consonant(c));
            }
            prev_vowel = Some('a');
            continue;
        }

        if let Some(c) = consonant(letter, first, last) {
            segs.push(Seg::Consonant(c));
        }

        match letter.vowel {
            Some(SHVA) => {
                if shva_is_spoken(letter, first, last, next_has_shva, prev_vowel) {
                    segs.push(Seg::Vowel('e'));
                    prev_vowel = Some('e');
                } else {
                    prev_vowel = None;
                }
            }
            Some(v) => {
                let ipa = vowel(v);
                segs.push(Seg::Vowel(ipa));
                prev_vowel = Some(ipa);
            }
            None => prev_vowel = None,
        }
    }

    let stress = stressed_vowel(&letters, &segs);
    render(&segs, stress)
}

/// A writer correcting one word usually marks only the part that was misread
/// and leaves a prefix bare: הסַפָּר rather than הַסַּפָּר. A bare first letter
/// in a word that is otherwise vocalized is read as the prefix it almost
/// always is, with the vowel that prefix carries.
fn vocalize_bare_prefix(letters: &mut [Letter]) {
    let Some(first) = letters.iter().position(|l| is_hebrew_letter(l.base)) else {
        return;
    };
    let rest_is_vocalized = letters[first + 1..]
        .iter()
        .any(|l| is_hebrew_letter(l.base) && l.vowel.is_some());
    let head = letters[first].clone();
    if head.vowel.is_some() || head.dagesh || !rest_is_vocalized {
        return;
    }
    // The vowel the prefix is actually spoken with, not the shva it is
    // written with, so the shva rules have nothing to guess.
    let default = match head.base {
        'ה' => Some(PATACH),
        'ו' | 'ב' | 'כ' | 'ל' | 'ש' => Some(SEGOL),
        'מ' => Some(HIRIQ),
        _ => None,
    };
    if let Some(v) = default {
        letters[first].vowel = Some(v);
        // A word-initial ב or כ is hard.
        if matches!(head.base, 'ב' | 'כ') {
            letters[first].dagesh = true;
        }
    }
}

fn parse(token: &str) -> Vec<Letter> {
    let mut out: Vec<Letter> = Vec::new();
    for c in token.chars() {
        match c {
            DAGESH => attach(&mut out, |l| l.dagesh = true),
            SHIN_DOT => {}
            SIN_DOT => attach(&mut out, |l| l.sin_dot = true),
            METEG => attach(&mut out, |l| l.meteg = true),
            RAFE => {}
            GERESH | '\'' | '\u{2019}' => attach(&mut out, |l| l.geresh = true),
            c if is_vowel_point(c) => attach(&mut out, |l| l.vowel = Some(c)),
            // Cantillation and anything else in the block is ignored.
            c if ('\u{0591}'..='\u{05C7}').contains(&c) => {}
            c => out.push(Letter { base: c, ..Letter::default() }),
        }
    }
    out
}

fn attach(out: &mut [Letter], apply: impl FnOnce(&mut Letter)) {
    if let Some(last) = out.last_mut() {
        apply(last);
    }
}

fn is_vowel_point(c: char) -> bool {
    matches!(
        c,
        SHVA | HATAF_SEGOL
            | HATAF_PATACH
            | HATAF_KAMATZ
            | HIRIQ
            | TSERE
            | SEGOL
            | PATACH
            | KAMATZ
            | HOLAM
            | HOLAM_HASER
            | KUBUTZ
            | KAMATZ_KATAN
    )
}

fn vowel(point: char) -> char {
    match point {
        PATACH | KAMATZ | HATAF_PATACH => 'a',
        TSERE | SEGOL | HATAF_SEGOL => 'e',
        HIRIQ => 'i',
        HOLAM | HOLAM_HASER | KAMATZ_KATAN | HATAF_KAMATZ => 'o',
        KUBUTZ => 'u',
        _ => 'e',
    }
}

fn consonant(letter: &Letter, first: bool, last: bool) -> Option<&'static str> {
    let voiced = letter.vowel.is_some();
    Some(match letter.base {
        // A glottal stop is written only where it is articulated: when the
        // letter carries a vowel. A bare final א (לֹא) is silent.
        'א' | 'ע' => {
            if voiced || (first && !last) {
                "ʔ"
            } else {
                return None;
            }
        }
        'ב' => {
            if letter.dagesh {
                "b"
            } else {
                "v"
            }
        }
        'ג' => {
            if letter.geresh {
                "dʒ"
            } else {
                "ɡ"
            }
        }
        'ד' => "d",
        // Final ה is a vowel letter unless a mappiq (dagesh) says otherwise.
        'ה' => {
            if last && !letter.dagesh && !voiced {
                return None;
            }
            "h"
        }
        'ו' => "v",
        'ז' => {
            if letter.geresh {
                "ʒ"
            } else {
                "z"
            }
        }
        'ח' => "χ",
        'ט' => "t",
        'י' => "j",
        'כ' | 'ך' => {
            if letter.dagesh {
                "k"
            } else {
                "χ"
            }
        }
        'ל' => "l",
        'מ' | 'ם' => "m",
        'נ' | 'ן' => "n",
        'ס' => "s",
        'פ' | 'ף' => {
            if letter.dagesh {
                "p"
            } else {
                "f"
            }
        }
        'צ' | 'ץ' => {
            if letter.geresh {
                "tʃ"
            } else {
                "ts"
            }
        }
        'ק' => "k",
        'ר' => "ʁ",
        'ש' => {
            if letter.sin_dot {
                "s"
            } else {
                "ʃ"
            }
        }
        'ת' => "t",
        _ => return None,
    })
}

/// Whether a shva is spoken. Israeli Hebrew silences most of them; these are
/// the cases where a vowel is heard.
fn shva_is_spoken(
    letter: &Letter,
    first: bool,
    last: bool,
    next_has_shva: bool,
    prev_vowel: Option<char>,
) -> bool {
    if last {
        return false;
    }
    // A shva under yod is spoken wherever it falls: הַיְלָדִים is "hayeladim".
    if letter.base == 'י' {
        return true;
    }
    if first {
        // Word-initial clusters are fine after most letters (שְׁנַיִם is
        // "shnayim") but not after these, where the vowel is always heard:
        // the לְ, מְ, נְ, רְ, וְ prefixes and the like.
        return matches!(letter.base, 'ל' | 'מ' | 'נ' | 'ר' | 'ו') || next_has_shva;
    }
    // Two shvas in a row: the second one is spoken (יִשְׁמְרוּ is "yishmeru").
    prev_vowel.is_none()
}

/// Index, into the sequence of spoken vowels, of the stressed one.
fn stressed_vowel(letters: &[Letter], segs: &[Seg]) -> Option<usize> {
    let vowel_positions: Vec<usize> = segs
        .iter()
        .enumerate()
        .filter_map(|(i, s)| matches!(s, Seg::Vowel(_)).then_some(i))
        .collect();
    let n = vowel_positions.len();
    if n == 0 {
        return None;
    }
    let vowel_at = |k: usize| match segs[vowel_positions[k]] {
        Seg::Vowel(v) => v,
        _ => unreachable!(),
    };

    // An explicit meteg wins. Find which spoken vowel it sits on by counting
    // the vowels of the letters before it.
    let mut spoken_before = 0usize;
    for l in letters.iter().filter(|l| is_hebrew_letter(l.base)) {
        if l.meteg && l.vowel.is_some() {
            return Some(spoken_before.min(n - 1));
        }
        spoken_before += match l.vowel {
            None => usize::from(l.base == 'ו' && l.dagesh),
            Some(SHVA) => 0,
            Some(_) => 1,
        };
    }

    if n >= 2 {
        let last = vowel_at(n - 1);
        let penult = vowel_at(n - 2);
        let final_letter = letters.iter().rev().find(|l| is_hebrew_letter(l.base));
        let furtive = final_letter
            .is_some_and(|l| l.vowel == Some(PATACH) && matches!(l.base, 'ח' | 'ע' | 'ה'));
        // Segolates (סֵפֶר, מֶלֶךְ, נַעַר) and furtive patach (רוּחַ, שָׁבוּעַ)
        // stress the penultimate vowel.
        if furtive || (last == 'e' && matches!(penult, 'e' | 'a' | 'o')) {
            return Some(n - 2);
        }
        // Segolates whose second vowel opened to a patach before a final
        // guttural: פֶּרַח, זֶרַע, נֶצַח. Still penultimate.
        let final_guttural = final_letter.is_some_and(|l| matches!(l.base, 'ח' | 'ע' | 'ה'));
        if penult == 'e' && last == 'a' && final_guttural {
            return Some(n - 2);
        }
        // The בַּיִת pattern: a, then a glide, then i, closed by one consonant.
        let between = &segs[vowel_positions[n - 2] + 1..vowel_positions[n - 1]];
        if penult == 'a' && last == 'i' && between == [Seg::Consonant("j")] {
            return Some(n - 2);
        }
    }
    Some(n - 1)
}

fn render(segs: &[Seg], stress: Option<usize>) -> String {
    let mut out = String::new();
    let mut vowel_index = 0usize;
    for seg in segs {
        match seg {
            Seg::Consonant(c) => out.push_str(c),
            Seg::Vowel(v) => {
                if stress == Some(vowel_index) {
                    out.push(STRESS);
                }
                out.push(*v);
                vowel_index += 1;
            }
            Seg::Literal(c) => out.push(*c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ambiguous_word_from_the_report() {
        // Same letters; the writer decides which word it is.
        assert_eq!(word_to_ipa("סֵפֶר"), "sˈefeʁ"); // book, segolate
        assert_eq!(word_to_ipa("סַפָּר"), "sapˈaʁ"); // barber
        assert_eq!(word_to_ipa("סִפֵּר"), "sipˈeʁ"); // told
    }

    #[test]
    fn begadkefat_follow_the_dagesh() {
        assert_eq!(word_to_ipa("בַּיִת"), "bˈajit");
        assert_eq!(word_to_ipa("אָבִיב"), "ʔavˈiv");
        assert_eq!(word_to_ipa("כֶּלֶב"), "kˈelev");
        assert_eq!(word_to_ipa("פֶּרַח"), "pˈeʁaχ");
    }

    #[test]
    fn vowel_letters_are_vowels() {
        assert_eq!(word_to_ipa("שָׁלוֹם"), "ʃalˈom");
        assert_eq!(word_to_ipa("רוּחַ"), "ʁˈuaχ");
        assert_eq!(word_to_ipa("הִיא"), "hˈi");
        assert_eq!(word_to_ipa("אֵין"), "ʔˈen");
        assert_eq!(word_to_ipa("דַּי"), "dˈaj");
    }

    #[test]
    fn final_he_and_alef_are_silent() {
        assert_eq!(word_to_ipa("תּוֹדָה"), "todˈa");
        assert_eq!(word_to_ipa("לֹא"), "lˈo");
        assert_eq!(word_to_ipa("גָּבוֹהַּ"), "ɡavˈoah");
    }

    #[test]
    fn shin_sin_and_geresh() {
        assert_eq!(word_to_ipa("שִׂמְחָה"), "simχˈa");
        assert_eq!(word_to_ipa("ג׳ִינְס"), "dʒˈins");
        assert_eq!(word_to_ipa("צ׳ִיפְּס"), "tʃˈips");
    }

    #[test]
    fn shva_rules() {
        assert_eq!(word_to_ipa("שְׁנַיִם"), "ʃnˈajim"); // initial cluster
        assert_eq!(word_to_ipa("לְאַט"), "leʔˈat"); // spoken after ל
        assert_eq!(word_to_ipa("יִשְׁמְרוּ"), "jiʃmeʁˈu"); // second of two
        assert_eq!(word_to_ipa("מֶלֶךְ"), "mˈeleχ"); // final shva silent
    }

    #[test]
    fn meteg_overrides_stress() {
        // Ultimate by default; the meteg moves it.
        assert_eq!(word_to_ipa("בָּנוּ"), "banˈu");
        assert_eq!(word_to_ipa("בָּֽנוּ"), "bˈanu");
    }

    #[test]
    fn a_bare_prefix_on_a_vocalized_word_gets_its_vowel() {
        assert_eq!(word_to_ipa("הסַפָּר"), "hasapˈaʁ"); // the barber
        assert_eq!(word_to_ipa("וסֵפֶר"), "vesˈefeʁ"); // and a book
        assert_eq!(word_to_ipa("בבַּיִת"), "bebˈajit"); // in a house
        // A fully bare word is not touched here; Renikud reads it.
        assert_eq!(word_to_ipa("הספר"), "hsfʁ");
    }

    #[test]
    fn kamatz_katan_reads_as_o() {
        assert_eq!(word_to_ipa("כָּל"), "kˈal");
        assert_eq!(word_to_ipa("כׇּל"), "kˈol");
    }

    #[test]
    fn punctuation_and_plain_tokens_pass_through() {
        assert_eq!(word_to_ipa("שָׁלוֹם,"), "ʃalˈom,");
        assert_eq!(word_to_ipa("2011"), "2011");
        assert_eq!(phonemize_vocalized("שָׁלוֹם  עוֹלָם").unwrap(), "ʃalˈom ʔolˈam");
    }

    #[test]
    fn the_poem_from_issue_7() {
        let ipa = phonemize_vocalized("כְּשֶׁכׇּל הַיְלָדִים בָּנוּ אַרְמוֹנוֹת בַּחוֹל").unwrap();
        assert_eq!(ipa, "kʃekˈol hajeladˈim banˈu ʔaʁmonˈot baχˈol");
    }
}
