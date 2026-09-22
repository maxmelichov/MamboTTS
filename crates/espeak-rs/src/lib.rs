use std::env;
use std::ffi::{c_char, c_void, CStr, CString};
use std::mem;
use std::path::PathBuf;
use std::ptr;
use std::sync::{Mutex, OnceLock};

const PIPER_ESPEAKNG_DATA_DIRECTORY: &str = "PIPER_ESPEAKNG_DATA_DIRECTORY";
const ESPEAKNG_DATA_DIR_NAME: &str = "espeak-ng-data";

#[derive(Debug, Clone)]
pub struct ESpeakError(pub String);

impl std::error::Error for ESpeakError {}

impl std::fmt::Display for ESpeakError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "eSpeak-ng error: {}", self.0)
    }
}

pub type ESpeakResult<T> = Result<T, ESpeakError>;

static ESPEAK_INIT: OnceLock<ESpeakResult<()>> = OnceLock::new();

/// espeak-ng keeps the active voice and the clause reader in global state, so
/// only one thread may select a voice and phonemize at a time.
static ESPEAK_LOCK: Mutex<()> = Mutex::new(());

fn init_espeak() -> ESpeakResult<()> {
    let data_dir = locate_espeak_data();
    // Keep the CString alive until after Initialize returns.
    let path_cstr = data_dir
        .as_ref()
        .and_then(|p| CString::new(p.to_string_lossy().as_ref()).ok());
    let path_ptr = path_cstr.as_ref().map_or(ptr::null(), |c| c.as_ptr());

    let sample_rate = unsafe {
        espeak_rs_sys::espeak_Initialize(
            espeak_rs_sys::espeak_AUDIO_OUTPUT_AUDIO_OUTPUT_RETRIEVAL,
            0,
            path_ptr,
            espeak_rs_sys::espeakINITIALIZE_DONT_EXIT as i32,
        )
    };

    if sample_rate <= 0 {
        Err(ESpeakError(format!(
            "Failed to initialize eSpeak-ng (code {sample_rate}). \
            Try setting `{PIPER_ESPEAKNG_DATA_DIRECTORY}` to the directory containing `{ESPEAKNG_DATA_DIR_NAME}`."
        )))
    } else {
        Ok(())
    }
}

fn locate_espeak_data() -> Option<PathBuf> {
    // 1. Environment variable
    if let Ok(dir) = env::var(PIPER_ESPEAKNG_DATA_DIRECTORY) {
        let p = PathBuf::from(dir);
        if p.join(ESPEAKNG_DATA_DIR_NAME).exists() {
            return Some(p);
        }
        if p.file_name()
            .is_some_and(|name| name == ESPEAKNG_DATA_DIR_NAME)
            && p.exists()
        {
            return Some(p);
        }
    }
    // 2. Current working directory
    if let Ok(cwd) = env::current_dir() {
        if cwd.join(ESPEAKNG_DATA_DIR_NAME).exists() {
            return Some(cwd);
        }
    }
    // 3. Directory of the current executable
    if let Ok(exe) = env::current_exe() {
        if let Some(dir) = exe.parent() {
            if dir.join(ESPEAKNG_DATA_DIR_NAME).exists() {
                return Some(dir.to_path_buf());
            }
        }
    }
    None
}

/// Strip inline language-switch markers of the form `(xx)` that espeak inserts
/// when the text contains words from a different language than the active voice.
fn strip_lang_switches(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth: usize = 0;
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// Punctuation that ends an espeak clause and is kept in the phoneme output.
fn is_clause_terminator(c: char) -> bool {
    matches!(c, '.' | ',' | '?' | '!' | ';' | ':' | '…')
}

/// Closing quotes and brackets that may follow a clause's terminator in the
/// source text (`"Stop!"`, `(really?)`) without being part of it.
fn is_closer(c: char) -> bool {
    matches!(c, '"' | '\'' | ')' | ']' | '}' | '»' | '”' | '’')
}

/// How far past a clause's punctuation espeak reads before it stops. It looks
/// at what follows the terminator to tell `Mr.` from the end of a sentence, so
/// the source text it reports as consumed runs a character or two into the
/// next clause.
const CLAUSE_LOOKAHEAD: usize = 2;

/// The punctuation that frames one clause, read back from the source text
/// espeak consumed for it: an inverted Spanish `¿` or `¡` that opens it, and
/// the run of terminators (`,`, `.`, `?!`, …) that closes it. The third field
/// is espeak's read-ahead, the source text that belongs to the next clause.
///
/// espeak splits clauses at this punctuation but leaves it out of the phonemes
/// it returns, so it has to be recovered from the source text.
fn clause_punctuation(source: &str) -> (Option<char>, &str, &str) {
    let opener = source
        .trim_start()
        .chars()
        .next()
        .filter(|c| matches!(c, '¿' | '¡'));

    // Walk back from the end of the consumed text, over the read-ahead, to the
    // clause's own punctuation.
    let mut head = source;
    let mut skipped = 0;
    loop {
        let body = head.trim_end().trim_end_matches(is_closer).trim_end();
        let run: usize = body
            .chars()
            .rev()
            .take_while(|&c| is_clause_terminator(c))
            .map(char::len_utf8)
            .sum();
        if run > 0 {
            return (opener, &body[body.len() - run..], &source[body.len()..]);
        }
        let Some(c) = head.chars().next_back() else {
            return (opener, "", "");
        };
        if !c.is_whitespace() {
            skipped += 1;
            if skipped > CLAUSE_LOOKAHEAD {
                return (opener, "", "");
            }
        }
        head = &head[..head.len() - c.len_utf8()];
    }
}

/// Convert `text` to IPA phonemes using the given espeak-ng voice/language.
///
/// `espeak_TextToPhonemes` returns one clause at a time, advancing a pointer
/// through the input, and the phonemes it returns carry neither the clause's
/// punctuation nor the word break between clauses. This function reads the
/// punctuation back from the source text espeak consumed for each clause,
/// appends it to the clause's phonemes, and joins clauses with a space, so
/// `Hello, world. Goodbye!` becomes `həlˈoʊ, wˈɜːld.` and `ɡʊdbˈaɪ!`.
///
/// One `String` is returned per sentence: a clause ending in `.`, `?`, `!` or
/// `…` closes a sentence, and so does the end of each input line.
///
/// Inline language-switch markers (`(en)`, `(ar)`, …) are always stripped.
pub fn text_to_phonemes(
    text: &str,
    language: &str,
    phoneme_separator: Option<char>,
) -> ESpeakResult<Vec<String>> {
    // Ensure the library is initialised exactly once.
    ESPEAK_INIT
        .get_or_init(init_espeak)
        .as_ref()
        .map_err(|e| e.clone())?;
    // The guarded state is espeak's, not the `()`, so a panic elsewhere while
    // holding the lock leaves nothing here to repair.
    let _espeak = ESPEAK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let lang_cstr = CString::new(language)
        .map_err(|_| ESpeakError("Language name contains a null byte".into()))?;
    let set_voice = unsafe { espeak_rs_sys::espeak_SetVoiceByName(lang_cstr.as_ptr()) };
    if set_voice != espeak_rs_sys::espeak_ERROR_EE_OK {
        return Err(ESpeakError(format!("Failed to set voice: `{language}`")));
    }

    let phoneme_mode = match phoneme_separator {
        Some(c) => ((c as u32) << 8) | espeak_rs_sys::espeakINITIALIZE_PHONEME_IPA,
        None => espeak_rs_sys::espeakINITIALIZE_PHONEME_IPA,
    } as i32;

    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        let text_cstr =
            CString::new(line).map_err(|_| ESpeakError("Text contains a null byte".into()))?;

        // espeak advances this pointer clause by clause, setting it to null when
        // done. For UTF-8 input it points into `text_cstr`, just past the source
        // text of the clause it returned.
        let base = text_cstr.as_ptr();
        let mut text_ptr: *const c_char = base;
        let mut consumed = 0;
        // Source text espeak read ahead of the clause it just returned.
        let mut read_ahead = String::new();

        while !text_ptr.is_null() {
            let phonemes = unsafe {
                let res = espeak_rs_sys::espeak_TextToPhonemes(
                    &mut text_ptr as *mut *const c_char as *mut *const c_void,
                    espeak_rs_sys::espeakCHARS_UTF8 as i32,
                    phoneme_mode,
                );
                if res.is_null() {
                    // espeak could not decode the text and did not advance.
                    return Err(ESpeakError(format!("Failed to phonemize `{line}`")));
                }
                CStr::from_ptr(res).to_string_lossy().into_owned()
            };

            let end = if text_ptr.is_null() {
                line.len()
            } else {
                (text_ptr as usize)
                    .saturating_sub(base as usize)
                    .clamp(consumed, line.len())
            };
            let source = format!(
                "{read_ahead}{}",
                line.get(consumed..end).unwrap_or_default()
            );
            consumed = end;

            let (opener, terminator, rest) = clause_punctuation(&source);
            let terminator = terminator.to_owned();
            read_ahead = rest.to_owned();
            let phonemes = strip_lang_switches(&phonemes);
            let phonemes = phonemes.trim();
            if phonemes.is_empty() {
                continue;
            }

            if !current.is_empty() {
                current.push(' ');
            }
            current.extend(opener);
            current.push_str(phonemes);
            current.push_str(&terminator);

            if terminator.ends_with(['.', '?', '!', '…']) {
                sentences.push(mem::take(&mut current));
            }
        }

        // Flush any trailing content that didn't end with sentence punctuation.
        if !current.is_empty() {
            sentences.push(mem::take(&mut current));
        }
    }

    Ok(sentences)
}

// ==============================

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT_ALICE: &str =
        "Who are you? said the Caterpillar. Replied Alice , rather shyly, I hardly know, sir!";

    #[test]
    fn test_basic_en() -> ESpeakResult<()> {
        // No punctuation in the text, so none is invented.
        let phonemes = text_to_phonemes("test", "en-US", None)?.join("");
        assert_eq!(phonemes, "tˈɛst");
        let phonemes = text_to_phonemes("test.", "en-US", None)?.join("");
        assert_eq!(phonemes, "tˈɛst.");
        Ok(())
    }

    #[test]
    fn clause_punctuation_reads_terminators_and_openers() {
        assert_eq!(clause_punctuation("Hello, "), (None, ",", " "));
        assert_eq!(clause_punctuation("world."), (None, ".", ""));
        assert_eq!(clause_punctuation(" Goodbye!  "), (None, "!", "  "));
        assert_eq!(clause_punctuation("really?!"), (None, "?!", ""));
        assert_eq!(clause_punctuation("\"Stop!\" "), (None, "!", "\" "));
        assert_eq!(clause_punctuation("(really?) "), (None, "?", ") "));
        assert_eq!(clause_punctuation(" ¿Cómo estás? "), (Some('¿'), "?", " "));
        assert_eq!(clause_punctuation("¡Adiós!"), (Some('¡'), "!", ""));
        assert_eq!(clause_punctuation("wait…"), (None, "…", ""));
        assert_eq!(clause_punctuation("no punctuation"), (None, "", ""));
        assert_eq!(clause_punctuation(""), (None, "", ""));
    }

    #[test]
    fn clause_punctuation_looks_past_espeaks_read_ahead() {
        // espeak stops a character or two into the next clause, so the text it
        // reports as consumed for `Hello,` is `Hello, w`.
        assert_eq!(clause_punctuation("Hello, w"), (None, ",", " w"));
        assert_eq!(clause_punctuation("orld. G"), (None, ".", " G"));
        assert_eq!(clause_punctuation("Hi! ¿Q"), (None, "!", " ¿Q"));
        // Far enough past the punctuation, it is the previous clause's.
        assert_eq!(clause_punctuation("Hello, world"), (None, "", ""));
    }

    #[test]
    fn keeps_word_breaks_and_punctuation_between_clauses() -> ESpeakResult<()> {
        // Issue #12: clauses used to be glued together with no space and no
        // punctuation, and the whole text came back as a single sentence.
        let sentences = text_to_phonemes("Hello, world. Goodbye!", "en-us", None)?;
        assert_eq!(sentences, ["həlˈoʊ, wˈɜːld.", "ɡʊdbˈaɪ!"]);
        Ok(())
    }

    #[test]
    fn keeps_word_breaks_and_punctuation_in_every_model_language() -> ESpeakResult<()> {
        for (voice, text) in [
            (
                "en-us",
                "Hello, world. How are you today? I am fine, thank you!",
            ),
            ("es", "Hola, mundo. ¿Cómo estás hoy? Estoy bien, gracias!"),
            (
                "de",
                "Hallo, Welt. Wie geht es dir heute? Mir geht es gut, danke!",
            ),
            ("it", "Ciao, mondo. Come stai oggi? Sto bene, grazie!"),
        ] {
            let sentences = text_to_phonemes(text, voice, None)?;
            assert_eq!(sentences.len(), 3, "{voice}: {sentences:?}");
            let joined = sentences.join(" ");
            // One word break per source word, give or take the odd pair of
            // words espeak runs together ("I am" -> "aɪɐm").
            let words = |s: &str| s.split_whitespace().count();
            assert!(words(&joined) + 1 >= words(text), "{voice}: {joined}");
            let punctuation: String = joined.chars().filter(|c| ",.?!".contains(*c)).collect();
            assert_eq!(punctuation, ",.?,!", "{voice}: {joined}");
        }
        Ok(())
    }

    #[test]
    fn keeps_spanish_inverted_marks() -> ESpeakResult<()> {
        let joined = text_to_phonemes("¡Hola! ¿Qué tal?", "es", None)?.join(" ");
        assert!(joined.starts_with('¡'), "{joined}");
        assert!(joined.contains("! ¿"), "{joined}");
        assert!(joined.ends_with('?'), "{joined}");
        Ok(())
    }

    #[test]
    fn does_not_break_inside_numbers_or_after_closing_quotes() -> ESpeakResult<()> {
        let sentences = text_to_phonemes("It costs 3.50 dollars.", "en-us", None)?;
        assert_eq!(sentences.len(), 1, "{sentences:?}");
        assert_eq!(sentences[0].matches('.').count(), 1, "{sentences:?}");

        let sentences = text_to_phonemes("He said \"stop!\" Then he left.", "en-us", None)?;
        assert_eq!(sentences.len(), 2, "{sentences:?}");
        assert!(sentences[0].ends_with('!'), "{sentences:?}");
        Ok(())
    }

    #[test]
    fn test_it_splits_sentences() -> ESpeakResult<()> {
        let phonemes = text_to_phonemes(TEXT_ALICE, "en-US", None)?;
        assert_eq!(phonemes.len(), 3);
        Ok(())
    }

    #[test]
    fn test_it_adds_phoneme_separator() -> ESpeakResult<()> {
        let phonemes = text_to_phonemes("test", "en-US", Some('_'))?.join("");
        assert_eq!(phonemes, "t_ˈɛ_s_t");
        Ok(())
    }

    #[test]
    fn test_it_preserves_clause_breakers() -> ESpeakResult<()> {
        let phonemes = text_to_phonemes(TEXT_ALICE, "en-US", None)?.join("");
        for c in ['.', ',', '?', '!'] {
            assert!(phonemes.contains(c), "Clause breaker `{c}` not preserved");
        }
        Ok(())
    }

    #[test]
    fn test_arabic() -> ESpeakResult<()> {
        let phonemes = text_to_phonemes("مَرْحَبَاً بِكَ أَيُّهَا الْرَّجُلْ", "ar", None)?.join("");
        assert_eq!(phonemes, "mˈarħabˌaː bikˌa ʔaˈiuːhˌaː alrrdʒˈul");
        Ok(())
    }

    #[test]
    fn test_lang_switch_markers_stripped() -> ESpeakResult<()> {
        // Mixed-language text: espeak inserts (en)/(ar) markers; we always strip them.
        let phonemes = text_to_phonemes("Hello معناها مرحباً", "ar", None)?.join("");
        assert!(!phonemes.contains("(en)"));
        assert!(!phonemes.contains("(ar)"));
        Ok(())
    }

    #[test]
    fn test_line_splitting() -> ESpeakResult<()> {
        let phonemes = text_to_phonemes("Hello\nThere\nAnd\nWelcome", "en-US", None)?;
        assert_eq!(phonemes.len(), 4);
        Ok(())
    }
}
