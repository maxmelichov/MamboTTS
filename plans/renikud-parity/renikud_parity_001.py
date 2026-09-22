#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = ["onnxruntime==1.23.2", "numpy", "hebrew-num2words==0.1.0"]
# ///
"""Upstream-parity harness for `crates/renikud-plus-rs`.

Runs one corpus of >=300 varied Hebrew texts through the upstream Python
package (`renikud_onnx`, imported from a local clone) and through the Rust port
(`cargo run -p renikud-plus-rs --example parity`), then diffs the two outputs
character by character.

    RENIKUD_UPSTREAM=C:/Users/maxme/code/renikud-plus-upstream \
    MAMBOTTS_RENIKUD_PATH=.../renikud-plus.onnx \
      uv run plans/renikud-parity/renikud_parity_001.py

Options: --only phonemize,vocalize,numbers,align,lexicon to restrict the ops,
--limit N to cut the corpus down, --skip-rust to reuse an existing results file.
The reference implementation is used for comparison only; nothing here ships.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "out"
UPSTREAM = Path(os.environ.get("RENIKUD_UPSTREAM", r"C:\Users\maxme\code\renikud-plus-upstream"))
MODEL = os.environ.get(
    "MAMBOTTS_RENIKUD_PATH",
    str(Path.home() / "AppData/Local/com.maxmelichov.mambotts/models/bluetts-2.5/renikud-plus.onnx"),
)

sys.path.insert(0, str(UPSTREAM / "src"))


# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------

SENTENCES = [
    "שלום עולם",
    "שלום לכולם",
    "מה שלומך היום?",
    "הילדים הלכו לבית הספר בבוקר.",
    "אני אוהב לקרוא ספרים בערב, במיוחד ספרי מדע בדיוני.",
    "השמש זורחת מעל הים התיכון ומאירה את החוף.",
    "הוא אמר לה שהוא יחזור מאוחר מהעבודה.",
    "המורה הסבירה לתלמידים את הנושא החדש בשיעור.",
    "בואו נצא לטיול בצפון בסוף השבוע הקרוב.",
    "הכלב נבח על החתול שישב על הגדר.",
    "קניתי לחם, חלב, גבינה וביצים בסופרמרקט.",
    "הרכבת יצאה מהתחנה בזמן ולא איחרה אפילו דקה.",
    "היא רצה מהר מאוד ונשמה בכבדות.",
    "היא רצה לעשות את זה בעצמה.",
    "הוא סיפר לי סיפור מרתק על סבא שלו.",
    "הספר הזה מעניין מאוד וכדאי לקרוא אותו.",
    "הספר גזר את שערו של הילד.",
    "מזג האוויר היום נעים ונוח לטיול רגלי.",
    "אנחנו נפגשים כל יום שלישי בשעות אחר הצהריים.",
    "הישיבה התחילה באיחור קל בגלל הפקק בכביש.",
    "תודה רבה על העזרה שלך, אני מעריך את זה מאוד.",
    "בבקשה תשלח לי את המסמך במייל עד מחר בבוקר.",
    "אין לי מושג מה קרה שם אתמול בלילה.",
    "הבית החדש שלהם נמצא ברחוב הרצל פינת ביאליק.",
    "המסעדה הזאת מגישה אוכל ביתי טעים במחירים סבירים.",
    "הילדה שרה שיר יפה בטקס של בית הספר.",
    "המנהל הודיע על שינויים בנוהל העבודה.",
    "העובדים ביקשו העלאה בשכר ותנאים טובים יותר.",
    "החתול ישן על הספה כל אחר הצהריים.",
    "שבת שלום ומבורך לכל בית ישראל.",
    "חג שמח ושנה טובה לכל המשפחה.",
    "הרופא בדק את החולה וכתב לו מרשם לתרופה.",
    "הסטודנטים נבחנו בקורס מבוא למדעי המחשב.",
    "הטיסה לניו יורק נמשכת כשתים עשרה שעות.",
    "הוא שכח את המפתחות שלו בתוך הרכב.",
    "אמא הכינה מרק ירקות חם לארוחת הערב.",
    "הצבא הודיע על גיוס מילואים באזור הצפון.",
    "הממשלה אישרה את התקציב החדש אחרי דיון ארוך.",
    "המשטרה עצרה את החשוד בגניבת הרכב.",
    "הכתבת דיווחה מהשטח על ההתרחשויות.",
    "הבורסה בתל אביב ננעלה בעליות שערים.",
    "המטבע הדיגיטלי איבד מערכו בחודש האחרון.",
    "השחקן כבש שלושה שערים במשחק אחד.",
    "הקבוצה ניצחה את היריבה בגמר האליפות.",
    "הזמר הופיע מול קהל של אלפי אנשים.",
    "הסרט החדש זכה בפרס בפסטיבל בינלאומי.",
    "הכנס יתקיים במרכז הכנסים בירושלים.",
    "התערוכה נפתחה במוזיאון תל אביב לאמנות.",
    "הגשם ירד כל הלילה והפסיק רק לפנות בוקר.",
    "השלג כיסה את פסגת החרמון בלבן.",
    "הים היה סוער והגלים היו גבוהים במיוחד.",
    "המדריך הוביל את הקבוצה בשביל ההרים.",
    "הרוח נשבה חזק והעצים התכופפו.",
    "החקלאי קטף את הפירות מהעצים בפרדס.",
    "הגנן שתל פרחים חדשים בגינה הציבורית.",
    "התינוק ישן שנת ישרים אחרי האמבטיה.",
    "הסבתא סיפרה לנכדים על הימים ההם.",
    "הדוד הגיע מחוץ לארץ לביקור קצר.",
    "השכנים ערכו מסיבה רועשת עד שעה מאוחרת.",
    "החנות סגורה היום בגלל שיפוצים.",
    "המחשב שלי נתקע ולא הצלחתי לשמור את הקובץ.",
    "הטלפון החכם שלה נפל ונשבר המסך.",
    "האינטרנט לא עבד כל הבוקר בבניין.",
    "הדואר האלקטרוני שלי מלא בהודעות פרסומת.",
    "התוכנה החדשה מאפשרת לערוך תמונות בקלות.",
    "הבינה המלאכותית משנה את עולם העבודה.",
    "המדענים גילו חיידק עמיד לאנטיביוטיקה.",
    "החוקרים פרסמו מאמר בכתב עת מדעי.",
    "הניסוי הצליח מעבר לכל הציפיות.",
    "התלמיד הצטיין בבחינת הבגרות במתמטיקה.",
    "המרצה העביר הרצאה מרתקת על ההיסטוריה של העם היהודי.",
    "הספרייה העירונית פתוחה גם בימי שישי.",
    "הילדים שיחקו בגן השעשועים עד שהחשיך.",
    "המורה לספורט ארגנה תחרות ריצה בשכבה.",
    "הנהג עצר ברמזור אדום והמתין בסבלנות.",
    "האוטובוס היה מלא בנוסעים בשעת העומס.",
    "המונית הגיעה תוך חמש דקות מההזמנה.",
    "האופניים החשמליים נעולים בחניון הבניין.",
    "הרכבת הקלה עוברת ברחוב יפו בירושלים.",
    "המטוס נחת בשלום בנמל התעופה בן גוריון.",
    "הספינה עגנה בנמל חיפה אחרי שבועיים בים.",
    "הפועלים סיימו את העבודה על הגשר החדש.",
    "העירייה הודיעה על סגירת כביש לתנועה.",
    "המהנדס תכנן את הבניין החדש בקפידה.",
    "האדריכלית הציגה את התוכנית לוועדה.",
    "הקבלן חתם על החוזה עם הדיירים.",
    "המתווך הראה לנו שלוש דירות באותו יום.",
    "הבנק אישר את המשכנתא אחרי בדיקה ארוכה.",
    "רואה החשבון הגיש את הדוח השנתי.",
    "עורך הדין ייצג את הלקוח בבית המשפט.",
    "השופט קבע את מועד הדיון הבא.",
    "העד מסר את גרסתו לחוקרים.",
    "הנאשם טען שהוא חף מפשע.",
    "הוועדה החליטה לדחות את ההצעה.",
    "הפגישה נדחתה לשבוע הבא בגלל מחלה.",
    "ההרשמה לקורס נסגרת בסוף החודש.",
    "המחירים עלו בשנה האחרונה בצורה ניכרת.",
    "המבצע בחנות נמשך עד גמר המלאי.",
    "הלקוחות התלוננו על השירות הגרוע.",
    "החברה גייסה הון מקרן השקעות אמריקאית.",
    "היזם פתח סטארט אפ בתחום הבריאות.",
]

POINTED = [
    "סֵפֶר",
    "סַפָּר",
    "סָפַר",
    "סִפֵּר",
    "שָׁלוֹם לְכֻּלָּם",
    "שָׁלוֹם עוֹלָם",
    "בְּנֹתָיו",
    "צִדְקֹתָיו",
    "קֳדָשָׁיו",
    "מִצְוֹתָיו",
    "דְּבָרָיו",
    "אֱלֹהָיו",
    "יָדָיו",
    "כָּל הַיּוֹם",
    "הַמְּנוֹרָה מאירה",
    "בֵּית הַסֵּפֶר",
    "מִשְׁפָּחָה",
    "יְרוּשָׁלַיִם",
    "תּוֹרָה",
    "אַהֲבָה",
    "חֲתוּנָה",
    "מְנַהֵל",
    "עֶרֶב טוֹב",
    "בֹּקֶר אוֹר",
    "רוּחַ",
    "תִּקְוָה",
    "גְּבִינָה",
    "כֻּלָּנוּ",
    "שֻׁלְחָן",
    "אִשָּׁה",
    "מַיִם קָרִים",
    "לֶחֶם חַם",
    "הוּא הָלַךְ",
    "הִיא אָמְרָה",
    "אֲנִי כּוֹתֵב",
    "הַכֶּלֶב נָבַח",
    "צָהֳרַיִם טוֹבִים",
    "שְׁנֵי יְלָדִים",
    "עִבְרִית קַלָּה",
    "פַּעַם אַחַת",
    "אֶן קֶלְוִין",
    "שָׂרָה וְרִבְקָה",
    "וָו הַחִבּוּר",
    "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ׃",
    "שְׁמַע יִשְׂרָאֵל יְהוָה אֱלֹהֵינוּ יְהוָה אֶחָד",
]

ACRONYMS = [
    'צה"ל הודיע על תרגיל',
    "צה״ל הודיע על תרגיל",
    'דו"ח שנתי של מבקר המדינה',
    'ארה"ב וישראל חתמו על הסכם',
    'עו"ד ייצג את הלקוח',
    'ד"ר כהן קיבל את המטופל',
    'רמטכ"ל נפגש עם שר הביטחון',
    'ת"א היא עיר גדולה',
    'ק"ג של תפוחים',
    'ש"ח לשעה',
    "לכן תשובה ב׳ נכונה",
    "סעיף ג' בחוזה",
    "אפשרות א׳ או אפשרות ב׳",
    "ג׳אז וצ׳יפס",
    "ג'ירפה בג'ונגל",
    "צ׳ק בנקאי",
    "ז׳קט חדש",
    "ג׳ון קיבל ג׳וב",
    "מנג׳ר של הקבוצה",
    "וינצ׳י ודה ג׳ורג׳יו",
]

MIXED = [
    "אני אוהב machine learning וגם ג׳אז.",
    "הוא עבר ל-GPU חדש עם 12 ליבות.",
    "שלח לי מייל ל-test@example.com בבקשה",
    "הורדתי את Windows 11 אתמול",
    "צוות ה-QA מצא באג ב-API",
    "הקובץ נשמר בתיקיית Downloads",
    "היא עובדת ב-Google בתל אביב",
    "קראתי מאמר על GPT בבלוג",
    "ה-CEO הודיע על שינוי אסטרטגי",
    "תוריד את האפליקציה מ-App Store",
    "הקוד כתוב ב-Rust ורץ מהר",
    "בדוק את ה-log לפני הדיפלוי",
    "שבת שלום, hello!\nספר",
    "Hello world שלום עולם",
    "טקסט עם emoji 🙂 וגם עברית",
]

NUMERIC = [
    "המחיר 1250 שקלים",
    "יש לי 8 שעות עבודה ביום",
    "קניתי 2 ספרים ו-3 מחברות",
    "הישיבה בשעה 14:30",
    "נפגשים ב 8:15 בבוקר",
    "רבע ל 10:45 יצאנו",
    "היום 14.3.2024 חתמנו",
    "התאריך 1.9 הוא תחילת הלימודים",
    "ב-1.9 מתחילים ללמוד",
    "בשנת 1948 קמה המדינה",
    "גיליון 2011 של העיתון",
    "הריבית עלתה ב-2.5%",
    "50% הנחה על הכל",
    "51,034% תשואה דמיונית",
    "10% אחוז מהעובדים",
    "המחיר 19.90 שקלים",
    "משקל 3.5 קילו",
    "קוד 4821 לכניסה",
    "מספר הטלפון 052-1234567",
    "חייג 1-800-123-456",
    "תעודת זהות 123456789",
    "מיקוד 7178000",
    "30,000 שקלים בחודש",
    "500000 איש הגיעו",
    "כ-150 דונם של שטח",
    "בשנים 2022 - 2025",
    "הקומה 7 בבניין",
    "פעם 3 ניסיתי",
    "ה-11 בחודש",
    "ה-3 בתור",
    "1/2 כוס סוכר",
    "3/4 מהכיתה",
    "התוצאה 3-0 לטובתנו",
    "הטמפרטורה 25 מעלות",
    "נסעתי 120 קילומטר",
    "עברו 45 דקות",
    "הוא בן 17 וחצי",
    "יש 0 סיכויים",
    "07 היא ספרה מובילה",
    "המשקל 0.5 קילו",
    "בערך 1000 איש",
    "2000 שנה של היסטוריה",
    "המלון בקומה 12",
    "חדר 305 בבית המלון",
    "הזמנה 88123 אושרה",
    "עד 24 שעות",
    "משעה 9 עד 17",
    "ב-7 בערב",
    "לפני 100 שנה",
    "המרחק 2.5 קילומטר",
]

LONG_TEXTS = [
    " ".join(SENTENCES) * 2,  # > 5000 characters, splits on sentence ends
    ("מילה " * 500).strip(),  # word-boundary splitting
    "א" * 2100,  # one hard cut past the 2046 window
    "שלום, " * 400,  # clause-boundary splitting
]

GENDERED = [
    "היא רצה",
    "היא אמרה לו שהיא תבוא",
    "הוא אמר לה שהוא יבוא",
    "את יודעת מה קרה",
    "אתה יודע מה קרה",
    "אני הלכתי לשוק",
    "רציתי לספר לך משהו",
    "תגידי לי מה שלומך",
    "תגיד לי מה שלומך",
    "המנהלת החליטה על שינוי",
]


def corpus() -> list[str]:
    texts = (
        SENTENCES + POINTED + ACRONYMS + MIXED + NUMERIC + GENDERED
        + [f"{a} {b}" for a, b in zip(SENTENCES[:40], SENTENCES[40:80])]
        + [s.rstrip(".") + "?" for s in SENTENCES[:20]]
        + [w for s in SENTENCES[:30] for w in s.split()[:1]]
    )
    seen: dict[str, None] = {}
    for text in texts:
        seen.setdefault(text, None)
    return list(seen)


def build_cases(limit: int | None, only: set[str]) -> list[dict]:
    texts = corpus()
    if limit:
        texts = texts[:limit]
    cases: list[dict] = []

    def add(**case):
        case["i"] = len(cases)
        cases.append(case)

    if "phonemize" in only:
        for text in texts:
            add(op="phonemize", text=text, niqqud="strip")
            add(op="phonemize", text=text, niqqud="use")
        for text in texts[:100]:
            add(op="phonemize", text=text, niqqud="strip", exact_map=False)
        for text in GENDERED + POINTED[:8]:
            for speaker, target in ((1, 1), (2, 2), (0, 2), (2, 0)):
                add(
                    op="phonemize",
                    text=text,
                    niqqud="use",
                    speaker=speaker,
                    target_speaker=target,
                )
        for text in texts[:40]:
            add(op="phonemize", text=text, niqqud="strip", number_norm="off")
        if not limit:
            for text in LONG_TEXTS:
                add(op="phonemize", text=text, niqqud="strip")
                add(op="vocalize", text=text, niqqud="strip")
    if "vocalize" in only:
        for text in texts:
            add(op="vocalize", text=text, niqqud="strip")
            add(op="vocalize", text=text, niqqud="use")
    if "numbers" in only:
        for text in texts + NUMERIC:
            add(op="numbers", text=text)
    if "align" in only:
        for surface, ipa in ALIGN_CASES:
            add(op="align", surface=surface, ipa=ipa)
            add(op="align", surface=surface, ipa=ipa, cells="extended")
    if "lexicon" in only:
        for text in ["שלום סבתא.", "סבתא שלי אוהבת ספר", "צה\"ל הודיע"] + texts[:20]:
            add(op="lexicon", text=text, niqqud="strip", entries=LEXICON)
        add(op="lexicon", text="", entries=LEXICON_MIXED)
    return cases


ALIGN_CASES = [
    ("סבתא", "sˈavta"),
    ("סבתא", "ɡˈavta"),
    ("סבתא", "sˈavtˈa"),
    ("רוח", "ʁˈuaχ"),
    ("שלום", "ʃalˈom"),
    ("ספר", "sˈefeʁ"),
    ("ג'אז", "dʒˈaz"),
    ("צ'יפס", "tʃˈips"),
    ("ת'", "s"),
    ("אבגד", "ʔavɡˈad"),
    ("תל-אביב", "tel avˈiv"),
    ("מים", "mˈajim"),
    ("ירושלים", "jeʁuʃalˈajim"),
    ("ירושלים", "jeʁuʃalajim"),
    ("בית", "bˈajit"),
    ("כלב", "kˈelev"),
    ("שמש", "ʃˈemeʃ"),
    ("שמש", "sˈemes"),
    ("ועד", "vaʔˈad"),
    ("אויב", "ʔojˈev"),
]

LEXICON = {
    "סבתא": "sˈavta",
    "ספר": "sˈefeʁ",
    'צה"ל': "tsˈahal",
    "שלום": "ʃalˈom",
}
LEXICON_MIXED = {
    "סבתא": "sˈavta",
    "סבתא2": "sˈavta",  # rejected: a digit has no aligner rule
    "ספר": "ɡˈefeʁ",  # rejected: does not align
    "תל-אביב": "tel avˈiv",
    "": "x",  # rejected: empty surface
}


# --------------------------------------------------------------------------
# The two implementations
# --------------------------------------------------------------------------


def run_upstream(cases: list[dict]) -> list[dict]:
    from renikud_onnx import G2P, ForceLexicon, align_word, normalize_numbers, resolve_align_cells

    started = time.time()
    g2p = G2P(MODEL)
    print(f"upstream model loaded in {time.time() - started:.1f}s", file=sys.stderr)
    results = []
    for case in cases:
        op = case.get("op", "phonemize")
        try:
            if op == "numbers":
                out = normalize_numbers(case["text"])
            elif op == "align":
                cells = resolve_align_cells(case.get("cells", ()))
                out = align_word(case["surface"], case["ipa"], cells) is not None
            elif op == "lexicon":
                cells = resolve_align_cells(case.get("cells", ()))
                lex = ForceLexicon(dict(case["entries"]), align_cells=cells, on_invalid="report")
                accepted = sorted(s for s in case["entries"] if lex.lookup(s))
                ipa = None
                if case["text"]:
                    # The upstream constructor takes the lexicon; injecting it
                    # avoids loading the 1.2 GB model a second time.
                    g2p._lexicon = lex
                    ipa = g2p.phonemize(
                        case["text"],
                        speaker=case.get("speaker", 0),
                        target_speaker=case.get("target_speaker", 0),
                        exact_map=case.get("exact_map"),
                        on_hebrew_leak="ignore",
                        number_norm=case.get("number_norm", "auto"),
                        niqqud=case.get("niqqud"),
                    )
                    g2p._lexicon = None
                out = {"accepted": accepted, "ipa": ipa}
            elif op == "vocalize":
                out = g2p.vocalize(
                    case["text"],
                    speaker=case.get("speaker", 0),
                    target_speaker=case.get("target_speaker", 0),
                    exact_map=case.get("exact_map"),
                    number_norm=case.get("number_norm", "auto"),
                    niqqud=case.get("niqqud"),
                )
            else:
                out = g2p.phonemize(
                    case["text"],
                    speaker=case.get("speaker", 0),
                    target_speaker=case.get("target_speaker", 0),
                    exact_map=case.get("exact_map"),
                    on_hebrew_leak="ignore",
                    number_norm=case.get("number_norm", "auto"),
                    niqqud=case.get("niqqud"),
                )
            results.append({"i": case["i"], "out": out})
        except Exception as exc:  # noqa: BLE001 - parity includes error parity
            results.append({"i": case["i"], "error": f"{type(exc).__name__}: {exc}"})
    return results


def run_rust(cases: list[dict], requests_path: Path, results_path: Path) -> list[dict]:
    requests_path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["MAMBOTTS_RENIKUD_PATH"] = MODEL
    # The ONNX Runtime build is gitignored, so it lives in the main checkout
    # rather than in a worktree; an already-set ORT_LIB_LOCATION wins.
    ort_dir = Path(
        env.get("ORT_LIB_LOCATION")
        or ROOT / "crates" / "blue-rs" / ".ort" / "onnxruntime-win-x64-1.23.2" / "lib"
    )
    if ort_dir.exists():
        env.setdefault("ORT_STRATEGY", "system")
        env.setdefault("ORT_LIB_LOCATION", str(ort_dir))
        env.setdefault("ORT_PREFER_DYNAMIC_LINK", "1")
        env.setdefault("RUSTFLAGS", f"-L native={ort_dir} -C link-arg=advapi32.lib")
        env["PATH"] = f"{ort_dir}{os.pathsep}{env['PATH']}"
    cmd = [
        "cargo", "run", "--release", "-p", "renikud-plus-rs", "--example", "parity",
        "--", str(requests_path), str(results_path),
    ]
    print("+ " + " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="phonemize,vocalize,numbers,align,lexicon")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-rust", action="store_true")
    parser.add_argument("--skip-upstream", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args.limit, set(args.only.split(",")))
    print(f"{len(corpus())} distinct texts, {len(cases)} cases", file=sys.stderr)

    requests_path = OUT_DIR / "requests.jsonl"
    rust_path = OUT_DIR / "rust.jsonl"
    upstream_path = OUT_DIR / "upstream.jsonl"

    if args.skip_rust:
        rust = [json.loads(line) for line in rust_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        started = time.time()
        rust = run_rust(cases, requests_path, rust_path)
        print(f"rust: {len(rust)} results in {time.time() - started:.1f}s", file=sys.stderr)

    if args.skip_upstream:
        upstream = [json.loads(line) for line in upstream_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        started = time.time()
        upstream = run_upstream(cases)
        upstream_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in upstream) + "\n", encoding="utf-8"
        )
        print(f"upstream: {len(upstream)} results in {time.time() - started:.1f}s", file=sys.stderr)

    by_index = {r["i"]: r for r in rust}
    diffs = []
    counts: dict[str, list[int]] = {}
    for case, reference in zip(cases, upstream):
        mine = by_index.get(case["i"], {"error": "missing result"})
        op = case.get("op", "phonemize")
        tally = counts.setdefault(f"{op}/{case.get('niqqud', '-')}", [0, 0])
        same = mine.get("out") == reference.get("out") and (
            ("error" in mine) == ("error" in reference)
        )
        tally[0] += 1
        if same:
            tally[1] += 1
        else:
            diffs.append(
                {
                    "case": case,
                    "upstream": reference.get("out", reference.get("error")),
                    "rust": mine.get("out", mine.get("error")),
                }
            )

    (OUT_DIR / "diffs.json").write_text(
        json.dumps(diffs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(t[0] for t in counts.values())
    same = sum(t[1] for t in counts.values())
    for key in sorted(counts):
        seen, agreed = counts[key]
        print(f"{key:24} {agreed}/{seen}")
    print(f"{'TOTAL':24} {same}/{total}  ({100 * same / max(total, 1):.2f}%)")
    for diff in diffs[:20]:
        print("\n--- case", diff["case"])
        print("upstream:", diff["upstream"])
        print("rust    :", diff["rust"])
    if diffs:
        print(f"\n{len(diffs)} differences written to {OUT_DIR / 'diffs.json'}")
    return 0 if not diffs else 1


if __name__ == "__main__":
    raise SystemExit(main())
