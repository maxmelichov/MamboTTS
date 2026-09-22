//! Diacritize Hebrew with RenikudPlus and show the IPA before and after.
//!
//! ```console
//! MAMBOTTS_RENIKUD_PATH=/path/to/renikud-plus.onnx cargo run -p renikud-plus-rs --example diacritize -- "שבת שלום"
//! ```

use anyhow::Context;
use renikud_plus_rs::G2P;

fn main() -> anyhow::Result<()> {
    let model = std::env::var("MAMBOTTS_RENIKUD_PATH")
        .context("set MAMBOTTS_RENIKUD_PATH to renikud-plus.onnx")?;
    let mut g2p = G2P::new(&model)?;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let texts = if args.is_empty() {
        [
            "שלום עולם, מה שלומך היום?",
            "הילדים הלכו לבית הספר בבוקר.",
            "ספר",
            "שבת שלום",
        ]
        .map(str::to_owned)
        .to_vec()
    } else {
        args
    };
    let mut mismatches = 0;
    for text in texts {
        let ipa = g2p.phonemize(&text, 0, 0)?;
        let nikud = g2p.diacritize(&text, 0, 0, true)?;
        let bare = g2p.diacritize(&text, 0, 0, false)?;
        let round_trip = g2p.phonemize(&nikud, 0, 0)?;
        let bare_round_trip = g2p.phonemize(&bare, 0, 0)?;
        println!("text:              {text}");
        println!("diacritized:       {nikud}");
        println!("without stress:    {bare}");
        println!("ipa:               {ipa}");
        println!("ipa (diacritized): {round_trip}");
        println!("ipa (no stress):   {bare_round_trip}");
        if round_trip != ipa {
            mismatches += 1;
            println!("MISMATCH");
        }
        println!();
    }
    anyhow::ensure!(mismatches == 0, "{mismatches} round trips changed the IPA");
    Ok(())
}
