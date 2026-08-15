fn commit_hash() -> String {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .expect("failed to get git commit hash");
    String::from_utf8(output.stdout).expect("git commit hash was not UTF-8")
}

fn main() {
    let hash = commit_hash();
    println!("cargo:rerun-if-env-changed=COMMIT_HASH");
    println!("cargo:rustc-env=COMMIT_HASH={}", hash);
    println!("cargo:rerun-if-env-changed=APTABASE_APP_KEY");
    println!("cargo:rerun-if-env-changed=APTABASE_BASE_URL");
    load_env("APTABASE_APP_KEY");
    load_env("APTABASE_BASE_URL");
    tauri_build::build()
}

fn load_env(name: &str) {
    if let Some(value) = std::env::var(name).ok().filter(|value| !value.is_empty()) {
        println!("cargo:rustc-env={name}={value}");
        println!("cargo:warning={name} is set; Aptabase analytics will be enabled");
    }
}
