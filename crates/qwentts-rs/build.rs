//! Builds qwentts.cpp and generates bindings for its C ABI.
//!
//! Mirrors the `ggml-rs-sys` arrangement: the upstream sources are cloned
//! at a pinned revision into `OUT_DIR` unless an override points at a
//! checkout, and every stage can be bypassed with an environment variable
//! so packaging and CI can supply prebuilt artifacts instead.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const QWENTTS_REPO: &str = "https://github.com/ServeurpersoCom/qwentts.cpp.git";
const QWENTTS_REV: &str = "a8a7716b530e49fed537c57711247c12fbbb903c";

fn main() {
    println!("cargo:rerun-if-env-changed=QWENTTS_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=QWENTTS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=QWENTTS_INCLUDE_DIR");
    println!("cargo:rerun-if-changed=wrapper.h");

    let include_dir = env::var_os("QWENTTS_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| source_dir().join("src"));

    let build_dir = env::var_os("QWENTTS_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| build_qwentts(&source_dir()));

    // The upstream CMake project has no install rules, so the artifacts
    // are picked up from the build tree itself.
    let mut available = Vec::new();
    for dir in library_dirs(&build_dir) {
        println!("cargo:rustc-link-search=native={}", dir.display());
        available.extend(static_libraries_in(&dir));
    }

    link_static_libraries(&available);
    link_system_frameworks();
    generate_bindings(&include_dir);
}

/// Everything is linked statically so the sidecar carries no extra
/// shared objects. A dependency's build script cannot add an rpath to
/// the final binary, so a shared `libqwen` would not be found at
/// run time without extra work in every consumer.
///
/// Order runs from dependents to dependencies, which is what GNU ld
/// needs; ld64 resolves across archives regardless.
fn link_static_libraries(available: &[String]) {
    // whole-archive keeps the GGML backends in the binary: each one
    // registers itself from a static initialiser that nothing references
    // directly, so a normal link would drop them and leave the runtime
    // with no compute backend at all.
    for library in [
        "qwen-core",
        "ggml",
        "ggml-cpu",
        "ggml-blas",
        "ggml-metal",
        "ggml-vulkan",
        "ggml-base",
    ] {
        if available.iter().any(|name| name == library) {
            println!("cargo:rustc-link-lib=static:+whole-archive={library}");
        }
    }
}

/// Names of the static archives in `dir`, stripped of the `lib` prefix
/// and the `.a` suffix, as the linker expects them.
fn static_libraries_in(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let stem = name.strip_suffix(".a")?;
            Some(stem.strip_prefix("lib").unwrap_or(stem).to_owned())
        })
        .collect()
}

fn link_system_frameworks() {
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        if cfg!(feature = "metal") {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
            println!("cargo:rustc-link-lib=framework=QuartzCore");
        }
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn source_dir() -> PathBuf {
    if let Some(path) = env::var_os("QWENTTS_SOURCE_DIR").map(PathBuf::from) {
        return path;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let checkout = out_dir.join("qwentts.cpp");
    if !checkout.exists() {
        // ggml is a submodule of qwentts.cpp, so the clone has to recurse.
        // A pinned revision is not fetchable by `clone --branch`, hence
        // init + fetch of the exact commit.
        run(Command::new("git").arg("init").arg(&checkout));
        run(Command::new("git")
            .current_dir(&checkout)
            .args(["remote", "add", "origin", QWENTTS_REPO]));
        run(Command::new("git")
            .current_dir(&checkout)
            .args(["fetch", "--depth", "1", "origin", QWENTTS_REV]));
        run(Command::new("git")
            .current_dir(&checkout)
            .args(["checkout", "FETCH_HEAD"]));
        run(Command::new("git")
            .current_dir(&checkout)
            .args(["submodule", "update", "--init", "--recursive", "--depth", "1"]));
    }
    checkout
}

fn build_qwentts(source_dir: &Path) -> PathBuf {
    let mut config = cmake::Config::new(source_dir);
    config
        // qwen-core carries the same entry points as the shared library
        // and links statically, so the sidecar ships no extra objects.
        .define("QWEN_SHARED", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .build_target("qwen-core");

    if cfg!(target_os = "windows") {
        config.generator("Ninja");
    }

    config.define("GGML_METAL", if cfg!(feature = "metal") { "ON" } else { "OFF" });
    config.define("GGML_VULKAN", if cfg!(feature = "vulkan") { "ON" } else { "OFF" });
    config.define("GGML_CUDA", if cfg!(feature = "cuda") { "ON" } else { "OFF" });

    config.build().join("build")
}

/// Every directory under the build tree that holds a linkable artifact.
fn library_dirs(build_dir: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    collect_library_dirs(build_dir, &mut dirs);
    dirs.sort();
    dirs.dedup();
    dirs
}

fn collect_library_dirs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_library_dirs(&path, out);
        } else if is_library(&path) {
            out.push(dir.to_path_buf());
        }
    }
}

fn is_library(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    ["\u{2e}so", ".dylib", ".dll", ".a", ".lib"]
        .iter()
        .any(|extension| name.contains(extension))
}

fn link_dir(dir: &Path) {
    println!("cargo:rustc-link-search=native={}", dir.display());
    // The sidecar ships the shared libraries beside the binary, so the
    // loader is pointed at both the build tree and the executable's own
    // directory.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    }
}

fn link_cxx_runtime() {
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn generate_bindings(include_dir: &Path) {
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_function("qt_.*")
        .allowlist_type("qt_.*")
        .allowlist_var("QT_.*")
        // Plain consts keep `qt_status` a bare integer, so the negative
        // failure convention survives an `as i32` cast.
        .default_enum_style(bindgen::EnumVariation::Consts)
        .derive_default(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("generate qwentts bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings.write_to_file(out_path).expect("write bindings");
}

fn run(command: &mut Command) {
    let status = command
        .status()
        .unwrap_or_else(|err| panic!("failed to spawn {command:?}: {err}"));
    assert!(status.success(), "command failed: {command:?}");
}
