fn main() {
    // NEO Sandbox (the `sandbox` feature) is a test build: a release build of it must be the
    // sandbox's own app, never one with NEO's real bundle id, home and port. `tauri build` passes
    // the --config overlay here as TAURI_CONFIG (scripts/sandbox_build.sh passes
    // tauri.sandbox.conf.json); tests (debug) may build the feature without it.
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_SANDBOX");
    println!("cargo:rerun-if-env-changed=TAURI_CONFIG");
    let sandbox = std::env::var_os("CARGO_FEATURE_SANDBOX").is_some();
    let release = std::env::var("PROFILE").map(|p| p == "release").unwrap_or(false);
    if sandbox && release {
        let overlay = std::env::var("TAURI_CONFIG").unwrap_or_default();
        let id = serde_json::from_str::<serde_json::Value>(&overlay)
            .ok()
            .and_then(|v| v.get("identifier").and_then(|i| i.as_str()).map(String::from));
        if id.as_deref() != Some("com.arnav.neo.sandbox") {
            panic!("the sandbox feature builds only as NEO Sandbox: tauri build --features sandbox --config src-tauri/tauri.sandbox.conf.json (scripts/sandbox_build.sh)");
        }
    }
    tauri_build::build()
}
