mod analytics;
mod files;
mod model;
mod runner;
mod voices;

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let mut builder = tauri::Builder::default()
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .manage(runner::RunnerState {
            process: std::sync::Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            files::reveal_path,
            model::get_model_bundle,
            model::get_phonikud_bundle,
            model::get_model_bundle_for_runtime,
            model::get_model_sources,
            model::download_model_bundle,
            model::download_phonikud_bundle,
            runner::start_runner,
            runner::stop_runner,
            runner::get_runner_url,
            runner::load_model,
            runner::get_languages,
            runner::get_voices,
            runner::phonemize,
            runner::diacritize,
            runner::get_phoneme_inventory,
            runner::synthesize,
            runner::copy_audio_file,
            voices::download_voice,
        ]);

    if analytics::is_aptabase_configured() {
        let options = tauri_plugin_aptabase::InitOptions {
            host: Some(analytics::APTABASE_BASE_URL.to_string()),
            ..Default::default()
        };
        builder = builder.plugin(
            tauri_plugin_aptabase::Builder::new(analytics::APTABASE_APP_KEY)
                .with_options(options)
                .build(),
        );
    }

    let app = builder
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app, event| match event {
        tauri::RunEvent::Ready => {
            analytics::track_event_handle(app, analytics::events::APP_STARTED);
        }
        tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit => {
            runner::stop_managed_runner(app);
        }
        _ => {}
    });
}
