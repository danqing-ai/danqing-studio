use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use tauri::{AppHandle, Manager, RunEvent, Url};

pub struct ApiProcess(pub Mutex<Option<Child>>);

fn sidecar_exe(app: &AppHandle) -> Result<PathBuf, String> {
    let res = app.path().resource_dir().map_err(|e| e.to_string())?;
    let inner = res.join("danqing-api");
    let name = if cfg!(target_os = "windows") {
        "danqing-api.exe"
    } else {
        "danqing-api"
    };
    let p = inner.join(name);
    if p.is_file() {
        Ok(p)
    } else {
        Err(format!(
            "Sidecar not found at {} (build with: python scripts/build_sidecar.py)",
            p.display()
        ))
    }
}

fn wait_for_health(port: u16) -> Result<(), String> {
    let url = format!("http://127.0.0.1:{port}/api/system/health");
    for _ in 0..120 {
        match ureq::get(&url).call() {
            Ok(resp) if (200..300).contains(&resp.status()) => return Ok(()),
            _ => thread::sleep(Duration::from_millis(500)),
        }
    }
    Err(format!(
        "Timed out waiting for API at {url} (build sidecar: python scripts/build_sidecar.py)"
    ))
}

fn start_sidecar(app: &AppHandle) -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    drop(listener);

    let user_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("server-data");
    std::fs::create_dir_all(&user_dir).map_err(|e| e.to_string())?;

    let exe = sidecar_exe(app)?;
    let child = Command::new(&exe)
        .env("DANQING_HTTP_HOST", "127.0.0.1")
        .env("DANQING_HTTP_PORT", port.to_string())
        .env("DANQING_USER_DATA_DIR", user_dir.as_os_str())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("spawn {}: {e}", exe.display()))?;

    wait_for_health(port)?;

    let api = app.state::<ApiProcess>();
    *api.0.lock().map_err(|_| "api process lock poisoned".to_string())? = Some(child);

    Ok(port)
}

fn navigate_main(app: &AppHandle, port: u16) -> Result<(), String> {
    let win = app
        .get_webview_window("main")
        .ok_or_else(|| "missing webview window 'main'".to_string())?;
    let target =
        Url::parse(&format!("http://127.0.0.1:{port}/")).map_err(|e| e.to_string())?;
    win.navigate(target).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn run() {
    let app = tauri::Builder::default()
        .manage(ApiProcess(Mutex::new(None)))
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                let _ = app;
                return Ok(());
            }
            #[cfg(not(debug_assertions))]
            {
                let handle = app.handle().clone();
                let port = start_sidecar(&handle)?;
                navigate_main(&handle, port)?;
                Ok(())
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        if matches!(event, RunEvent::Exit) {
            if let Some(api) = app_handle.try_state::<ApiProcess>() {
                if let Ok(mut g) = api.0.lock() {
                    if let Some(mut c) = g.take() {
                        let _ = c.kill();
                        let _ = c.wait();
                    }
                }
            }
        }
    });
}
