//! FindFirstFileExA through the VFS hooks, without loading a title or initializing graphics.

use std::collections::BTreeSet;
use std::ffi::{c_void, CStr};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use windows_sys::Win32::Foundation::{GetLastError, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{FindClose, FindFirstFileExA, FindNextFileA, WIN32_FIND_DATAA};

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Names listed by FindFirstFileExA (`level`: FindExInfoStandard 0 / FindExInfoBasic 1) and
/// FindNextFileA, without "." and "..".
unsafe fn list_ansi(pattern: &CStr, level: i32) -> BTreeSet<String> {
    let mut d: WIN32_FIND_DATAA = std::mem::zeroed();
    let h = FindFirstFileExA(pattern.as_ptr().cast(), level, &mut d as *mut _ as *mut c_void, 0, std::ptr::null(), 0);
    assert_ne!(h, INVALID_HANDLE_VALUE, "FindFirstFileExA({pattern:?}, {level}) failed: error {}", GetLastError());
    let mut names = BTreeSet::new();
    loop {
        let name = CStr::from_ptr(d.cFileName.as_ptr().cast()).to_string_lossy().into_owned();
        if name != "." && name != ".." {
            names.insert(name);
        }
        if FindNextFileA(h, &mut d) == 0 {
            break;
        }
    }
    assert_ne!(FindClose(h), 0);
    names
}

#[test]
fn find_first_file_ex_a_lists_the_merged_game_folder() {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("xcl_vfs_find_ansi_{}_{nonce}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let root = TempDir(dir);
    let game = root.0.join("game");
    let data = root.0.join("xcldata");
    let overlay = data.join("overlay");
    fs::create_dir_all(game.join("data")).unwrap();
    fs::create_dir_all(overlay.join("data")).unwrap();
    fs::write(game.join("data").join("original.txt"), b"original").unwrap();
    fs::write(game.join("data").join("deleted.txt"), b"deleted").unwrap();
    fs::write(overlay.join("data").join("added.txt"), b"added").unwrap();
    fs::write(overlay.join(".xcl_whiteouts"), "data\\deleted.txt\n").unwrap();

    let mounts = vec![("G:".to_string(), game.to_string_lossy().into_owned())];
    xcl::vfs::init(xcl::vfs::Vfs::new(&game, &data, mounts));
    xcl::vfs_hooks::install().unwrap();

    let expected: BTreeSet<String> = ["added.txt", "original.txt"].into_iter().map(String::from).collect();
    unsafe {
        assert_eq!(list_ansi(c"G:\\data\\*", 0), expected);
        assert_eq!(list_ansi(c"G:\\data\\*", 1), expected);
    }
}
