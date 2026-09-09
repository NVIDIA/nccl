// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::{Path, PathBuf};

const NCCL_HEADER: &str = "nccl.h";
const NCCL_DEVICE_HEADER: &str = "nccl_device.h";

fn main() {
    if let Err(error) = build() {
        panic!("nccl-sys build failed: {error}");
    }
}

fn build() -> Result<(), String> {
    for variable in [
        "NCCL_INCLUDE_DIR",
        "NCCL_SOURCE_DIR",
        "NCCL_LIB_DIR",
        "CUDA_HOME",
        "CUDA_PATH",
        "PATH",
        "LIBCLANG_PATH",
        "BINDGEN_EXTRA_CLANG_ARGS",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    println!("cargo:rerun-if-changed=wrapper.hpp");

    let manifest_dir = required_path("CARGO_MANIFEST_DIR")?;
    let repo_root = nccl_repo_root(&manifest_dir);
    let explicit_include_dir = env::var_os("NCCL_INCLUDE_DIR").is_some();
    let headers = find_nccl_headers(repo_root.as_deref())?;
    let cuda_include = find_cuda_include()?;

    for header in [&headers.nccl_header, &headers.device_header] {
        println!("cargo:rerun-if-changed={}", header.display());
    }

    let mut builder = bindgen::Builder::default()
        .header(manifest_dir.join("wrapper.hpp").display().to_string())
        .rust_target(
            bindgen::RustTarget::stable(85, 0)
                .map_err(|error| format!("invalid bindgen Rust target: {error}"))?,
        )
        .rust_edition(bindgen::RustEdition::Edition2024)
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .clang_arg(format!("-I{}", cuda_include.display()))
        .allowlist_function("^(?:nccl|NCCL).*")
        .allowlist_type("^(?:nccl|NCCL).*")
        .allowlist_var("^(?:nccl|NCCL).*")
        .blocklist_item("^pnccl.*")
        // These macros expand through libc's INT_MIN/NULL definitions and are
        // not emitted consistently by libclang. Stable Rust equivalents live
        // next to the generated bindings in src/lib.rs.
        .blocklist_item("^NCCL_CONFIG_UNDEF_(?:INT|PTR)$")
        .generate_comments(true)
        // bindgen 0.71.1's layout-test pass incorrectly converts NCCL's
        // complete public CUDA C++ structs into one-byte opaque types.
        .layout_tests(false)
        .prepend_enum_name(false)
        .use_core()
        .merge_extern_blocks(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    for include_dir in &headers.include_dirs {
        builder = builder.clang_arg(format!("-I{}", include_dir.display()));
    }

    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") => builder = builder.clang_arg("-DNCCL_OS_LINUX=1"),
        Ok("windows") => builder = builder.clang_arg("-DNCCL_OS_WINDOWS=1"),
        _ => {}
    }

    let bindings = builder
        .generate()
        .map_err(|error| format!("bindgen could not parse NCCL's public headers: {error}"))?;
    let out_dir = required_path("OUT_DIR")?;
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .map_err(|error| format!("could not write generated bindings: {error}"))?;

    if let Some(lib_dir) = find_nccl_lib_dir(repo_root.as_deref(), explicit_include_dir)? {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
    println!("cargo:rustc-link-lib=dylib=nccl");

    Ok(())
}

struct NcclHeaders {
    include_dirs: Vec<PathBuf>,
    nccl_header: PathBuf,
    device_header: PathBuf,
}

fn find_nccl_headers(repo_root: Option<&Path>) -> Result<NcclHeaders, String> {
    if let Some(include_dir) = optional_env_path("NCCL_INCLUDE_DIR")? {
        return headers_from_installed_include(include_dir, "NCCL_INCLUDE_DIR");
    }

    if let Some(source_dir) = optional_env_path("NCCL_SOURCE_DIR")? {
        return headers_from_source_tree(&source_dir, "NCCL_SOURCE_DIR");
    }

    if let Some(repo_root) = repo_root {
        return headers_from_source_tree(repo_root, "the containing NCCL checkout");
    }

    Err(format!(
        "set NCCL_INCLUDE_DIR to an installed include directory containing {NCCL_HEADER} and \
         {NCCL_DEVICE_HEADER}, or set NCCL_SOURCE_DIR to a built NCCL source checkout"
    ))
}

fn headers_from_installed_include(
    include_dir: PathBuf,
    source: &str,
) -> Result<NcclHeaders, String> {
    require_directory(&include_dir, source)?;
    let nccl_header = include_dir.join(NCCL_HEADER);
    let device_header = include_dir.join(NCCL_DEVICE_HEADER);
    require_file(&nccl_header, source)?;
    require_file(&device_header, source)?;
    require_directory(&include_dir.join("nccl_device"), source)?;

    Ok(NcclHeaders {
        include_dirs: vec![include_dir],
        nccl_header,
        device_header,
    })
}

fn headers_from_source_tree(source_dir: &Path, source: &str) -> Result<NcclHeaders, String> {
    require_directory(source_dir, source)?;

    // Prefer the staged public include tree because it exactly models an NCCL
    // installation. Fall back to src/include only for the public device tree;
    // nccl.h must still be generated from nccl.h.in in build/include.
    let staged = source_dir.join("build/include");
    let public_source = source_dir.join("src/include");
    let nccl_header = staged.join(NCCL_HEADER);
    require_file(&nccl_header, source)?;

    let (device_include, device_header) = if staged.join(NCCL_DEVICE_HEADER).is_file() {
        (staged.clone(), staged.join(NCCL_DEVICE_HEADER))
    } else if public_source.join(NCCL_DEVICE_HEADER).is_file() {
        (
            public_source.clone(),
            public_source.join(NCCL_DEVICE_HEADER),
        )
    } else {
        return Err(format!(
            "{source} ({}) has {NCCL_HEADER}, but no public {NCCL_DEVICE_HEADER} in {} or {}",
            source_dir.display(),
            staged.display(),
            public_source.display()
        ));
    };
    require_directory(&device_include.join("nccl_device"), source)?;

    let mut include_dirs = vec![staged];
    if device_include != include_dirs[0] {
        include_dirs.push(device_include);
    }

    Ok(NcclHeaders {
        include_dirs,
        nccl_header,
        device_header,
    })
}

fn find_cuda_include() -> Result<PathBuf, String> {
    for variable in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = optional_env_path(variable)? {
            require_directory(&root, variable)?;
            let include = root.join("include");
            require_cuda_include(&include, variable)?;
            return Ok(include);
        }
    }

    let path = env::var_os("PATH").unwrap_or_default();
    for bin_dir in env::split_paths(&path) {
        for executable in ["nvcc", "nvcc.exe"] {
            let nvcc = bin_dir.join(executable);
            if nvcc.is_file() {
                let resolved_nvcc = nvcc.canonicalize().unwrap_or(nvcc);
                for executable_path in [bin_dir.join(executable), resolved_nvcc] {
                    let Some(root) = executable_path.parent().and_then(Path::parent) else {
                        continue;
                    };
                    let include = root.join("include");
                    if has_cuda_headers(&include) {
                        return Ok(include);
                    }
                }
            }
        }
    }

    Err(
        "could not find CUDA headers; set CUDA_HOME to the toolkit root or put nvcc in PATH"
            .to_owned(),
    )
}

fn find_nccl_lib_dir(
    repo_root: Option<&Path>,
    explicit_include_dir: bool,
) -> Result<Option<PathBuf>, String> {
    if let Some(lib_dir) = optional_env_path("NCCL_LIB_DIR")? {
        require_directory(&lib_dir, "NCCL_LIB_DIR")?;
        require_nccl_library(&lib_dir, "NCCL_LIB_DIR")?;
        return Ok(Some(lib_dir));
    }

    if explicit_include_dir {
        println!(
            "cargo:warning=NCCL_INCLUDE_DIR is set without a matching NCCL_LIB_DIR; \
             nccl-sys will use the platform linker's search paths for -lnccl rather than a \
             library from NCCL_SOURCE_DIR or this checkout's build/lib. Ensure the discovered \
             libnccl matches the selected headers."
        );
        return Ok(None);
    }

    if let Some(source_dir) = optional_env_path("NCCL_SOURCE_DIR")? {
        let lib_dir = source_dir.join("build/lib");
        if has_nccl_library(&lib_dir) {
            return Ok(Some(lib_dir));
        }
    }

    if let Some(repo_root) = repo_root {
        let lib_dir = repo_root.join("build/lib");
        if has_nccl_library(&lib_dir) {
            return Ok(Some(lib_dir));
        }
    }

    // An installed libnccl in the platform linker's standard search path does
    // not require an explicit cargo:rustc-link-search directive.
    Ok(None)
}

fn nccl_repo_root(manifest_dir: &Path) -> Option<PathBuf> {
    let candidate = manifest_dir.ancestors().nth(4)?;
    if candidate.join("src/nccl.h.in").is_file() {
        Some(candidate.to_path_buf())
    } else {
        None
    }
}

fn required_path(variable: &str) -> Result<PathBuf, String> {
    let value = env::var_os(variable).ok_or_else(|| format!("{variable} is not set"))?;
    if value.is_empty() {
        return Err(format!("{variable} is empty"));
    }
    Ok(PathBuf::from(value))
}

fn optional_env_path(variable: &str) -> Result<Option<PathBuf>, String> {
    match env::var_os(variable) {
        None => Ok(None),
        Some(value) if value.is_empty() => Err(format!("{variable} is set but empty")),
        Some(value) => Ok(Some(PathBuf::from(value))),
    }
}

fn require_file(path: &Path, source: &str) -> Result<(), String> {
    if path.is_file() {
        Ok(())
    } else {
        Err(format!(
            "{source} does not provide required public header {}",
            path.display()
        ))
    }
}

fn require_directory(path: &Path, source: &str) -> Result<(), String> {
    if path.is_dir() {
        Ok(())
    } else {
        Err(format!(
            "{source} does not name a required directory: {}",
            path.display()
        ))
    }
}

fn require_cuda_include(path: &Path, source: &str) -> Result<(), String> {
    if has_cuda_headers(path) {
        Ok(())
    } else {
        Err(format!(
            "{source} does not provide cuda.h and cuda_runtime.h under {}",
            path.display()
        ))
    }
}

fn has_cuda_headers(path: &Path) -> bool {
    path.join("cuda.h").is_file() && path.join("cuda_runtime.h").is_file()
}

fn require_nccl_library(path: &Path, source: &str) -> Result<(), String> {
    if has_nccl_library(path) {
        Ok(())
    } else {
        Err(format!(
            "{source} does not contain libnccl in {}",
            path.display()
        ))
    }
}

fn has_nccl_library(path: &Path) -> bool {
    ["libnccl.so", "libnccl.dylib", "nccl.lib"]
        .iter()
        .any(|name| path.join(name).is_file())
}
