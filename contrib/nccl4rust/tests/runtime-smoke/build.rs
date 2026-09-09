// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let cuda_root = env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let cuda_lib = [cuda_root.join("lib64"), cuda_root.join("lib")]
        .into_iter()
        .find(|directory| has_cudart(directory))
        .unwrap_or_else(|| {
            panic!(
                "CUDA_HOME={} does not contain lib64/libcudart.so or lib/libcudart.so",
                cuda_root.display()
            )
        });

    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
}

fn has_cudart(directory: &Path) -> bool {
    directory.join("libcudart.so").is_file()
}
