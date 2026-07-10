# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

def pytest_addoption(parser):
    parser.addoption(
        "--checkpoint-mode",
        action="store",
        default="shim",
        choices=("shim", "no-shim"),
        help="Run checkpoint scenarios with or without the checkpoint shim preloaded.",
    )
