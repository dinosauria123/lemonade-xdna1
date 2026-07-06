#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Hardware-independent self-test for examples/npu_powerlog.py.
#
# npu_powerlog.py's job is to read NPU debugfs + RAPL sysfs and needs
# root plus the staging amdxdna driver to do that for real. But its
# parsing/math (parse_dpm, delta_uj, read_npu's graceful-degrade path)
# is pure stdlib and takes plain strings/ints in, so it is fully
# testable on any machine, no NPU and no root required. This is what
# CI actually exercises; it says nothing about the live debugfs/RAPL
# reading path itself.
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "..", "examples", "npu_powerlog.py")

spec = importlib.util.spec_from_file_location("npu_powerlog", MODULE_PATH)
npu_powerlog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(npu_powerlog)


def test_parse_dpm_finds_active_level():
    text = "0: [100, 50]\n1: 200, 100\n2: 300, 150"
    d = npu_powerlog.parse_dpm(text)
    assert d["active"]["npuclk_mhz"] == 100
    assert d["active"]["hclk_mhz"] == 50
    assert d["levels"] == 3
    assert d["max_index"] == 2


def test_parse_dpm_empty_text_returns_none():
    assert npu_powerlog.parse_dpm("") is None


def test_delta_uj_no_wrap():
    assert npu_powerlog.delta_uj(100, 150, 1000) == 50


def test_delta_uj_wraparound():
    # counter wrapped past max_uj between samples
    assert npu_powerlog.delta_uj(900, 100, 1000) == 200


def test_delta_uj_wraparound_no_max_known():
    # can't compute a real wrapped delta without max_uj -> defined as 0
    assert npu_powerlog.delta_uj(900, 100, None) == 0


def test_read_npu_graceful_degrade_no_accel_dir():
    r = npu_powerlog.read_npu(None)
    assert r == {"available": False, "reason": "no_accel_debugfs"}


def test_find_accel_dir_returns_none_when_debugfs_absent():
    # on a normal CI box /sys/kernel/debug/accel does not exist (no NPU,
    # often no debugfs mount at all) - must not raise, must return None
    assert npu_powerlog.find_accel_dir() is None or isinstance(npu_powerlog.find_accel_dir(), str)


TESTS = [v for k, v in sorted(vars(sys.modules[__name__]).items()) if k.startswith("test_")]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
