"""Tests for the per-module coverage gate (scripts/check_coverage.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_coverage.py"
_spec = importlib.util.spec_from_file_location("check_coverage", _SCRIPT)
assert _spec and _spec.loader
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def _report(**modules: float) -> dict:
    """Build a coverage-json-shaped report from module -> percent."""
    return {
        "files": {
            f"src/{mod}": {"summary": {"percent_covered": pct}}
            for mod, pct in modules.items()
        }
    }


def test_all_modules_above_floor_passes():
    report = _report(**{"parquet_analyzer/a.py": 96.0, "parquet_analyzer/b.py": 100.0})
    ok, rows, warnings = cc.evaluate(report)
    assert ok is True
    assert {r["module"] for r in rows} == {
        "parquet_analyzer/a.py",
        "parquet_analyzer/b.py",
    }
    assert warnings == []


def test_module_below_floor_without_gap_fails():
    report = _report(**{"parquet_analyzer/a.py": 94.9})
    ok, rows, _ = cc.evaluate(report)
    assert ok is False
    assert rows[0]["passed"] is False


def test_excluded_modules_are_not_gated(monkeypatch):
    monkeypatch.setattr(cc, "EXCLUDE", {"parquet_analyzer/__init__.py"})
    report = _report(
        **{"parquet_analyzer/__init__.py": 0.0, "parquet_analyzer/a.py": 99.0}
    )
    ok, rows, _ = cc.evaluate(report)
    assert ok is True
    assert {r["module"] for r in rows} == {"parquet_analyzer/a.py"}


def test_known_gap_at_baseline_passes(monkeypatch):
    monkeypatch.setattr(
        cc, "KNOWN_GAPS", {"parquet_analyzer/legacy.py": {"floor": 76.0, "issue": 28}}
    )
    report = _report(**{"parquet_analyzer/legacy.py": 76.0})
    ok, rows, _ = cc.evaluate(report)
    assert ok is True
    assert rows[0]["issue"] == 28


def test_known_gap_below_baseline_fails_ratchet(monkeypatch):
    monkeypatch.setattr(
        cc, "KNOWN_GAPS", {"parquet_analyzer/legacy.py": {"floor": 76.0, "issue": 28}}
    )
    report = _report(**{"parquet_analyzer/legacy.py": 75.0})
    ok, _, _ = cc.evaluate(report)
    assert ok is False  # coverage regressed below the recorded baseline


def test_known_gap_improved_warns_to_ratchet(monkeypatch):
    monkeypatch.setattr(
        cc, "KNOWN_GAPS", {"parquet_analyzer/legacy.py": {"floor": 76.0, "issue": 28}}
    )
    report = _report(**{"parquet_analyzer/legacy.py": 80.0})
    ok, _, warnings = cc.evaluate(report)
    assert ok is True
    assert any("ratchet" in w for w in warnings)


def test_known_gap_cleared_floor_warns_to_remove(monkeypatch):
    monkeypatch.setattr(
        cc, "KNOWN_GAPS", {"parquet_analyzer/legacy.py": {"floor": 76.0, "issue": 28}}
    )
    report = _report(**{"parquet_analyzer/legacy.py": 96.0})
    ok, _, warnings = cc.evaluate(report)
    assert ok is True
    assert any("remove" in w for w in warnings)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("src/parquet_analyzer/_html.py", "parquet_analyzer/_html.py"),
        ("parquet_analyzer/_html.py", "parquet_analyzer/_html.py"),
        ("src\\parquet_analyzer\\_html.py", "parquet_analyzer/_html.py"),
    ],
)
def test_module_key_normalization(path, expected):
    assert cc._module_key(path) == expected


def test_main_exit_code_on_failure(tmp_path):
    import json

    report = _report(**{"parquet_analyzer/a.py": 10.0})
    p = tmp_path / "coverage.json"
    p.write_text(json.dumps(report))
    assert cc.main(["check_coverage.py", str(p)]) == 1


def test_main_exit_code_on_success(tmp_path):
    import json

    report = _report(**{"parquet_analyzer/a.py": 99.0})
    p = tmp_path / "coverage.json"
    p.write_text(json.dumps(report))
    assert cc.main(["check_coverage.py", str(p)]) == 0
