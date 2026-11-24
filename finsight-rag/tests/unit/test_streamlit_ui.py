"""The demo UI renders without the API (clear error, no crash), via Streamlit AppTest."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "ui" / "streamlit_app.py"


def test_ui_renders_and_reports_unreachable_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINSIGHT_API_URL", "http://127.0.0.1:9")  # nothing listens here
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert any("API not reachable" in e.value for e in at.sidebar.error)
    labels = [s.label for s in at.selectbox]
    assert {"Example questions", "Pipeline"} <= set(labels)
    assert at.button[0].disabled  # no question yet
    examples = next(s for s in at.selectbox if s.label == "Example questions")
    examples.select(examples.options[1]).run()
    assert at.text_area[0].value.startswith("What is the FY2018 capital expenditure")
    assert not at.button[0].disabled
