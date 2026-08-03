import pytest

from rvai.results import (
    ReportRenderError,
    escape_markdown,
    render_markdown,
    save_markdown_report,
)


def test_escape_markdown_neutralizes_structural_characters() -> None:
    assert escape_markdown("unsafe|*value*_[x]") == (
        r"unsafe\|\*value\*\_\[x\]"
    )


@pytest.mark.parametrize("character", ["|", "`", "<", ">", "[", "]"])
def test_escape_markdown_covers_audited_characters(character) -> None:
    assert escape_markdown(character) == f"\\{character}"


def test_markdown_contains_complete_result_fields(run_record_factory) -> None:
    report = render_markdown(run_record_factory())

    for expected in (
        "Schema version",
        "Manifest digest",
        "Correctness verified",
        "Mean latency",
        "P95 latency",
        "Throughput",
        "Target architecture",
        "Performance representative",
        "Hardware profile was unavailable",
    ):
        assert expected in report


def test_markdown_escapes_dynamic_model_text(run_record_factory) -> None:
    report = render_markdown(run_record_factory(model="unsafe|*model*"))

    assert r"unsafe\|\*model\*" in report
    assert "# RVAI Run Report: unsafe|*model*" not in report


def test_report_store_creates_parent_and_prevents_overwrite(tmp_path) -> None:
    destination = tmp_path / "nested" / "report.md"

    save_markdown_report("first\n", destination)

    assert destination.read_text(encoding="utf-8") == "first\n"
    with pytest.raises(ReportRenderError, match="use --force"):
        save_markdown_report("second\n", destination)

    save_markdown_report("second\n", destination, force=True)
    assert destination.read_text(encoding="utf-8") == "second\n"
