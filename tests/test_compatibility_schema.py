import pytest
from pydantic import ValidationError

from rvai.compatibility.schema import CompatibilityIssue, CompatibilityReport


def test_report_lists_have_independent_defaults() -> None:
    first = CompatibilityReport(model="first", compatible=True, ready=True)
    second = CompatibilityReport(model="second", compatible=True, ready=True)

    first.warnings.append(CompatibilityIssue(code="example", message="Example"))

    assert second.warnings == []


def test_issue_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CompatibilityIssue(code="example", message="Example", unexpected=True)
