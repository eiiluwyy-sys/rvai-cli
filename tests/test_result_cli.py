import json

from typer.testing import CliRunner

import rvai.cli as cli
from rvai.hardware import HardwareProbeError
from rvai.results import NON_REPRESENTATIVE_MESSAGE, save_run_record


runner = CliRunner()


class FakeProbe:
    def detect(self):
        return None


def test_run_without_output_does_not_probe_hardware(
    monkeypatch,
    benchmark_result_factory,
) -> None:
    benchmark = benchmark_result_factory()

    class FakeAdapter:
        def execute(self, manifest, target):
            return benchmark

    def fail_if_probed():
        raise AssertionError("hardware must not be probed without --output")

    monkeypatch.setattr(cli, "BuiltinAdapter", FakeAdapter)
    monkeypatch.setattr(cli, "_hardware_probe", fail_if_probed)

    result = runner.invoke(cli.app, ["run", "builtin-gemm-int8"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == benchmark.model_dump(mode="json")


def test_run_output_saves_record_without_changing_stdout(
    tmp_path,
    monkeypatch,
    benchmark_result_factory,
) -> None:
    benchmark = benchmark_result_factory()

    class FakeAdapter:
        def execute(self, manifest, target):
            return benchmark

    monkeypatch.setattr(cli, "BuiltinAdapter", FakeAdapter)
    monkeypatch.setattr(cli, "_hardware_probe", FakeProbe)
    output = tmp_path / "nested" / "native.json"

    result = runner.invoke(
        cli.app,
        ["run", "builtin-gemm-int8", "--output", str(output)],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == benchmark.model_dump(mode="json")
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["result"] == benchmark.model_dump(mode="json")
    assert record["target"] == "native"
    assert record["hardware_profile"] is None
    assert record["command"] == [
        "rvai",
        "run",
        "builtin-gemm-int8",
        "--target",
        "native",
    ]


def test_run_output_requires_force_to_overwrite(
    tmp_path,
    monkeypatch,
    benchmark_result_factory,
) -> None:
    benchmark = benchmark_result_factory()

    class FakeAdapter:
        def execute(self, manifest, target):
            return benchmark

    monkeypatch.setattr(cli, "BuiltinAdapter", FakeAdapter)
    monkeypatch.setattr(cli, "_hardware_probe", FakeProbe)
    output = tmp_path / "native.json"
    arguments = ["run", "builtin-gemm-int8", "--output", str(output)]

    assert runner.invoke(cli.app, arguments).exit_code == 0
    refused = runner.invoke(cli.app, arguments)
    forced = runner.invoke(cli.app, [*arguments, "--force"])

    assert refused.exit_code == 1
    assert "use --force" in refused.output
    assert "Traceback" not in refused.output
    assert forced.exit_code == 0


def test_hardware_snapshot_failure_does_not_discard_successful_run(
    tmp_path,
    monkeypatch,
    benchmark_result_factory,
) -> None:
    benchmark = benchmark_result_factory()

    class FakeAdapter:
        def execute(self, manifest, target):
            return benchmark

    class FailingProbe:
        def detect(self):
            raise HardwareProbeError("profile unavailable")

    monkeypatch.setattr(cli, "BuiltinAdapter", FakeAdapter)
    monkeypatch.setattr(cli, "_hardware_probe", FailingProbe)
    output = tmp_path / "native.json"

    result = runner.invoke(
        cli.app,
        ["run", "builtin-gemm-int8", "--output", str(output)],
    )

    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["hardware_profile"] is None


def test_nondefault_workload_is_captured_in_reproducible_command(
    tmp_path,
    monkeypatch,
    benchmark_result_factory,
) -> None:
    benchmark = benchmark_result_factory(m=32, n=32, k=32, iterations=2)

    class FakeAdapter:
        def execute(self, manifest, target):
            return benchmark

    monkeypatch.setattr(cli, "BuiltinAdapter", FakeAdapter)
    monkeypatch.setattr(cli, "_hardware_probe", FakeProbe)
    output = tmp_path / "controlled.json"

    result = runner.invoke(
        cli.app,
        ["run", "builtin-gemm-int8", "--output", str(output)],
    )

    command = json.loads(output.read_text(encoding="utf-8"))["command"]
    assert result.exit_code == 0
    assert command[:5] == [
        "env",
        "RVAI_GEMM_M=32",
        "RVAI_GEMM_N=32",
        "RVAI_GEMM_K=32",
        "RVAI_GEMM_ITERATIONS=2",
    ]


def test_report_command_writes_markdown(
    tmp_path,
    run_record_factory,
) -> None:
    source = tmp_path / "record.json"
    output = tmp_path / "reports" / "record.md"
    save_run_record(run_record_factory(), source)

    result = runner.invoke(
        cli.app,
        [
            "report",
            str(source),
            "--format",
            "markdown",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "# RVAI Run Report" in output.read_text(encoding="utf-8")


def test_report_rejects_invalid_json_without_traceback(tmp_path) -> None:
    source = tmp_path / "broken.json"
    source.write_text("not-json", encoding="utf-8")

    result = runner.invoke(cli.app, ["report", str(source)])

    assert result.exit_code == 1
    assert "Invalid JSON" in result.output
    assert "Traceback" not in result.output


def test_compare_command_blocks_qemu_speedup(
    tmp_path,
    benchmark_result_factory,
    run_record_factory,
) -> None:
    native_path = tmp_path / "native.json"
    qemu_path = tmp_path / "qemu.json"
    save_run_record(run_record_factory(), native_path)
    save_run_record(
        run_record_factory(
            run_id="run-right",
            target="qemu-riscv64",
            result=benchmark_result_factory(
                representative=False,
                execution_environment="qemu-user",
                target_architecture="riscv64",
                mean_latency=30.0,
            ),
        ),
        qemu_path,
    )

    result = runner.invoke(cli.app, ["compare", str(native_path), str(qemu_path)])
    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert payload["performance"]["available"] is False
    assert payload["performance"]["latency_ratio_right_over_left"] is None
    assert payload["performance"]["message"] == NON_REPRESENTATIVE_MESSAGE
