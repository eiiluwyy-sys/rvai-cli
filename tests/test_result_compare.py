from rvai.results import NON_REPRESENTATIVE_MESSAGE, compare_run_records


def test_native_and_qemu_never_report_performance_ratio(
    benchmark_result_factory,
    run_record_factory,
) -> None:
    native = run_record_factory()
    qemu = run_record_factory(
        run_id="run-right",
        target="qemu-riscv64",
        result=benchmark_result_factory(
            representative=False,
            execution_environment="qemu-user",
            target_architecture="riscv64",
            mean_latency=30.0,
        ),
    )

    comparison = compare_run_records(native, qemu)

    assert comparison.left.result.latency_ms.mean == 10.0
    assert comparison.right.result.latency_ms.mean == 30.0
    assert comparison.performance.available is False
    assert comparison.performance.latency_ratio_right_over_left is None
    assert comparison.performance.message == NON_REPRESENTATIVE_MESSAGE
    assert any("execution_environment" in item for item in comparison.differences)


def test_different_matrix_dimensions_disable_ratio(
    benchmark_result_factory,
    run_record_factory,
) -> None:
    left = run_record_factory()
    right = run_record_factory(
        run_id="run-right",
        result=benchmark_result_factory(m=128),
    )

    comparison = compare_run_records(left, right)

    assert comparison.performance.available is False
    assert "matrix dimensions differ" in comparison.performance.message


def test_different_iterations_disable_ratio(
    benchmark_result_factory,
    run_record_factory,
) -> None:
    comparison = compare_run_records(
        run_record_factory(),
        run_record_factory(
            run_id="run-right",
            result=benchmark_result_factory(iterations=10),
        ),
    )

    assert comparison.performance.available is False
    assert "iteration counts differ" in comparison.performance.message


def test_representative_equivalent_runs_report_explicit_ratio(
    benchmark_result_factory,
    run_record_factory,
) -> None:
    comparison = compare_run_records(
        run_record_factory(result=benchmark_result_factory(mean_latency=10.0)),
        run_record_factory(
            run_id="run-right",
            result=benchmark_result_factory(mean_latency=25.0),
        ),
    )

    assert comparison.performance.available is True
    assert comparison.performance.latency_ratio_right_over_left == 2.5
    assert comparison.performance.message == (
        "Latency ratio is right / left: values below 1.0 mean right has lower "
        "latency, 1.0 means equal latency, and values above 1.0 mean right has "
        "higher latency."
    )


def test_different_models_disable_ratio(run_record_factory) -> None:
    comparison = compare_run_records(
        run_record_factory(),
        run_record_factory(run_id="run-right", model="other-model"),
    )

    assert comparison.performance.available is False
    assert "models differ" in comparison.performance.message


def test_unverified_correctness_disables_ratio(
    benchmark_result_factory,
    run_record_factory,
) -> None:
    comparison = compare_run_records(
        run_record_factory(),
        run_record_factory(
            run_id="run-right",
            result=benchmark_result_factory(correctness_verified=False),
        ),
    )

    assert comparison.performance.available is False
    assert "correctness was not verified" in comparison.performance.message
