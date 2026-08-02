"""Generate structured workload execution plans."""

from pydantic import BaseModel, ConfigDict

from rvai.manifest import ModelManifest


class RunPlan(BaseModel):
    """Serializable plan produced without launching a workload."""

    model_config = ConfigDict(extra="forbid")

    model: str
    task: str
    format: str
    quantization: str
    runtime: str
    target: str
    resources: dict[str, int | str]
    riscv: dict[str, bool]
    requires_model_file: bool
    dry_run: bool


class RunPlanner:
    """Turn a validated Manifest into a dry-run execution plan."""

    def plan(
        self,
        manifest: ModelManifest,
        *,
        dry_run: bool = True,
        target: str = "native",
    ) -> RunPlan:
        if not dry_run:
            raise ValueError("RVAI V0.1 only supports dry-run planning")

        return RunPlan(
            model=manifest.name,
            task=manifest.task,
            format=manifest.format,
            quantization=manifest.quantization,
            runtime=manifest.runtime,
            target=target,
            resources={
                "min_memory_mb": manifest.resources.min_memory_mb,
                "recommended_threads": manifest.resources.recommended_threads,
            },
            riscv={
                "require_rv64": manifest.riscv.require_rv64,
                "prefer_rvv": manifest.riscv.prefer_rvv,
            },
            requires_model_file=manifest.format != "builtin",
            dry_run=True,
        )
