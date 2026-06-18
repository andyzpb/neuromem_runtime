from __future__ import annotations

import csv
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neuromem.evals.external import EXTERNAL_ARTIFACTS, build_external_memory_artifacts, run_external_memory_eval
from neuromem.evals.framework import run_paper_eval


REQUIRED_ARTIFACTS = [
    "report.json",
    "report.jsonl",
    "manifest.json",
    "tables/main_baselines.csv",
    "tables/internal_ablations.csv",
    "tables/scorecard.csv",
    "tables/cost_latency.csv",
    "tables/metadata.csv",
    "case_studies/trace_replay.md",
    "summary.md",
]

MAIN_BASELINE_COLUMNS = [
    "baseline",
    "answer_accuracy",
    "evidence_recall",
    "stale_memory_reuse",
    "repeat_failure_reduction",
    "procedural_rule_adoption",
    "memory_pollution",
    "explanation_completeness",
    "context_tokens",
    "latency_ms",
    "token_cost_proxy",
]

ABLATION_COLUMNS = [
    "variant",
    "multi_hop_recall",
    "stale_memory_reuse",
    "procedural_rule_adoption",
    "memory_pollution",
    "explanation_completeness",
    "policy_rejection_accuracy",
    "capture_precision",
    "edge_reinforcement_usefulness",
    "conflict_invalidation_accuracy",
    "cache_hit_rate",
    "cache_stale_hit_rate",
]

SCORECARD_COLUMNS = [
    "baseline",
    "category",
    "metric",
    "value",
]

COST_COLUMNS = [
    "baseline",
    "suite",
    "scenario",
    "context_tokens",
    "latency_ms",
    "token_cost_proxy",
    "memory_item_count",
    "edge_count",
]

METADATA_COLUMNS = [
    "baseline",
    "family",
    "claim_axis",
    "paper_role",
    "dependency_mode",
    "known_limitation",
]


def _csv(rows: list[dict[str, object]], columns: list[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return output.getvalue()


def _metric(metrics: dict[str, object], key: str) -> object:
    return metrics.get(key, "")


def _paper_report(seed: int, budget_tokens: int) -> dict[str, Any]:
    report = run_paper_eval(seed=seed, budget_tokens=budget_tokens)
    if not isinstance(report, dict):
        raise TypeError("run_paper_eval must return a dict for artifact export")
    return report


def _git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        head = Path(".git/HEAD")
        if not head.exists():
            return "unknown"
        content = head.read_text(encoding="utf-8").strip()
        if content.startswith("ref: "):
            ref_path = Path(".git") / content.removeprefix("ref: ").strip()
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip() or "unknown"
            return "unknown"
        return content or "unknown"
    return result.stdout.strip() or "unknown"


def _main_baseline_rows(report: dict[str, Any]) -> list[dict[str, object]]:
    aggregate = dict(report.get("aggregate", {}))
    metadata = dict(report.get("baseline_metadata", {}))
    rows: list[dict[str, object]] = []
    for baseline, meta in sorted(metadata.items()):
        paper_role = str(meta.get("paper_role", "")) if isinstance(meta, dict) else ""
        if paper_role != "full_system" and not paper_role.startswith("main_table"):
            continue
        metrics = dict(aggregate.get(baseline, {}))
        rows.append(
            {
                "baseline": baseline,
                "answer_accuracy": _metric(metrics, "answer_accuracy"),
                "evidence_recall": _metric(metrics, "evidence_recall"),
                "stale_memory_reuse": _metric(metrics, "stale_memory_reuse"),
                "repeat_failure_reduction": _metric(metrics, "repeat_failure_reduction"),
                "procedural_rule_adoption": _metric(metrics, "procedural_rule_adoption"),
                "memory_pollution": _metric(metrics, "memory_pollution"),
                "explanation_completeness": _metric(metrics, "explanation_completeness"),
                "context_tokens": _metric(metrics, "context_tokens"),
                "latency_ms": _metric(metrics, "latency_ms"),
                "token_cost_proxy": _metric(metrics, "token_cost_proxy"),
            }
        )
    return rows


def _ablation_rows(report: dict[str, Any]) -> list[dict[str, object]]:
    ablation = dict(report.get("ablation_report", {}))
    aggregate = dict(ablation.get("aggregate", {}))
    rows: list[dict[str, object]] = []
    for variant, metrics_obj in sorted(aggregate.items()):
        metrics = dict(metrics_obj) if isinstance(metrics_obj, dict) else {}
        row = {"variant": variant}
        row.update({column: _metric(metrics, column) for column in ABLATION_COLUMNS if column != "variant"})
        rows.append(row)
    return rows


def _scorecard_rows(report: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in list(report.get("runs", [])):
        if not isinstance(run, dict):
            continue
        baseline = str(run.get("baseline", ""))
        scorecard = dict(run.get("scorecard", {}))
        for category, metrics_obj in sorted(scorecard.items()):
            if not isinstance(metrics_obj, dict):
                continue
            for metric, value in sorted(metrics_obj.items()):
                rows.append({"baseline": baseline, "category": category, "metric": metric, "value": value})
    return rows


def _cost_rows(report: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in list(report.get("runs", [])):
        if isinstance(run, dict):
            rows.append({column: run.get(column, "") for column in COST_COLUMNS})
    return rows


def _metadata_rows(report: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metadata = dict(report.get("baseline_metadata", {}))
    for baseline, meta_obj in sorted(metadata.items()):
        meta = dict(meta_obj) if isinstance(meta_obj, dict) else {}
        row = {"baseline": baseline}
        row.update({column: meta.get(column, "") for column in METADATA_COLUMNS if column != "baseline"})
        rows.append(row)
    return rows


def _trace_case_study(report: dict[str, Any]) -> str:
    neuromem_runs = [run for run in list(report.get("runs", [])) if isinstance(run, dict) and run.get("baseline") == "NeuroMem"]
    selected = next((run for run in neuromem_runs if run.get("suite") == "memory-only"), neuromem_runs[0] if neuromem_runs else {})
    selected_ids = selected.get("selected_ids", [])
    suppressed_ids = selected.get("suppressed_ids", [])
    graph_paths = selected.get("graph_paths", [])
    validator_rejections = selected.get("validator_rejections", [])
    ledger = selected.get("ledger", {})
    ledger_transactions = ledger.get("transactions", []) if isinstance(ledger, dict) else []
    evidence = []
    for transaction in ledger_transactions[:5]:
        if isinstance(transaction, dict):
            evidence.append(
                {
                    "operation": transaction.get("operation", ""),
                    "phase": transaction.get("phase", ""),
                    "validator_decision": transaction.get("validator_decision", ""),
                    "evidence": transaction.get("evidence", []),
                }
            )
    return "\n".join(
        [
            "# NeuroMem Trace Replay Case Study",
            "",
            f"Suite: {selected.get('suite', '')}",
            f"Scenario: {selected.get('scenario', '')}",
            "",
            f"Selected ids: {json.dumps(selected_ids, sort_keys=True)}",
            f"Suppressed ids: {json.dumps(suppressed_ids, sort_keys=True)}",
            f"Validator decisions: {json.dumps(validator_rejections, sort_keys=True)}",
            f"Graph paths: {json.dumps(graph_paths, sort_keys=True)}",
            f"Ledger evidence: {json.dumps(evidence, sort_keys=True)}",
            "",
        ]
    )


def _summary(report: dict[str, Any], *, seed: int, budget_tokens: int, command: str, git_commit: str) -> str:
    main_rows = _main_baseline_rows(report)
    ablation_rows = _ablation_rows(report)
    return "\n".join(
        [
            "# NeuroMem Paper Experiment Summary",
            "",
            f"Command: `{command}`",
            f"Git commit: `{git_commit}`",
            f"Seed: `{seed}`",
            f"Budget tokens: `{budget_tokens}`",
            "",
            f"Main baselines: {len(main_rows)}",
            f"Internal ablations: {len(ablation_rows)}",
            "",
            "Generated artifacts are stored next to this summary.",
            "",
        ]
    )


def build_paper_artifacts(
    report: dict[str, Any] | None = None,
    *,
    seed: int = 0,
    budget_tokens: int = 220,
    command: str = "python -m neuromem.evals.run_experiment --suite paper",
    git_commit: str | None = None,
    timestamp: str | None = None,
) -> dict[str, str]:
    report = _paper_report(seed, budget_tokens) if report is None else report
    git_commit = git_commit or _git_commit()
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    report_json = json.dumps(report, sort_keys=True)
    report_jsonl_lines = [json.dumps(run, sort_keys=True) for run in list(report.get("runs", []))]
    report_jsonl_lines.append(json.dumps({key: report.get(key) for key in ["aggregate", "baseline_metadata", "ablation_report", "scorecard"]}, sort_keys=True))
    generated_files = list(REQUIRED_ARTIFACTS)
    manifest = {
        "suite": "paper",
        "seed": seed,
        "budget_tokens": budget_tokens,
        "command": command,
        "git_commit": git_commit,
        "timestamp": timestamp,
        "docker_note": "Run with docker compose run --rm neuromem for accepted verification.",
        "generated_files": generated_files,
        "baseline_list": sorted(dict(report.get("baseline_metadata", {}))),
        "ablation_variant_list": sorted(dict(dict(report.get("ablation_report", {})).get("aggregate", {}))),
    }
    artifacts = {
        "report.json": report_json,
        "report.jsonl": "\n".join(report_jsonl_lines) + "\n",
        "manifest.json": json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        "tables/main_baselines.csv": _csv(_main_baseline_rows(report), MAIN_BASELINE_COLUMNS),
        "tables/internal_ablations.csv": _csv(_ablation_rows(report), ABLATION_COLUMNS),
        "tables/scorecard.csv": _csv(_scorecard_rows(report), SCORECARD_COLUMNS),
        "tables/cost_latency.csv": _csv(_cost_rows(report), COST_COLUMNS),
        "tables/metadata.csv": _csv(_metadata_rows(report), METADATA_COLUMNS),
        "case_studies/trace_replay.md": _trace_case_study(report),
        "summary.md": _summary(report, seed=seed, budget_tokens=budget_tokens, command=command, git_commit=git_commit),
    }
    return artifacts


def build_external_memory_artifacts(
    *,
    data_path: str | Path | None = None,
    benchmark_name: str = "longmemeval_style",
    split: str = "dev-fixture",
    baselines: list[str] | None = None,
    seeds: list[int] | None = None,
    budget_tokens: list[int] | None = None,
    context_packings: list[str] | None = None,
    provider_mode: str = "offline",
    provider_model: str = "deepseek-v4-flash",
    provider_base_url: str = "https://api.deepseek.com",
    provider_api_key_env: str = "DEEPSEEK_API_KEY",
    include_external_adapters: bool = False,
    command: str = "python -m neuromem.evals.run_experiment --suite external-memory",
) -> dict[str, str]:
    from neuromem.evals.external import build_external_memory_artifacts as _build

    return _build(
        data_path=data_path,
        benchmark_name=benchmark_name,
        split=split,
        baselines=baselines,
        seeds=seeds,
        budget_tokens=budget_tokens,
        context_packings=context_packings,  # type: ignore[arg-type]
        provider_mode=provider_mode,  # type: ignore[arg-type]
        provider_model=provider_model,
        provider_base_url=provider_base_url,
        provider_api_key_env=provider_api_key_env,
        include_external_adapters=include_external_adapters,
        command=command,
    )


def write_paper_artifacts(artifacts: dict[str, str], out_dir: Path) -> None:
    for relative, content in artifacts.items():
        target = out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def write_external_memory_artifacts(artifacts: dict[str, str], out_dir: Path) -> None:
    for relative, content in artifacts.items():
        target = out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def run_paper_experiment(
    *,
    out_dir: str | Path,
    seed: int = 0,
    budget_tokens: int = 220,
    command: str = "python -m neuromem.evals.run_experiment --suite paper",
) -> dict[str, Any]:
    out_path = Path(out_dir)
    artifacts = build_paper_artifacts(seed=seed, budget_tokens=budget_tokens, command=command)
    write_paper_artifacts(artifacts, out_path)
    manifest = json.loads(artifacts["manifest.json"])
    return manifest


def run_external_memory_experiment(
    *,
    out_dir: str | Path,
    data_path: str | Path | None = None,
    benchmark_name: str = "longmemeval_style",
    split: str = "dev-fixture",
    baselines: list[str] | None = None,
    seeds: list[int] | None = None,
    budget_tokens: list[int] | None = None,
    context_packings: list[str] | None = None,
    provider_mode: str = "offline",
    provider_model: str = "deepseek-v4-flash",
    provider_base_url: str = "https://api.deepseek.com",
    provider_api_key_env: str = "DEEPSEEK_API_KEY",
    include_external_adapters: bool = False,
    command: str = "python -m neuromem.evals.run_experiment --suite external-memory",
) -> dict[str, Any]:
    out_path = Path(out_dir)
    artifacts = build_external_memory_artifacts(
        data_path=data_path,
        benchmark_name=benchmark_name,
        split=split,
        baselines=baselines,
        seeds=seeds,
        budget_tokens=budget_tokens,
        context_packings=context_packings,
        provider_mode=provider_mode,
        provider_model=provider_model,
        provider_base_url=provider_base_url,
        provider_api_key_env=provider_api_key_env,
        include_external_adapters=include_external_adapters,
        command=command,
    )
    write_external_memory_artifacts(artifacts, out_path)
    manifest = json.loads(artifacts["manifest.json"])
    return manifest
