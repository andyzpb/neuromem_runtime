from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuromem.evals.bench import SCENARIOS, VARIANTS, run_neuromem_bench
from neuromem.evals.external import CONTEXT_PACKING_STRATEGIES, EXTERNAL_ADAPTER_FACTORIES, EXTERNAL_MEMORY_DATASETS, PROVIDER_MODES, run_external_memory_eval
from neuromem.evals.framework import BASELINE_FACTORIES, run_coding_agent_eval, run_lifecycle_diagnostic_eval, run_memory_only_eval, run_paper_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic NeuroMem evaluation and ablation benchmarks.")
    parser.add_argument("--suite", choices=["diagnostic", "memory-only", "external-memory", "coding-agent", "lifecycle-diagnostic", "paper", "all"], default="diagnostic")
    parser.add_argument("--output", choices=["json", "jsonl"], default="json")
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS), help="Scenario to run. Repeatable. Defaults to all.")
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS), help="Variant to run. Repeatable. Defaults to all.")
    parser.add_argument("--baseline", action="append", choices=sorted(set(BASELINE_FACTORIES) | set(EXTERNAL_ADAPTER_FACTORIES)), help="Baseline to run for experiment-framework suites. Repeatable. Defaults depend on suite.")
    parser.add_argument("--benchmark", choices=sorted({name for name, _ in EXTERNAL_MEMORY_DATASETS} | {"longmemeval", "longmemeval_v2"}), default="longmemeval_style")
    parser.add_argument("--split", default="dev-fixture")
    parser.add_argument("--data-path", type=Path, help="Local JSON/JSONL file for real benchmark ingestion.")
    parser.add_argument("--context-packing", choices=sorted(CONTEXT_PACKING_STRATEGIES), default="RawRetrievedContext")
    parser.add_argument("--provider-mode", choices=sorted(PROVIDER_MODES), default="offline")
    parser.add_argument("--provider-model", default="deepseek-v4-flash")
    parser.add_argument("--provider-base-url", default="https://api.deepseek.com")
    parser.add_argument("--provider-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--include-external-adapters", action="store_true", help="Enable opt-in real external baseline adapters when registered.")
    parser.add_argument("--scoring-mode", choices=["retrieval_only", "official_eval_function", "both"], default="both")
    parser.add_argument("--enable-llm-judge", action="store_true")
    parser.add_argument("--max-examples", type=int, help="Limit external-memory examples for real-data smoke runs.")
    parser.add_argument("--budget-tokens", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="Optional output file.")
    args = parser.parse_args()

    if args.suite == "diagnostic":
        rendered = run_neuromem_bench(
            variants=args.variant,
            scenarios=args.scenario,
            seed=args.seed,
            output_format=args.output,
        )
    elif args.suite == "memory-only":
        rendered = run_memory_only_eval(
            baselines=args.baseline,
            budget_tokens=args.budget_tokens,
            seed=args.seed,
            output_format=args.output,
        )
    elif args.suite == "external-memory":
        rendered = run_external_memory_eval(
            benchmark_name=args.benchmark,
            split=args.split,
            data_path=args.data_path,
            baselines=args.baseline,
            budget_tokens=args.budget_tokens,
            seed=args.seed,
            context_packing=args.context_packing,
            provider_mode=args.provider_mode,
            provider_model=args.provider_model,
            provider_base_url=args.provider_base_url,
            provider_api_key_env=args.provider_api_key_env,
            include_external_adapters=args.include_external_adapters,
            scoring_mode=args.scoring_mode,
            enable_llm_judge=args.enable_llm_judge,
            max_examples=args.max_examples,
            output_format=args.output,
        )
    elif args.suite == "coding-agent":
        rendered = run_coding_agent_eval(
            baselines=args.baseline,
            budget_tokens=args.budget_tokens,
            seed=args.seed,
            output_format=args.output,
        )
    elif args.suite == "lifecycle-diagnostic":
        rendered = run_lifecycle_diagnostic_eval(
            baselines=args.baseline,
            budget_tokens=args.budget_tokens,
            seed=args.seed,
            output_format=args.output,
        )
    elif args.suite == "paper":
        rendered = run_paper_eval(
            baselines=args.baseline,
            budget_tokens=args.budget_tokens,
            seed=args.seed,
            output_format=args.output,
        )
    else:
        diagnostic = run_neuromem_bench(variants=args.variant, scenarios=args.scenario, seed=args.seed)
        memory_only = run_memory_only_eval(baselines=args.baseline, budget_tokens=args.budget_tokens, seed=args.seed)
        external_memory = run_external_memory_eval(
            benchmark_name=args.benchmark,
            split=args.split,
            data_path=args.data_path,
            baselines=args.baseline,
            budget_tokens=args.budget_tokens,
            seed=args.seed,
            context_packing=args.context_packing,
            provider_mode=args.provider_mode,
            provider_model=args.provider_model,
            provider_base_url=args.provider_base_url,
            provider_api_key_env=args.provider_api_key_env,
            include_external_adapters=args.include_external_adapters,
            scoring_mode=args.scoring_mode,
            enable_llm_judge=args.enable_llm_judge,
        )
        coding_agent = run_coding_agent_eval(baselines=args.baseline, budget_tokens=args.budget_tokens, seed=args.seed)
        lifecycle = run_lifecycle_diagnostic_eval(baselines=args.baseline, budget_tokens=args.budget_tokens, seed=args.seed)
        paper = run_paper_eval(baselines=args.baseline, budget_tokens=args.budget_tokens, seed=args.seed)
        combined = {"diagnostic": diagnostic, "memory_only": memory_only, "external_memory": external_memory, "coding_agent": coding_agent, "lifecycle_diagnostic": lifecycle, "paper": paper}
        if args.output == "json":
            rendered = json.dumps(combined, sort_keys=True)
        else:
            rendered = "\n".join(
                [
                    json.dumps({"suite": "diagnostic", "report": diagnostic}, sort_keys=True),
                    json.dumps({"suite": "memory-only", "report": memory_only}, sort_keys=True),
                    json.dumps({"suite": "external-memory", "report": external_memory}, sort_keys=True),
                    json.dumps({"suite": "coding-agent", "report": coding_agent}, sort_keys=True),
                    json.dumps({"suite": "lifecycle-diagnostic", "report": lifecycle}, sort_keys=True),
                    json.dumps({"suite": "paper", "report": paper}, sort_keys=True),
                ]
            )
    assert isinstance(rendered, str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
