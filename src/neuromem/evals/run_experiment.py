from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuromem.evals.experiment import run_paper_experiment
from neuromem.evals.external import CONTEXT_PACKING_STRATEGIES, PROVIDER_MODES, run_external_memory_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-ready NeuroMem experiment artifacts.")
    parser.add_argument("--suite", choices=["paper", "external-memory"], default="paper")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", help="Seed list for external-memory experiments.")
    parser.add_argument("--budget-tokens", type=int, nargs="+", default=[220])
    parser.add_argument("--benchmark", default="longmemeval_style")
    parser.add_argument("--split", default="dev-fixture")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--context-packing", choices=sorted(CONTEXT_PACKING_STRATEGIES), action="append", help="Context packing strategy for external-memory experiments. Repeatable. Defaults to RawRetrievedContext.")
    parser.add_argument("--provider-mode", choices=sorted(PROVIDER_MODES), default="offline")
    parser.add_argument("--provider-model", default="deepseek-v4-flash")
    parser.add_argument("--provider-base-url", default="https://api.deepseek.com")
    parser.add_argument("--provider-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--baseline", action="append")
    parser.add_argument("--include-external-adapters", action="store_true")
    parser.add_argument("--scoring-mode", choices=["retrieval_only", "official_eval_function", "both"], default="both")
    parser.add_argument("--enable-llm-judge", action="store_true")
    parser.add_argument("--max-examples", type=int, help="Limit external-memory examples for real-data smoke runs.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel external-memory report tasks. Defaults to 1.")
    args = parser.parse_args()

    command = f"python -m neuromem.evals.run_experiment --suite {args.suite} --out-dir {args.out_dir}"
    if args.suite == "paper":
        manifest = run_paper_experiment(out_dir=args.out_dir, seed=args.seed, budget_tokens=args.budget_tokens[0], command=command)
    else:
        manifest = run_external_memory_experiment(
            out_dir=args.out_dir,
            data_path=args.data_path,
            benchmark_name=args.benchmark,
            split=args.split,
            baselines=args.baseline,
            seeds=args.seeds or [args.seed],
            budget_tokens=args.budget_tokens,
            context_packings=args.context_packing,
            provider_mode=args.provider_mode,
            provider_model=args.provider_model,
            provider_base_url=args.provider_base_url,
            provider_api_key_env=args.provider_api_key_env,
            include_external_adapters=args.include_external_adapters,
            scoring_mode=args.scoring_mode,
            enable_llm_judge=args.enable_llm_judge,
            max_examples=args.max_examples,
            jobs=args.jobs,
            command=command,
        )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
