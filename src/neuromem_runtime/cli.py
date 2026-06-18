from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import neuromem_runtime as nmem
from neuromem_runtime.runtime import MemoryRuntime


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nmem", description="NeuroMem Runtime local CLI")
    parser.add_argument("--path", default=".neuromem", help="NeuroMem workspace path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a local NeuroMem workspace")
    init_parser.add_argument("--namespace", default="default")
    init_parser.add_argument("--agent-id", default="local-agent")

    doctor_parser = subparsers.add_parser("doctor", help="Check local runtime readiness")
    doctor_parser.add_argument("--namespace", default="default")

    observe_parser = subparsers.add_parser("observe", help="Observe JSONL memory events")
    observe_parser.add_argument("events", type=Path)
    observe_parser.add_argument("--namespace", default="default")
    observe_parser.add_argument("--commit", action="store_true", help="Validate and commit observed events as long-term memories")

    query_parser = subparsers.add_parser("query", help="Query local memory")
    query_parser.add_argument("query")
    query_parser.add_argument("--namespace", default="default")
    query_parser.add_argument("--budget-tokens", type=int, default=800)
    query_parser.add_argument("--json", action="store_true")

    sleep_parser = subparsers.add_parser("sleep", help="Run replay consolidation")
    sleep_parser.add_argument("--namespace", default="default")

    trace_parser = subparsers.add_parser("trace", help="Replay trace commands")
    trace_subparsers = trace_parser.add_subparsers(dest="trace_command", required=True)
    trace_show = trace_subparsers.add_parser("show", help="Show a trace")
    trace_show.add_argument("trace_id")
    trace_show.add_argument("--namespace", default="default")
    trace_export = trace_subparsers.add_parser("export", help="Export a trace")
    trace_export.add_argument("trace_id")
    trace_export.add_argument("--namespace", default="default")
    trace_export.add_argument("--format", choices=["json"], default="json")
    trace_export.add_argument("--out", type=Path)

    ledger_parser = subparsers.add_parser("ledger", help="Memory ledger commands")
    ledger_subparsers = ledger_parser.add_subparsers(dest="ledger_command", required=True)
    ledger_show = ledger_subparsers.add_parser("show", help="Show ledger events for a transaction")
    ledger_show.add_argument("transaction_id")
    ledger_show.add_argument("--namespace", default="default")
    ledger_written = ledger_subparsers.add_parser("why-written", help="Show why a memory was written or mutated")
    ledger_written.add_argument("memory_id")
    ledger_written.add_argument("--namespace", default="default")
    ledger_retrieved = ledger_subparsers.add_parser("why-retrieved", help="Show why a memory was retrieved in a trace")
    ledger_retrieved.add_argument("trace_id")
    ledger_retrieved.add_argument("memory_id")
    ledger_retrieved.add_argument("--namespace", default="default")
    ledger_replay = ledger_subparsers.add_parser("replay", help="Replay ledger events")
    ledger_replay.add_argument("--to-txn")
    ledger_replay.add_argument("--namespace", default="default")
    ledger_diff = ledger_subparsers.add_parser("diff", help="Compare two transactions")
    ledger_diff.add_argument("left_txn")
    ledger_diff.add_argument("right_txn")
    ledger_diff.add_argument("--namespace", default="default")

    retrieval_parser = subparsers.add_parser("retrieval", help="Retrieval inspection commands")
    retrieval_subparsers = retrieval_parser.add_subparsers(dest="retrieval_command", required=True)
    retrieval_explain = retrieval_subparsers.add_parser("explain", help="Explain activation retrieval for a trace")
    retrieval_explain.add_argument("trace_id")
    retrieval_explain.add_argument("--namespace", default="default")

    bench_parser = subparsers.add_parser("bench", help="Benchmark commands")
    bench_subparsers = bench_parser.add_subparsers(dest="bench_command", required=True)
    bench_run = bench_subparsers.add_parser("run", help="Run a benchmark")
    bench_run.add_argument("suite", choices=["mini"])
    bench_run.add_argument("--methods", default="full,vector,no_graph,no_pfc")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result = asyncio.run(_cmd_init(args))
        elif args.command == "doctor":
            result = asyncio.run(_cmd_doctor(args))
        elif args.command == "observe":
            result = asyncio.run(_cmd_observe(args))
        elif args.command == "query":
            result = asyncio.run(_cmd_query(args))
        elif args.command == "sleep":
            result = asyncio.run(_cmd_sleep(args))
        elif args.command == "trace":
            result = asyncio.run(_cmd_trace(args))
        elif args.command == "ledger":
            result = asyncio.run(_cmd_ledger(args))
        elif args.command == "retrieval":
            result = asyncio.run(_cmd_retrieval(args))
        elif args.command == "bench":
            result = _cmd_bench(args)
        else:
            parser.error(f"unsupported command: {args.command}")
            return
    except Exception as exc:  # pragma: no cover - argparse surface
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if result is not None:
        print(result)


async def _runtime(args: argparse.Namespace) -> MemoryRuntime:
    return await MemoryRuntime.local(namespace=getattr(args, "namespace", "default"), path=args.path)


async def _cmd_init(args: argparse.Namespace) -> str:
    runtime = await MemoryRuntime.local(namespace=args.namespace, path=args.path, agent_id=args.agent_id)
    return json.dumps({"workspace": str(runtime.config.path), "db_path": str(runtime.config.db_path), "traces_path": str(runtime.config.traces_path)}, sort_keys=True)


async def _cmd_doctor(args: argparse.Namespace) -> str:
    runtime = await _runtime(args)
    checks: dict[str, object] = {
        "python": sys.version.split()[0],
        "neuromem_runtime": nmem.__version__,
        "workspace_writable": _is_writable(runtime.config.path),
        "sqlite_ok": _sqlite_ok(runtime.config.db_path),
    }
    try:
        import langgraph  # noqa: F401

        checks["langgraph"] = "available"
    except ModuleNotFoundError:
        checks["langgraph"] = "not installed; install neuromem-runtime[langgraph] for LangGraph integration"
    checks["ok"] = checks["workspace_writable"] is True and checks["sqlite_ok"] is True
    return json.dumps(checks, sort_keys=True)


async def _cmd_observe(args: argparse.Namespace) -> str:
    runtime = await _runtime(args)
    count = 0
    memory_ids: list[str] = []
    with args.events.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            bundle = await runtime.observe_and_commit(event) if args.commit else await runtime.observe(event)
            count += 1
            if bundle.memory_id is not None:
                memory_ids.append(bundle.memory_id)
    return json.dumps({"observed": count, "memory_ids": memory_ids}, sort_keys=True)


async def _cmd_query(args: argparse.Namespace) -> str:
    runtime = await _runtime(args)
    context = await runtime.query(args.query, budget_tokens=args.budget_tokens)
    if args.json:
        return json.dumps(context.to_dict(), sort_keys=True)
    selected = ", ".join(context.selected_memory_ids) if context.selected_memory_ids else "none"
    return f"{context.to_prompt()}\n\nselected_memory_ids: {selected}\ntrace_id: {context.trace_id}"


async def _cmd_sleep(args: argparse.Namespace) -> str:
    runtime = await _runtime(args)
    report = await runtime.sleep()
    return json.dumps(report, sort_keys=True)


async def _cmd_trace(args: argparse.Namespace) -> str | None:
    runtime = await _runtime(args)
    trace = await runtime.replay_trace(args.trace_id)
    if trace is None:
        raise ValueError(f"trace not found: {args.trace_id}")
    rendered = json.dumps(trace, indent=2, sort_keys=True)
    if args.trace_command == "export" and args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
        return json.dumps({"out": str(args.out)}, sort_keys=True)
    return rendered


async def _cmd_ledger(args: argparse.Namespace) -> str:
    runtime = await _runtime(args)
    if args.ledger_command == "show":
        return json.dumps(runtime.ledger.show_transaction(args.transaction_id), indent=2, sort_keys=True)
    if args.ledger_command == "why-written":
        return json.dumps(runtime.ledger.why_written(args.memory_id), indent=2, sort_keys=True)
    if args.ledger_command == "why-retrieved":
        events = runtime.ledger.events_for_trace(args.trace_id)
        filtered = [
            event
            for event in events
            if args.memory_id in event.get("targets", [])
            or any(item.get("memory_id") == args.memory_id or item.get("event_id") == args.memory_id for item in event.get("evidence", []))
        ]
        return json.dumps(filtered, indent=2, sort_keys=True)
    if args.ledger_command == "replay":
        return json.dumps(runtime.ledger.replay(args.to_txn), indent=2, sort_keys=True)
    if args.ledger_command == "diff":
        return json.dumps(runtime.ledger.diff(args.left_txn, args.right_txn), indent=2, sort_keys=True)
    raise ValueError(f"unsupported ledger command: {args.ledger_command}")


async def _cmd_retrieval(args: argparse.Namespace) -> str:
    runtime = await _runtime(args)
    if args.retrieval_command == "explain":
        trace = await runtime.replay_trace(args.trace_id)
        if trace is None:
            raise ValueError(f"trace not found: {args.trace_id}")
        query_plan = trace.get("query_plan", {}) if isinstance(trace, dict) else {}
        if isinstance(query_plan, dict) and isinstance(query_plan.get("retrieval_ledger"), dict):
            return json.dumps(query_plan["retrieval_ledger"], indent=2, sort_keys=True)
        ledger = runtime.ledger.retrieval_explain(args.trace_id)
        if ledger is not None:
            return json.dumps(ledger, indent=2, sort_keys=True)
        raise ValueError(f"retrieval ledger not found for trace: {args.trace_id}")
    raise ValueError(f"unsupported retrieval command: {args.retrieval_command}")


def _cmd_bench(args: argparse.Namespace) -> str:
    try:
        from neuromem.evals.bench import run_neuromem_bench
    except ModuleNotFoundError as exc:
        raise RuntimeError("bench requires the bundled NeuroMem evaluation package and optional neuromem-runtime[eval] dependencies") from exc
    variants = []
    for method in args.methods.split(","):
        name = method.strip()
        if name == "full":
            variants.append("Full")
        elif name == "vector":
            variants.append("FlatRetrieval")
        elif name == "no_graph":
            variants.append("NoGraph")
        elif name == "no_pfc":
            variants.append("NoPFC")
    return run_neuromem_bench(variants=variants or None, output_format="json")


def _is_writable(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".doctor-write"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _sqlite_ok(path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute("SELECT 1").fetchone()
        return True
    except sqlite3.Error:
        return False


if __name__ == "__main__":
    main()
