from __future__ import annotations

import json

from neuromem_runtime.cli import main


def test_cli_local_workflow(tmp_path, capsys) -> None:
    workspace = tmp_path / ".neuromem"
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "type": "task_result",
                "content": "Session refresh order fixed login redirect loop.",
                "task": "Fix login",
                "evidence": "trace-1",
                "keywords": ["session", "login"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    main(["--path", str(workspace), "init", "--namespace", "demo"])
    assert "memory.sqlite3" in capsys.readouterr().out

    main(["--path", str(workspace), "doctor", "--namespace", "demo"])
    assert '"ok": true' in capsys.readouterr().out

    main(["--path", str(workspace), "observe", str(events), "--namespace", "demo"])
    observed = json.loads(capsys.readouterr().out)
    assert observed["observed"] == 1
    assert observed["memory_ids"] == []

    main(["--path", str(workspace), "query", "login session", "--namespace", "demo", "--json"])
    empty_context = json.loads(capsys.readouterr().out)
    assert empty_context["selected_memory_ids"] == []

    main(["--path", str(workspace), "observe", str(events), "--namespace", "demo", "--commit"])
    committed = json.loads(capsys.readouterr().out)
    assert committed["memory_ids"]

    main(["--path", str(workspace), "query", "login session", "--namespace", "demo", "--json"])
    context = json.loads(capsys.readouterr().out)
    assert context["selected_memory_ids"]
    assert context["trace_id"]
    assert context["results"][0]["why_retrieved"]

    main(["--path", str(workspace), "retrieval", "explain", context["trace_id"], "--namespace", "demo"])
    retrieval = json.loads(capsys.readouterr().out)
    assert retrieval["selected_ids"] == context["selected_memory_ids"]
    assert retrieval["fusion_scores"]

    main(["--path", str(workspace), "sleep", "--namespace", "demo"])
    assert "processed" in json.loads(capsys.readouterr().out)

    main(["--path", str(workspace), "trace", "show", context["trace_id"], "--namespace", "demo"])
    trace = json.loads(capsys.readouterr().out)
    assert trace["trace_id"] == context["trace_id"]

    out = tmp_path / "trace.json"
    main(["--path", str(workspace), "trace", "export", context["trace_id"], "--namespace", "demo", "--out", str(out)])
    capsys.readouterr()
    assert out.exists()

    txn_id = context["transactions"][0]["transaction_id"]
    main(["--path", str(workspace), "ledger", "show", txn_id, "--namespace", "demo"])
    ledger_events = json.loads(capsys.readouterr().out)
    assert ledger_events

    main(["--path", str(workspace), "ledger", "why-retrieved", context["trace_id"], context["selected_memory_ids"][0], "--namespace", "demo"])
    assert json.loads(capsys.readouterr().out)

    main(["--path", str(workspace), "ledger", "replay", "--to-txn", txn_id, "--namespace", "demo"])
    assert json.loads(capsys.readouterr().out)
