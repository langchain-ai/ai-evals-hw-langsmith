"""Upload pre-generated traces to LangSmith.

Reads traces.json, shifts timestamps to the recent past, regenerates UUIDs,
and uploads each trace to the specified LangSmith project.

Usage:
    python upload_traces.py                          # default project name
    python upload_traces.py --project my-project     # custom project name
    python upload_traces.py --input traces.json      # custom input file
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client, uuid7


def parse_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def upload(project: str = "recipe-agent-hw2", input_path: str = "traces.json"):
    """Upload traces from a JSON file to LangSmith."""
    input_file = Path(__file__).parent / input_path
    with open(input_file) as f:
        runs = json.load(f)

    print(f"Loaded {len(runs)} runs from {input_file}")

    # Shift timestamps so traces appear recent
    latest = max(parse_dt(r["start_time"]) for r in runs if r["start_time"])
    time_delta = datetime.now(timezone.utc).replace(tzinfo=None) - latest
    print(f"Shifting timestamps by: {time_delta}")

    # Build ID map — root runs get trace_id == id
    id_map = {}
    for run in runs:
        if run.get("parent_run_id") is None:
            root_new_id = str(uuid7())
            id_map[run["id"]] = root_new_id
            id_map[run["trace_id"]] = root_new_id
    for run in runs:
        for field in ("id", "parent_run_id"):
            old_id = run.get(field)
            if old_id and old_id not in id_map:
                id_map[old_id] = str(uuid7())

    # Group runs by trace and transform
    traces = defaultdict(list)
    for run in runs:
        trace_id = id_map[run["trace_id"]]
        extra = dict(run.get("extra") or {})

        # Strip _feedback from metadata — feedback is created separately
        metadata = dict(extra.get("metadata") or {})
        metadata.pop("_feedback", None)
        if metadata:
            extra["metadata"] = metadata
        elif "metadata" in extra:
            del extra["metadata"]

        traces[trace_id].append({
            "id": id_map[run["id"]],
            "trace_id": trace_id,
            "dotted_order": None,
            "parent_run_id": id_map.get(run["parent_run_id"]),
            "name": run["name"],
            "run_type": run["run_type"],
            "inputs": run["inputs"],
            "outputs": run.get("outputs"),
            "error": run.get("error"),
            "extra": extra,
            "tags": run.get("tags"),
            "start_time": parse_dt(run["start_time"]) + time_delta,
            "end_time": parse_dt(run["end_time"]) + time_delta if run.get("end_time") else None,
        })

    client = Client()
    print(f"Uploading {len(traces)} traces to project '{project}'...")

    for i, (trace_id, trace_runs) in enumerate(traces.items()):
        # Sort: root first, then children by start_time
        trace_runs.sort(key=lambda r: (r["parent_run_id"] is not None, r["start_time"]))

        # Build dotted_order for proper nesting
        dotted_orders = {}
        for run in trace_runs:
            ts = run["start_time"].strftime("%Y%m%dT%H%M%S%f") + "Z"
            if run["parent_run_id"] is None:
                run["dotted_order"] = f"{ts}{run['id']}"
            else:
                parent_order = dotted_orders.get(run["parent_run_id"], "")
                run["dotted_order"] = f"{parent_order}.{ts}{run['id']}"
            dotted_orders[run["id"]] = run["dotted_order"]

        # Upload each run
        for run in trace_runs:
            client.create_run(
                id=run["id"],
                trace_id=run["trace_id"],
                dotted_order=run["dotted_order"],
                parent_run_id=run["parent_run_id"],
                name=run["name"],
                run_type=run["run_type"],
                inputs=run["inputs"],
                outputs=run.get("outputs"),
                error=run.get("error"),
                extra=run.get("extra"),
                tags=run.get("tags"),
                start_time=run["start_time"],
                end_time=run["end_time"],
                project_name=project,
            )

        if (i + 1) % 50 == 0:
            print(f"  Uploaded {i + 1}/{len(traces)} traces")

    print("Flushing traces...")
    client.flush()
    print(f"Done! Uploaded {len(traces)} traces to '{project}'.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Upload traces to LangSmith")
    parser.add_argument("--project", default="recipe-agent-hw2", help="Target project name")
    parser.add_argument("--input", default="traces.json", help="Input file path")
    args = parser.parse_args()

    upload(project=args.project, input_path=args.input)


if __name__ == "__main__":
    main()
