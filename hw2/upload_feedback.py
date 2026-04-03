"""Upload feedback annotations to LangSmith traces.

Reads _feedback metadata from root runs in traces.json, fetches the
corresponding runs from LangSmith (matched by chronological order),
and creates feedback annotations via the API.

Run this AFTER upload_traces.py so the runs are fully ingested.

Usage:
    python upload_feedback.py
    python upload_feedback.py --project my-project
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client


def upload_feedback(project: str = "recipe-bot-hw2", input_path: str = "traces.json"):
    """Create feedback annotations by matching traces in chronological order."""
    input_file = Path(__file__).parent / input_path
    with open(input_file) as f:
        runs = json.load(f)

    # Extract feedback from root runs in file order
    feedback_entries = []
    for r in runs:
        if r.get("parent_run_id") is not None:
            continue
        fb = (r.get("extra") or {}).get("metadata", {}).get("_feedback")
        feedback_entries.append(fb)  # None for traces without feedback

    print(f"Found {sum(1 for f in feedback_entries if f)} feedback entries in {len(feedback_entries)} traces")

    # Fetch root runs from LangSmith sorted by start time (same order as upload)
    client = Client()
    ls_runs = list(client.list_runs(project_name=project, is_root=True))
    ls_runs.sort(key=lambda r: r.start_time, reverse=True)
    ls_runs = ls_runs[: len(feedback_entries)]
    ls_runs.reverse()  # back to chronological order for zip

    feedback_count = 0
    for ls_run, fb in zip(ls_runs, feedback_entries):
        if fb is None:
            continue
        client.create_feedback(
            run_id=ls_run.id,
            key=fb["key"],
            comment=fb["comment"],
        )
        feedback_count += 1

    print("Flushing feedback...")
    client.flush()
    print(f"Done! Created {feedback_count} feedback annotations.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Upload feedback to LangSmith traces")
    parser.add_argument("--project", default="recipe-bot-hw2", help="Target project name")
    parser.add_argument("--input", default="traces.json", help="Input file path")
    args = parser.parse_args()

    upload_feedback(project=args.project, input_path=args.input)


if __name__ == "__main__":
    main()
