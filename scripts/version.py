"""Compute the published image version: <major>.<minor>.<build>.

Base major.minor comes from version.json; the build number is passed by CI
(github.run_number). Mirrors uni-voices' versioning so tags stay consistent.
"""
import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="0")
    args = parser.parse_args()

    base = json.loads((Path(__file__).resolve().parent.parent / "version.json").read_text())["version"]
    major_minor = ".".join(base.split(".")[:2])  # "0.1.0" -> "0.1"
    print(f"{major_minor}.{args.build}")
