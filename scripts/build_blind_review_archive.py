"""Export source plus sanitized results without Git history for double-blind review.

This creates a review copy. Keep the original analysis ZIPs unchanged offline.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

from reproduce_paper_analysis import analyze


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "tests", "config", "docs", "scripts")
SOURCE_FILES = ("README.md", "pyproject.toml")
HISTORY_PATTERN = re.compile(r"(?i)([\w.%-]+@[\w.-]+\.(?:com|edu|org|net|io)|github\.com/[^/\s]+/gfm-auditor)")


def _add(archive: ZipFile, name: str, payload: bytes) -> None:
    from zipfile import ZipInfo

    member = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    member.compress_type = ZIP_DEFLATED
    member.external_attr = 0o644 << 16
    archive.writestr(member, payload)


def _scrub_result(value: dict) -> dict:
    value = dict(value)
    value["run_identity"] = {
        k: v for k, v in value["run_identity"].items()
        if k not in ("commit", "config_sha256")
    }
    value["summary"] = {k: v for k, v in value["summary"].items() if k != "commit"}
    value["checkpoints"] = [
        {k: v for k, v in row.items() if k != "commit"}
        for row in value["checkpoints"]
    ]
    return value


def _sanitize_bundle(path: Path) -> bytes:
    output = io.BytesIO()
    with ZipFile(path) as original, ZipFile(output, "w") as clean:
        names = sorted(n for n in original.namelist() if n.startswith("results/") and n.endswith(".json"))
        if len(names) != 750:
            raise ValueError(f"{path}: missing trajectories")
        for name in names:
            row = _scrub_result(json.loads(original.read(name)))
            _add(clean, name, (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--20m", required=True, type=Path)
    parser.add_argument("--116m", required=True, type=Path)
    parser.add_argument("--nucel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bundles = {k: getattr(args, k) for k in ("20m", "116m", "nucel")}
    originals = {k: analyze(p, k) for k, p in bundles.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(args.output, "w") as release:
        for name in SOURCE_FILES:
            payload = (ROOT / name).read_bytes()
            if HISTORY_PATTERN.search(payload.decode("utf-8")):
                raise ValueError(f"author-identifying source: {name}")
            _add(release, name, payload)
        for directory in SOURCE_DIRS:
            for file in sorted((ROOT / directory).rglob("*")):
                if not file.is_file() or file.suffix not in (".py", ".md", ".yaml", ".toml"):
                    continue
                relative = file.relative_to(ROOT).as_posix()
                content = file.read_text(encoding="utf-8")
                if relative == "config/locked_primary_omnidna20m_k2.yaml":
                    content = re.sub(r"^base_repository_commit: .*\n", "base_repository_commit: redacted_for_review\n", content, flags=re.M)
                if HISTORY_PATTERN.search(content):
                    raise ValueError(f"author-identifying source: {relative}")
                _add(release, relative, content.encode())
        for key, path in bundles.items():
            sanitized = _sanitize_bundle(path)
            _add(release, f"artifacts/{key}_analysis_bundle.blind.zip", sanitized)
        _add(release, "paper_analysis.json", (json.dumps(originals, indent=2, sort_keys=True) + "\n").encode())
        _add(release, "ANONYMIZATION.md", (
            "# Review export\n\nThis ZIP intentionally omits .git and notebook outputs. "
            "Original commit fields and the 20M config hash were removed from "
            "the result JSONs; the base-repository commit in the frozen config "
            "was redacted. All quantitative fields, gene IDs, input hashes, "
            "and model revisions are unchanged. Keep the unredacted originals "
            "offline. Full model inference also requires the frozen source CSV, "
            "control manifest, and metadata described in the guide.\n"
        ).encode())
    # Check the exact release bytes, not merely the originals.
    with ZipFile(args.output) as release:
        for key in bundles:
            extracted = args.output.parent / f".blind_check_{key}.zip"
            try:
                extracted.write_bytes(release.read(f"artifacts/{key}_analysis_bundle.blind.zip"))
                if analyze(extracted, key) != originals[key]:
                    raise ValueError(f"redaction changed the {key} quantitative analysis")
            finally:
                extracted.unlink(missing_ok=True)
    print(f"Validated blind review source and analysis: {args.output}")


if __name__ == "__main__":
    main()
