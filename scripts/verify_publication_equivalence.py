"""A5 gate — prove the cleaned publishable lineage differs only by the blob.

Runs the eight equivalence checks from ``P2_0_CORPUS_AUDIT.md`` §5 against the
rewritten history and reports PASS/FAIL for each. Every check is a proof about
object identity, not a sample: Git blob SHAs *are* content hashes, so an equal
blob SHA is byte-identity.

    python -m scripts.verify_publication_equivalence
    python -m scripts.verify_publication_equivalence --published publish/hf-spaces

Exit code is 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ORIGINAL = "a153f0c7162529d312267806729a2ab220252b3d"
PUBLISHED_REF = "publish/hf-spaces"
REMOTE_TIP = "e0184e08db24e9a8af7a61fd5eaa9639fbdcda7c"
REMOVED_PATH = "mao/data/bm25_corpus.json"
EXPECTED_COMMITS = 108
GITHUB_BLOB_LIMIT = 100 * 1024 * 1024


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, encoding="utf-8"
    )
    return result.stdout


def _tree(ref: str) -> dict[str, str]:
    """Map path -> blob sha for every file in a ref's tree."""
    entries: dict[str, str] = {}
    for line in git("ls-tree", "-r", ref).splitlines():
        if not line.strip():
            continue
        meta, path = line.split("\t", 1)
        _mode, obj_type, sha = meta.split()
        if obj_type == "blob":
            entries[path] = sha
    return entries


def _commit_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2 or parts[0] == "old":
            continue
        mapping[parts[0]] = parts[1]
    return mapping


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, number: str, name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {number}  {name}")
        if detail:
            for line in detail.splitlines():
                print(f"          {line}")
        if not passed:
            self.failures.append(f"{number} {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default=ORIGINAL)
    parser.add_argument("--published", default=PUBLISHED_REF)
    parser.add_argument("--remote-tip", default=REMOTE_TIP)
    parser.add_argument("--commit-map", default=r"E:\mao_phase1_archive\commit-map-original-to-published.txt")
    args = parser.parse_args(argv)

    report = Report()
    original, published = args.original, args.published

    # 1 — the only difference is the deletion of the blob
    diff = git("diff", "--name-status", original, published).strip()
    expected = f"D\t{REMOVED_PATH}"
    report.check(
        "5.1",
        "Stage-1 diff from the reviewed Phase 1 tree is exactly one deletion",
        diff == expected,
        f"expected: {expected!r}\nactual:   {diff!r}",
    )

    # 2 — every surviving blob is byte-identical
    before, after = _tree(original), _tree(published)
    survivors = {p: s for p, s in before.items() if p != REMOVED_PATH}
    mismatched = {p for p, s in survivors.items() if after.get(p) != s}
    extra = set(after) - set(survivors)
    report.check(
        "5.2",
        "Every surviving Phase 1 blob is byte-identical",
        not mismatched and not extra,
        f"{len(survivors)} files compared; {len(mismatched)} differing, {len(extra)} unexpected",
    )

    # 3 — commit count preserved across the unpublished range
    count = int(git("rev-list", "--count", f"{args.remote_tip}..{published}").strip())
    report.check(
        "5.3",
        "Commit count preserved across the unpublished range",
        count == EXPECTED_COMMITS,
        f"expected {EXPECTED_COMMITS}, got {count}",
    )

    # 4 — mapping completeness: every original commit maps, none to null
    map_path = Path(args.commit_map)
    originals = git("rev-list", f"{args.remote_tip}..{original}").split()
    if map_path.is_file():
        mapping = _commit_map(map_path)
        missing = [c for c in originals if c not in mapping]
        dropped = [c for c in originals if mapping.get(c, "").strip("0") == ""]
        report.check(
            "5.4",
            "Original -> published mapping is complete, no commit dropped",
            not missing and not dropped,
            f"{len(originals)} originals; {len(missing)} unmapped, {len(dropped)} mapped to null",
        )
    else:
        report.check("5.4", "Original -> published mapping present", False, f"not found: {map_path}")
        mapping = {}

    # 5 — topology preserved: translate each original's parents and compare
    if mapping:
        def parents(ref_range: str) -> dict[str, list[str]]:
            out: dict[str, list[str]] = {}
            for line in git("rev-list", "--parents", ref_range).splitlines():
                shas = line.split()
                out[shas[0]] = shas[1:]
            return out

        before_parents = parents(f"{args.remote_tip}..{original}")
        after_parents = parents(f"{args.remote_tip}..{published}")
        shape_ok = True
        detail: list[str] = []
        for commit, ps in before_parents.items():
            translated = mapping.get(commit, commit)
            actual = after_parents.get(translated)
            if actual is None:
                shape_ok = False
                detail.append(f"missing published commit for {commit[:8]}")
                continue
            wanted = [mapping.get(p, p) for p in ps]
            if wanted != actual:
                shape_ok = False
                detail.append(f"{commit[:8]}: parents {wanted} != {actual}")
        report.check(
            "5.5",
            "Topology preserved (parent edges identical under the mapping)",
            shape_ok,
            "\n".join(detail[:5]) or f"{len(before_parents)} commits, all parent edges match",
        )

    # 6 — no oversized object remains anywhere in the published range
    oversized: list[str] = []
    rev_list = git("rev-list", "--objects", f"{args.remote_tip}..{published}")
    proc = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objecttype) %(objectname) %(objectsize) %(rest)"],
        input=rev_list, capture_output=True, text=True, check=True, encoding="utf-8",
    )
    largest = 0
    for line in proc.stdout.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) < 3 or parts[0] != "blob":
            continue
        size = int(parts[2])
        largest = max(largest, size)
        if size > GITHUB_BLOB_LIMIT:
            oversized.append(f"{parts[1]} {size} {parts[3] if len(parts) > 3 else ''}")
    report.check(
        "5.6",
        "No object above GitHub's 100 MiB limit remains",
        not oversized,
        f"largest blob: {largest:,} bytes ({largest / 1024 / 1024:.2f} MiB)",
    )

    # 7 — the remote tip is still an ancestor, so publishing fast-forwards
    ff = subprocess.run(
        ["git", "merge-base", "--is-ancestor", args.remote_tip, published],
        capture_output=True, text=True,
    )
    report.check(
        "5.7",
        "Remote GitHub tip is an ancestor — publication is a fast-forward",
        ff.returncode == 0,
        f"{args.remote_tip[:8]} -> {published}",
    )

    print()
    if report.failures:
        print(f"FAILED: {len(report.failures)} check(s): {', '.join(report.failures)}")
        return 1
    print("All equivalence checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
