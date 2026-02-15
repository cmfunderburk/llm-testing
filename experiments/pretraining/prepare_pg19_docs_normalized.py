"""
Normalize PG-19 document JSONL corpora while preserving document boundaries.

Input files (from download_corpora.py pg19_docs / pg19_docs_small):
    experiments/pretraining/corpus/pg19_{train,validation,test}_docs.jsonl
    experiments/pretraining/corpus/pg19_{train,validation,test}_small_docs.jsonl

Output files:
    experiments/pretraining/corpus/pg19_{split}_docs_normalized.jsonl
    experiments/pretraining/corpus/pg19_{split}_small_docs_normalized.jsonl

Each JSONL record remains one document with:
  {"doc_id": "...", "text": "..."}
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .prepare_pg19_normalized import NormalizeStats, _normalize_block

CORPUS_DIR = Path(__file__).parent / "corpus"
SPLITS: Sequence[str] = ("train", "validation", "test")


@dataclass
class DocStats:
    docs_total: int = 0
    docs_failed: int = 0
    input_chars: int = 0
    output_chars: int = 0
    normalize: NormalizeStats = field(default_factory=NormalizeStats)


def normalize_document_text(text: str) -> tuple[str, NormalizeStats]:
    """
    Normalize a single document string with block-aware prose unwrapping.

    Blocks are split by blank lines; each block is processed with the same
    heuristics used by prepare_pg19_normalized.py.
    """
    stats = NormalizeStats()

    nul_count = text.count("\x00")
    if nul_count:
        stats.nul_chars_removed += nul_count
        text = text.replace("\x00", "")

    stats.input_chars += len(text)

    out_blocks: list[str] = []
    block: list[str] = []

    def flush_block() -> None:
        if not block:
            return
        normalized, preserved = _normalize_block(block)
        stats.blocks_total += 1
        if preserved:
            stats.blocks_preserved += 1
        else:
            stats.blocks_unwrapped += 1
        if normalized:
            out_blocks.append(normalized)
        block.clear()

    for raw_line in text.splitlines():
        stats.lines_total += 1
        line = raw_line.rstrip()
        if line:
            block.append(line)
        else:
            flush_block()

    flush_block()
    normalized_text = "\n\n".join(out_blocks)
    stats.output_chars += len(normalized_text)
    return normalized_text, stats


def _dataset_paths(dataset: str) -> list[tuple[Path, Path]]:
    if dataset not in {"pg19_docs", "pg19_docs_small"}:
        raise ValueError(f"Unknown dataset: {dataset}")

    suffix = "_small" if dataset == "pg19_docs_small" else ""
    result: list[tuple[Path, Path]] = []
    for split in SPLITS:
        in_name = f"pg19_{split}{suffix}_docs.jsonl"
        out_name = f"pg19_{split}{suffix}_docs_normalized.jsonl"
        result.append((CORPUS_DIR / in_name, CORPUS_DIR / out_name))
    return result


def _merge_stats(total: NormalizeStats, inc: NormalizeStats) -> None:
    total.blocks_total += inc.blocks_total
    total.blocks_unwrapped += inc.blocks_unwrapped
    total.blocks_preserved += inc.blocks_preserved
    total.lines_total += inc.lines_total
    total.nul_chars_removed += inc.nul_chars_removed
    total.input_chars += inc.input_chars
    total.output_chars += inc.output_chars


def normalize_jsonl_file(
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
    progress_every_docs: int = 1000,
) -> DocStats:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path} (use --overwrite to replace it)"
        )
    if progress_every_docs <= 0:
        raise ValueError("progress_every_docs must be > 0")

    stats = DocStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_output_path.unlink(missing_ok=True)

    try:
        with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
            with open(temp_output_path, "w", encoding="utf-8") as fout:
                for i, raw_line in enumerate(fin, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        stats.docs_failed += 1
                        continue

                    text = payload.get("text")
                    if not isinstance(text, str):
                        stats.docs_failed += 1
                        continue

                    normalized_text, per_doc = normalize_document_text(text)
                    payload["text"] = normalized_text

                    fout.write(json.dumps(payload, ensure_ascii=False))
                    fout.write("\n")

                    stats.docs_total += 1
                    stats.input_chars += len(text)
                    stats.output_chars += len(normalized_text)
                    _merge_stats(stats.normalize, per_doc)

                    if i % progress_every_docs == 0:
                        print(
                            f"  processed {i:,} lines / {stats.docs_total:,} docs"
                            f" ({stats.normalize.blocks_total:,} blocks)...",
                            flush=True,
                        )
    except Exception:
        temp_output_path.unlink(missing_ok=True)
        raise

    temp_output_path.replace(output_path)

    return stats


def _print_summary(input_path: Path, output_path: Path, stats: DocStats) -> None:
    input_mb = input_path.stat().st_size / (1024 * 1024)
    output_mb = output_path.stat().st_size / (1024 * 1024)
    nstats = stats.normalize
    kept_ratio = (nstats.blocks_preserved / nstats.blocks_total * 100.0) if nstats.blocks_total else 0.0
    unwrap_ratio = (nstats.blocks_unwrapped / nstats.blocks_total * 100.0) if nstats.blocks_total else 0.0

    print(f"\n{input_path.name} -> {output_path.name}")
    print(f"  Size: {input_mb:.1f} MB -> {output_mb:.1f} MB")
    print(f"  Docs: {stats.docs_total:,} (failed: {stats.docs_failed:,})")
    print(f"  Blocks: {nstats.blocks_total:,}")
    print(f"    Preserved line breaks: {nstats.blocks_preserved:,} ({kept_ratio:.1f}%)")
    print(f"    Unwrapped prose:       {nstats.blocks_unwrapped:,} ({unwrap_ratio:.1f}%)")
    print(f"  Lines processed: {nstats.lines_total:,}")
    if nstats.nul_chars_removed:
        print(f"  NUL chars removed: {nstats.nul_chars_removed:,}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize PG-19 document JSONL while preserving doc boundaries."
    )
    parser.add_argument(
        "dataset",
        choices=["pg19_docs", "pg19_docs_small"],
        help="Which PG-19 docs variant to normalize",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing normalized output files",
    )
    parser.add_argument(
        "--progress-every-docs",
        type=int,
        default=1000,
        help="Print progress every N JSONL records (default: 1000)",
    )
    args = parser.parse_args(argv)
    if args.progress_every_docs <= 0:
        parser.error("--progress-every-docs must be > 0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    try:
        pairs = _dataset_paths(args.dataset)
    except ValueError as exc:
        print(str(exc))
        return 2

    for input_path, output_path in pairs:
        print(f"\nNormalizing {input_path} ...", flush=True)
        try:
            stats = normalize_jsonl_file(
                input_path=input_path,
                output_path=output_path,
                overwrite=args.overwrite,
                progress_every_docs=args.progress_every_docs,
            )
        except (FileNotFoundError, FileExistsError) as exc:
            print(f"Error: {exc}")
            return 1

        _print_summary(input_path, output_path, stats)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
