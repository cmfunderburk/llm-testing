"""
Prepare normalized PG-19 corpora that unwrap hard-wrapped prose lines.

PG-19 contains many Project Gutenberg texts with line breaks that reflect
historical formatting and fixed-width transcription artifacts. This script
normalizes those breaks for training a more natural prose model, while trying
to preserve verse-like formatting (poems, plays, lists, headings).

Input files (already downloaded):
    experiments/pretraining/corpus/pg19_{train,validation,test}.txt
    experiments/pretraining/corpus/pg19_{train,validation,test}_small.txt

Output files:
    experiments/pretraining/corpus/pg19_{split}_normalized.txt
    experiments/pretraining/corpus/pg19_{split}_small_normalized.txt

Usage:
    uv run python -m experiments.pretraining.prepare_pg19_normalized pg19_small
    uv run python -m experiments.pretraining.prepare_pg19_normalized pg19
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

CORPUS_DIR = Path(__file__).parent / "corpus"
SPLITS: Sequence[str] = ("train", "validation", "test")

LINE_END_PUNCT_RE = re.compile(r"""[.!?;:)"'\]\u201d\u2019]$""")
LIST_LINE_RE = re.compile(
    r"""^\s*(?:[-*]|\d+[.)]|[ivxlcdm]+[.)]|chapter\b|book\b|act\b|scene\b)""",
    flags=re.IGNORECASE,
)
STRUCTURED_MARKER_RE = re.compile(
    r"""(?:\.{3,}|_{3,}|\[\s*page\b|\bcontents\b|\billustrations\b)""",
    flags=re.IGNORECASE,
)


@dataclass
class NormalizeStats:
    blocks_total: int = 0
    blocks_unwrapped: int = 0
    blocks_preserved: int = 0
    lines_total: int = 0
    nul_chars_removed: int = 0
    input_chars: int = 0
    output_chars: int = 0


def _is_heading_like(line: str) -> bool:
    text = line.strip()
    if not text:
        return False

    if re.match(r"^(chapter|book|act|scene)\b", text, flags=re.IGNORECASE):
        return True

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False

    uppercase_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    if uppercase_ratio >= 0.9 and len(text) <= 70:
        return True

    # Short title-like lines are often section headers, not prose wraps.
    if len(text) <= 28 and text == text.title() and not text.endswith((".", "!", "?")):
        return True

    return False


def _should_preserve_lines(raw_lines: Sequence[str]) -> bool:
    """
    Decide whether to preserve line breaks for this block.

    Heuristic:
    - Preserve clearly structured/verse blocks.
    - Unwrap likely hard-wrapped prose blocks.
    """
    if len(raw_lines) <= 1:
        return True

    stripped = [line.strip() for line in raw_lines if line.strip()]
    if len(stripped) <= 1:
        return True

    if any("\t" in line for line in raw_lines):
        return True
    if any(STRUCTURED_MARKER_RE.search(line) for line in stripped):
        return True

    n = len(stripped)
    lengths = [len(line) for line in stripped]
    avg_len = sum(lengths) / n

    short_ratio = sum(length <= 55 for length in lengths) / n
    very_short_ratio = sum(length <= 35 for length in lengths) / n
    heading_ratio = sum(_is_heading_like(line) for line in stripped) / n
    list_ratio = sum(bool(LIST_LINE_RE.match(line)) for line in stripped) / n

    indents = [len(line) - len(line.lstrip(" ")) for line in raw_lines]
    indented_ratio = sum(indent >= 2 for indent in indents) / len(indents)

    nonfinal = stripped[:-1]
    if nonfinal:
        nonfinal_long_ratio = sum(len(line) >= 60 for line in nonfinal) / len(nonfinal)
        nonfinal_terminal_ratio = (
            sum(bool(LINE_END_PUNCT_RE.search(line)) for line in nonfinal) / len(nonfinal)
        )
    else:
        nonfinal_long_ratio = 0.0
        nonfinal_terminal_ratio = 0.0

    prose_wrap_like = n >= 2 and nonfinal_long_ratio >= 0.60 and avg_len >= 45
    poetry_like = (
        n >= 3
        and short_ratio >= 0.55
        and very_short_ratio >= 0.20
        and nonfinal_long_ratio <= 0.40
        and nonfinal_terminal_ratio <= 0.70
    )

    if list_ratio >= 0.30:
        return True
    if heading_ratio >= 0.50 and avg_len <= 45:
        return True
    if indented_ratio >= 0.40 and nonfinal_long_ratio <= 0.50:
        return True
    if poetry_like and not prose_wrap_like:
        return True

    return False


def _unwrap_prose_lines(raw_lines: Sequence[str]) -> str:
    parts: List[str] = []
    for raw in raw_lines:
        line = re.sub(r"\s+", " ", raw.strip())
        if not line:
            continue
        if not parts:
            parts.append(line)
            continue

        prev = parts[-1]

        # Join split words from hard line-wrap hyphenation.
        if prev.endswith("-") and line[0].islower():
            parts[-1] = prev[:-1] + line
        elif prev.endswith(("—", "–", "/")):
            parts[-1] = prev + line
        else:
            parts[-1] = prev + " " + line

    return parts[0] if parts else ""


def _normalize_block(raw_lines: Sequence[str]) -> Tuple[str, bool]:
    preserve = _should_preserve_lines(raw_lines)
    if preserve:
        normalized = "\n".join(line.rstrip() for line in raw_lines)
        return normalized, True

    normalized = _unwrap_prose_lines(raw_lines)
    return normalized, False


def normalize_file(
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
    progress_every_lines: int = 1_000_000,
) -> NormalizeStats:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path} (use --overwrite to replace it)"
        )

    stats = NormalizeStats()
    block: List[str] = []
    first_block = True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
        with open(output_path, "w", encoding="utf-8") as fout:
            for raw_line in fin:
                stats.lines_total += 1
                stats.input_chars += len(raw_line)
                if stats.lines_total % progress_every_lines == 0:
                    print(
                        f"  processed {stats.lines_total:,} lines ({stats.blocks_total:,} blocks)...",
                        flush=True,
                    )

                nul_count = raw_line.count("\x00")
                if nul_count:
                    stats.nul_chars_removed += nul_count
                    raw_line = raw_line.replace("\x00", "")

                line = raw_line.rstrip("\n\r").rstrip()

                if line:
                    block.append(line)
                    continue

                if not block:
                    continue

                normalized, preserved = _normalize_block(block)
                if normalized:
                    if not first_block:
                        fout.write("\n\n")
                        stats.output_chars += 2
                    fout.write(normalized)
                    stats.output_chars += len(normalized)
                    first_block = False

                stats.blocks_total += 1
                if preserved:
                    stats.blocks_preserved += 1
                else:
                    stats.blocks_unwrapped += 1
                block = []

            # Flush final block.
            if block:
                normalized, preserved = _normalize_block(block)
                if normalized:
                    if not first_block:
                        fout.write("\n\n")
                        stats.output_chars += 2
                    fout.write(normalized)
                    stats.output_chars += len(normalized)

                stats.blocks_total += 1
                if preserved:
                    stats.blocks_preserved += 1
                else:
                    stats.blocks_unwrapped += 1

    return stats


def _dataset_paths(dataset: str) -> List[Tuple[Path, Path]]:
    if dataset not in {"pg19", "pg19_small"}:
        raise ValueError(f"Unknown dataset: {dataset}")

    suffix = "_small" if dataset == "pg19_small" else ""
    result: List[Tuple[Path, Path]] = []
    for split in SPLITS:
        in_name = f"pg19_{split}{suffix}.txt"
        out_name = f"pg19_{split}{suffix}_normalized.txt"
        result.append((CORPUS_DIR / in_name, CORPUS_DIR / out_name))
    return result


def _print_summary(input_path: Path, output_path: Path, stats: NormalizeStats):
    input_mb = input_path.stat().st_size / (1024 * 1024)
    output_mb = output_path.stat().st_size / (1024 * 1024)
    kept_ratio = (stats.blocks_preserved / stats.blocks_total * 100.0) if stats.blocks_total else 0.0
    unwrap_ratio = (stats.blocks_unwrapped / stats.blocks_total * 100.0) if stats.blocks_total else 0.0

    print(f"\n{input_path.name} -> {output_path.name}")
    print(f"  Size: {input_mb:.1f} MB -> {output_mb:.1f} MB")
    print(f"  Blocks: {stats.blocks_total:,}")
    print(f"    Preserved line breaks: {stats.blocks_preserved:,} ({kept_ratio:.1f}%)")
    print(f"    Unwrapped prose:       {stats.blocks_unwrapped:,} ({unwrap_ratio:.1f}%)")
    print(f"  Lines processed: {stats.lines_total:,}")
    if stats.nul_chars_removed:
        print(f"  NUL chars removed: {stats.nul_chars_removed:,}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize PG-19 line breaks while preserving likely verse blocks."
    )
    parser.add_argument(
        "dataset",
        choices=["pg19", "pg19_small"],
        help="Which PG-19 variant to normalize",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing normalized output files",
    )
    parser.add_argument(
        "--progress-every-lines",
        type=int,
        default=1_000_000,
        help="Print progress every N input lines (default: 1,000,000)",
    )
    return parser.parse_args(argv)


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
            stats = normalize_file(
                input_path=input_path,
                output_path=output_path,
                overwrite=args.overwrite,
                progress_every_lines=args.progress_every_lines,
            )
        except (FileNotFoundError, FileExistsError) as exc:
            print(f"Error: {exc}")
            return 1

        _print_summary(input_path, output_path, stats)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
