"""
Prepare Wikipedia Good Article+ introductions corpus for pretraining.

Reads the JSONL from the reader project (read-only), cleans it, and writes
a plain-text corpus file into this repo's corpus directory.

Source: /home/cmf/Dropbox/Apps/reader/scripts/prepare-corpus/wikipedia-ga.jsonl
Output: experiments/pretraining/corpus/wikipedia-ga-intros.txt

Cleaning steps:
- Filters out passages with LaTeX math markup (\displaystyle, \frac, etc.)
- Filters out passages with wiki markup ({{ }}, [[ ]])
- Filters out passages with raw URLs
- Filters out very short passages (<20 chars)
- Normalizes whitespace
- Reconstructs article introductions by joining passages per article
- Separates articles with double newlines
"""

import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

SOURCE_PATH = Path("/home/cmf/Dropbox/Apps/reader/scripts/prepare-corpus/wikipedia-ga.jsonl")
OUTPUT_PATH = Path(__file__).parent / "corpus" / "wikipedia-ga-intros.txt"

# Patterns that indicate non-prose content
ARTIFACT_PATTERNS = [
    re.compile(r'\\displaystyle'),       # LaTeX math
    re.compile(r'\\frac\b'),             # LaTeX fractions
    re.compile(r'\\mathrm\b'),           # LaTeX formatting
    re.compile(r'\\text\b'),             # LaTeX text
    re.compile(r'\\cdot'),               # LaTeX dot operator
    re.compile(r'\[\[|\]\]'),            # Wiki internal links
    re.compile(r'\{\{|\}\}'),            # Wiki templates
    re.compile(r'https?://'),            # URLs
]


def has_artifacts(text: str) -> bool:
    return any(p.search(text) for p in ARTIFACT_PATTERNS)


def clean_passage(text: str) -> str:
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def main():
    if not SOURCE_PATH.exists():
        print(f"Source not found: {SOURCE_PATH}")
        sys.exit(1)

    # Group passages by article, preserving insertion order
    articles = OrderedDict()
    total_passages = 0
    skipped_passages = 0

    with open(SOURCE_PATH) as f:
        for line in f:
            obj = json.loads(line)
            total_passages += 1
            text = obj["text"]
            source = obj["source"]

            if has_artifacts(text):
                skipped_passages += 1
                continue

            cleaned = clean_passage(text)
            if len(cleaned) < 20:
                skipped_passages += 1
                continue

            if source not in articles:
                articles[source] = []
            articles[source].append(cleaned)

    # Write clean corpus (read-only from source, write only to this repo)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    articles_written = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for i, (source, passages) in enumerate(articles.items()):
            article_text = " ".join(passages)
            total_chars += len(article_text)
            articles_written += 1

            out.write(article_text)
            if i < len(articles) - 1:
                out.write("\n\n")

    print(f"Source passages:  {total_passages:,}")
    print(f"Skipped:          {skipped_passages:,} ({skipped_passages / total_passages * 100:.1f}%)")
    print(f"Articles written: {articles_written:,}")
    print(f"Total chars:      {total_chars:,} ({total_chars / 1024 / 1024:.1f} MB)")
    print(f"Output:           {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
