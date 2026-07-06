"""Dataset readers for text folders and labeled pair CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Iterable


@dataclass(frozen=True, slots=True)
class TextDocument:
    document_id: str
    path: Path
    text: str


@dataclass(frozen=True, slots=True)
class LabeledPair:
    pair_id: str
    text_a: str
    text_b: str
    label: int


def load_text_directory(
    directory: str | Path,
    *,
    pattern: str = "*.txt",
    encoding: str = "utf-8",
) -> list[TextDocument]:
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus directory not found: {root}")

    documents: list[TextDocument] = []
    for path in sorted(root.rglob(pattern)):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            raise ValueError(f"Could not decode {path} with {encoding!r}.") from exc
        documents.append(TextDocument(path.stem, path, text))

    if not documents:
        raise ValueError(f"No files matching {pattern!r} were found in {root}.")
    return documents


def _parse_label(raw: str) -> int:
    normalized = str(raw).strip().casefold()
    positive = {"1", "true", "yes", "duplicate", "positive", "plagiarized"}
    negative = {"0", "false", "no", "non-duplicate", "negative", "original"}
    if normalized in positive:
        return 1
    if normalized in negative:
        return 0
    try:
        value = int(float(normalized))
    except ValueError as exc:
        raise ValueError(f"Unsupported label value: {raw!r}") from exc
    if value not in {0, 1}:
        raise ValueError(f"Label must be binary, got {raw!r}.")
    return value


def load_labeled_pairs(
    csv_path: str | Path,
    *,
    text_col_a: str,
    text_col_b: str,
    label_col: str,
    id_col: str | None = None,
    limit: int | None = None,
    encoding: str = "utf-8-sig",
) -> list[LabeledPair]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pair dataset not found: {path}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided.")

    pairs: list[LabeledPair] = []
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("The pair CSV has no header row.")
        missing = {text_col_a, text_col_b, label_col} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for row_index, row in enumerate(reader, start=1):
            if limit is not None and len(pairs) >= limit:
                break
            text_a = row.get(text_col_a) or ""
            text_b = row.get(text_col_b) or ""
            pair_id = (row.get(id_col) if id_col else None) or str(row_index)
            pairs.append(
                LabeledPair(
                    pair_id=str(pair_id),
                    text_a=text_a,
                    text_b=text_b,
                    label=_parse_label(row.get(label_col, "")),
                )
            )

    if not pairs:
        raise ValueError("The pair dataset is empty after applying the limit.")
    return pairs


def write_csv(path: str | Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
