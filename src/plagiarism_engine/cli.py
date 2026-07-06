"""Command-line interface for the plagiarism detection project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

from .bonus import HybridTfidfSimHasher
from .dataset import load_labeled_pairs, load_text_directory, write_csv
from .evaluation import evaluate_pairs
from .ensemble import CalibratedEnsemble, calibrate_ensemble
from .lsh import LSHIndex
from .minhash import MinHasher, exact_jaccard
from .preprocessing import TextPreprocessor
from .simhash import TfidfSimHasher, hamming_distance, simhash_similarity


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_components(args: argparse.Namespace) -> tuple[TextPreprocessor, MinHasher]:
    preprocessor = TextPreprocessor(
        shingle_size=args.shingle_size,
        language=args.language,
        remove_stopwords=not args.keep_stopwords,
        adaptive_short_text=args.adaptive_short_text,
        short_text_token_limit=args.short_text_token_limit,
    )
    minhasher = MinHasher(num_perm=args.num_perm, seed=args.seed)
    return preprocessor, minhasher


def _make_simhasher(args: argparse.Namespace) -> TfidfSimHasher:
    if args.simhash_mode == "hybrid":
        return HybridTfidfSimHasher(
            ngram_size=args.simhash_ngram_size,
            char_ngram_size=3,
            use_char_ngrams=True,
        )
    return TfidfSimHasher(ngram_size=args.simhash_ngram_size)


def command_compare(args: argparse.Namespace) -> int:
    preprocessor, minhasher = _make_components(args)
    left = preprocessor.preprocess_file(args.file_a)
    right = preprocessor.preprocess_file(args.file_b)

    signature_a = minhasher.signature(left.shingles)
    signature_b = minhasher.signature(right.shingles)
    lsh = LSHIndex(num_perm=args.num_perm, bands=args.bands)
    candidate = bool(left.shingles and right.shingles) and lsh.is_candidate(
        signature_a, signature_b
    )
    jaccard = exact_jaccard(left.shingles, right.shingles)
    minhash_similarity = minhasher.similarity(signature_a, signature_b)

    simhasher = _make_simhasher(args)
    # Fit SimHash TF-IDF on background corpus if available to ensure correct weights
    ref_documents = [left.tokens, right.tokens]
    try:
        sample_corpus_dir = Path("data/sample_corpus")
        if sample_corpus_dir.is_dir():
            sample_docs = load_text_directory(sample_corpus_dir)
            ref_documents.extend(preprocessor.tokenize(doc.text) for doc in sample_docs)
    except Exception:
        pass
    simhasher.fit(ref_documents)
    sim_a = simhasher.fingerprint(left.tokens)
    sim_b = simhasher.fingerprint(right.tokens)
    distance = hamming_distance(sim_a, sim_b)

    payload: dict[str, object] = {
        "file_a": str(Path(args.file_a)),
        "file_b": str(Path(args.file_b)),
        "parameters": {
            "shingle_size": args.shingle_size,
            "num_perm": args.num_perm,
            "bands": args.bands,
            "jaccard_threshold": args.jaccard_threshold,
            "simhash_max_distance": args.simhash_max_distance,
            "simhash_ngram_size": args.simhash_ngram_size,
            "simhash_mode": args.simhash_mode,
            "adaptive_short_text": args.adaptive_short_text,
        },
        "preprocessing": {
            "tokens_a": len(left.tokens),
            "tokens_b": len(right.tokens),
            "shingles_a": len(left.shingles),
            "shingles_b": len(right.shingles),
            "language_a": left.language,
            "language_b": right.language,
        },
        "minhash_lsh": {
            "lsh_candidate": candidate,
            "exact_jaccard": round(jaccard, 6),
            "minhash_similarity": round(minhash_similarity, 6),
            "is_similar": bool(candidate and jaccard >= args.jaccard_threshold),
        },
        "simhash": {
            "fingerprint_a_hex": f"{sim_a:016x}",
            "fingerprint_b_hex": f"{sim_b:016x}",
            "hamming_distance": distance,
            "similarity": round(simhash_similarity(sim_a, sim_b), 6),
            "is_similar": bool(left.tokens and right.tokens) and distance <= args.simhash_max_distance,
        },
    }

    if args.ensemble_model:
        ensemble = CalibratedEnsemble.load(args.ensemble_model)
        model_preprocessor = ensemble.make_preprocessor()
        model_left = model_preprocessor.preprocess_file(args.file_a)
        model_right = model_preprocessor.preprocess_file(args.file_b)
        ensemble_score = ensemble.score_documents(model_left, model_right)
        payload["calibrated_ensemble"] = {
            "model": str(Path(args.ensemble_model)),
            "score": round(ensemble_score, 6),
            "threshold": round(ensemble.classifier.threshold, 6),
            "is_similar": ensemble_score >= ensemble.classifier.threshold,
        }

    _write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def command_corpus(args: argparse.Namespace) -> int:
    preprocessor, minhasher = _make_components(args)
    documents = load_text_directory(args.data)
    processed = [preprocessor.preprocess(doc.text) for doc in documents]
    signatures = [minhasher.signature(doc.shingles) for doc in processed]

    start = perf_counter()
    index = LSHIndex(num_perm=args.num_perm, bands=args.bands)
    for document, processed_document, signature in zip(
        documents,
        processed,
        signatures,
    ):
        if processed_document.shingles:
            index.insert(document.document_id, signature)
    candidate_pairs = index.candidate_pairs()

    simhasher = _make_simhasher(args)
    simhasher.fit([doc.tokens for doc in processed])
    fingerprints = [simhasher.fingerprint(doc.tokens) for doc in processed]
    by_id = {doc.document_id: position for position, doc in enumerate(documents)}

    ensemble = None
    ensemble_processed = None
    if args.ensemble_model:
        ensemble = CalibratedEnsemble.load(args.ensemble_model)
        ensemble_preprocessor = ensemble.make_preprocessor()
        ensemble_processed = [
            ensemble_preprocessor.preprocess(document.text)
            for document in documents
        ]

    rows: list[dict[str, object]] = []
    for doc_a, doc_b in sorted(candidate_pairs, key=lambda pair: (str(pair[0]), str(pair[1]))):
        index_a = by_id[str(doc_a)]
        index_b = by_id[str(doc_b)]
        exact = exact_jaccard(processed[index_a].shingles, processed[index_b].shingles)
        estimate = minhasher.similarity(signatures[index_a], signatures[index_b])
        distance = hamming_distance(fingerprints[index_a], fingerprints[index_b])
        row: dict[str, object] = {
            "document_a": doc_a,
            "document_b": doc_b,
            "exact_jaccard": round(exact, 6),
            "minhash_similarity": round(estimate, 6),
            "simhash_hamming": distance,
            "simhash_similarity": round(simhash_similarity(fingerprints[index_a], fingerprints[index_b]), 6),
            "passes_jaccard_threshold": int(exact >= args.jaccard_threshold),
            "passes_simhash_threshold": int(
                bool(processed[index_a].tokens and processed[index_b].tokens)
                and distance <= args.simhash_max_distance
            ),
            "ensemble_score": "",
            "passes_ensemble_threshold": "",
        }
        if ensemble is not None and ensemble_processed is not None:
            ensemble_score = ensemble.score_documents(
                ensemble_processed[index_a],
                ensemble_processed[index_b],
            )
            row["ensemble_score"] = round(ensemble_score, 6)
            row["passes_ensemble_threshold"] = int(
                ensemble_score >= ensemble.classifier.threshold
            )
        rows.append(row)

    all_pairs = len(documents) * (len(documents) - 1) // 2
    elapsed = perf_counter() - start
    fieldnames = [
        "document_a", "document_b", "exact_jaccard", "minhash_similarity",
        "simhash_hamming", "simhash_similarity", "passes_jaccard_threshold",
        "passes_simhash_threshold", "ensemble_score",
        "passes_ensemble_threshold",
    ]
    write_csv(args.output, rows, fieldnames)
    summary = {
        "documents": len(documents),
        "all_possible_pairs": all_pairs,
        "lsh_candidate_pairs": len(candidate_pairs),
        "reduction_ratio": round(1.0 - len(candidate_pairs) / all_pairs, 6) if all_pairs else 0.0,
        "accepted_by_jaccard": sum(int(row["passes_jaccard_threshold"]) for row in rows),
        "accepted_by_ensemble": (
            sum(int(row["passes_ensemble_threshold"]) for row in rows)
            if ensemble is not None
            else None
        ),
        "runtime_seconds": round(elapsed, 6),
        "output": str(args.output),
        "lsh_approximate_threshold": round(index.approximate_threshold, 6),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_pairs(args: argparse.Namespace) -> int:
    pairs = load_labeled_pairs(
        args.pairs,
        text_col_a=args.text_col_a,
        text_col_b=args.text_col_b,
        label_col=args.label_col,
        id_col=args.id_col,
        limit=args.limit,
    )
    preprocessor, minhasher = _make_components(args)
    metric_rows, prediction_rows = evaluate_pairs(
        pairs,
        preprocessor=preprocessor,
        minhasher=minhasher,
        bands=args.bands,
        jaccard_threshold=args.jaccard_threshold,
        simhash_max_distance=args.simhash_max_distance,
        simhash_ngram_size=args.simhash_ngram_size,
        simhash_mode=args.simhash_mode,
        cosine_threshold=args.cosine_threshold,
    )
    metric_fields = [
        "method",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "tp",
        "fp",
        "tn",
        "fn",
        "runtime_seconds",
        "pairs_evaluated",
        "candidate_pairs",
        "candidate_recall",
        "comparison_reduction",
        "jaccard_threshold",
        "simhash_max_distance",
        "cosine_threshold",
        "num_perm",
        "bands",
    ]
    write_csv(args.output, metric_rows, metric_fields)

    predictions_output = args.predictions_output
    if predictions_output:
        prediction_fields = [
            "pair_id",
            "label",
            "lsh_candidate",
            "jaccard",
            "minhash_similarity",
            "lsh_prediction",
            "simhash_hamming",
            "simhash_similarity",
            "simhash_prediction",
            "cosine_similarity",
            "cosine_prediction",
            "error_lsh",
            "error_simhash",
            "error_cosine",
            "text_a",
            "text_b",
        ]
        write_csv(predictions_output, prediction_rows, prediction_fields)

    print(json.dumps({"metrics": metric_rows, "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


def command_calibrate(args: argparse.Namespace) -> int:
    """Train and evaluate the calibrated lexical ensemble on held-out data."""

    pairs = load_labeled_pairs(
        args.pairs,
        text_col_a=args.text_col_a,
        text_col_b=args.text_col_b,
        label_col=args.label_col,
        id_col=args.id_col,
        limit=args.limit,
    )
    preprocessor, _ = _make_components(args)
    model, metric_rows, prediction_rows, split_counts = calibrate_ensemble(
        pairs,
        preprocessor=preprocessor,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        l2=args.ensemble_l2,
        max_iter=args.ensemble_max_iter,
    )
    model.save(args.model_output)

    metric_fields = [
        "method", "split", "precision", "recall", "f1", "accuracy",
        "tp", "fp", "tn", "fn", "threshold", "train_pairs",
        "validation_pairs", "test_pairs", "feature_count", "l2",
    ]
    write_csv(args.output, metric_rows, metric_fields)

    if args.predictions_output:
        prediction_fields = [
            "pair_id", "label", "ensemble_score", "ensemble_prediction",
            "error_ensemble", "word_tfidf_cosine", "char_tfidf_cosine",
            "token_jaccard", "overlap_coefficient", "sequence_ratio",
            "text_a", "text_b",
        ]
        write_csv(args.predictions_output, prediction_rows, prediction_fields)

    test_row = next(
        row for row in metric_rows
        if row["method"] == "Calibrated lexical ensemble" and row["split"] == "test"
    )
    print(json.dumps({
        "model_output": str(args.model_output),
        "metrics_output": str(args.output),
        "predictions_output": str(args.predictions_output) if args.predictions_output else None,
        "split": split_counts,
        "held_out_test_metrics": test_row,
    }, ensure_ascii=False, indent=2))
    return 0



def command_bonus_eval(args: argparse.Namespace) -> int:
    from .bonus import find_adaptive_lsh_params, HybridTfidfSimHasher, persian_lemmatize
    from .evaluation import classification_metrics

    pairs = load_labeled_pairs(
        args.pairs,
        text_col_a=args.text_col_a,
        text_col_b=args.text_col_b,
        label_col=args.label_col,
        id_col=args.id_col,
        limit=args.limit,
    )

    # 1. Solve Adaptive LSH Parameters
    best_b, best_r = find_adaptive_lsh_params(args.num_perm, args.jaccard_threshold)

    # 2. Setup standard components
    preprocessor = TextPreprocessor(
        shingle_size=args.shingle_size,
        language=args.language,
        remove_stopwords=not args.keep_stopwords,
        adaptive_short_text=args.adaptive_short_text,
        short_text_token_limit=args.short_text_token_limit,
    )
    minhasher = MinHasher(num_perm=args.num_perm, seed=args.seed)

    processed_a = [preprocessor.preprocess(pair.text_a) for pair in pairs]
    processed_b = [preprocessor.preprocess(pair.text_b) for pair in pairs]
    labels = [pair.label for pair in pairs]

    # Evaluate Static MinHash+LSH
    static_lsh = LSHIndex(num_perm=args.num_perm, bands=args.bands)
    static_preds = []
    for left, right in zip(processed_a, processed_b):
        sig_a = minhasher.signature(left.shingles)
        sig_b = minhasher.signature(right.shingles)
        candidate = bool(left.shingles and right.shingles) and static_lsh.is_candidate(sig_a, sig_b)
        exact = exact_jaccard(left.shingles, right.shingles)
        static_preds.append(int(candidate and exact >= args.jaccard_threshold))
    static_metrics = classification_metrics(labels, static_preds)

    # Evaluate Adaptive MinHash+LSH
    adaptive_lsh = LSHIndex(num_perm=args.num_perm, bands=best_b)
    adaptive_preds = []
    for left, right in zip(processed_a, processed_b):
        sig_a = minhasher.signature(left.shingles)
        sig_b = minhasher.signature(right.shingles)
        candidate = bool(left.shingles and right.shingles) and adaptive_lsh.is_candidate(sig_a, sig_b)
        exact = exact_jaccard(left.shingles, right.shingles)
        adaptive_preds.append(int(candidate and exact >= args.jaccard_threshold))
    adaptive_metrics = classification_metrics(labels, adaptive_preds)

    # Evaluate Standard SimHash
    sim_tokens = [doc.tokens for doc in processed_a] + [doc.tokens for doc in processed_b]
    std_simhasher = TfidfSimHasher()
    std_simhasher.fit(sim_tokens)
    std_sim_preds = []
    for left, right in zip(processed_a, processed_b):
        fp_a = std_simhasher.fingerprint(left.tokens)
        fp_b = std_simhasher.fingerprint(right.tokens)
        dist = hamming_distance(fp_a, fp_b)
        std_sim_preds.append(int(bool(left.tokens and right.tokens) and dist <= args.simhash_max_distance))
    std_sim_metrics = classification_metrics(labels, std_sim_preds)

    # Evaluate Hybrid SimHash
    hybrid_simhasher = HybridTfidfSimHasher(use_char_ngrams=True)
    hybrid_simhasher.fit(sim_tokens)
    hybrid_preds = []
    for left, right in zip(processed_a, processed_b):
        fp_a = hybrid_simhasher.fingerprint(left.tokens)
        fp_b = hybrid_simhasher.fingerprint(right.tokens)
        dist = hamming_distance(fp_a, fp_b)
        hybrid_preds.append(int(bool(left.tokens and right.tokens) and dist <= args.simhash_max_distance))
    hybrid_metrics = classification_metrics(labels, hybrid_preds)

    comparison_rows = [
        {
            "pipeline": "Static MinHash+LSH",
            "precision": round(static_metrics.precision, 6),
            "recall": round(static_metrics.recall, 6),
            "f1": round(static_metrics.f1, 6),
        },
        {
            "pipeline": "Adaptive MinHash+LSH",
            "precision": round(adaptive_metrics.precision, 6),
            "recall": round(adaptive_metrics.recall, 6),
            "f1": round(adaptive_metrics.f1, 6),
        },
        {
            "pipeline": "Standard SimHash",
            "precision": round(std_sim_metrics.precision, 6),
            "recall": round(std_sim_metrics.recall, 6),
            "f1": round(std_sim_metrics.f1, 6),
        },
        {
            "pipeline": "Hybrid SimHash (Bonus)",
            "precision": round(hybrid_metrics.precision, 6),
            "recall": round(hybrid_metrics.recall, 6),
            "f1": round(hybrid_metrics.f1, 6),
        }
    ]

    fieldnames = ["pipeline", "precision", "recall", "f1"]
    write_csv(args.output, comparison_rows, fieldnames)
    print(json.dumps({"bonus_metrics": comparison_rows, "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shingle-size", type=int, default=3, choices=[3, 4, 5])
    parser.add_argument("--language", default="auto", choices=["auto", "en", "fa"])
    parser.add_argument("--keep-stopwords", action="store_true")
    parser.add_argument(
        "--adaptive-short-text",
        action="store_true",
        help="Preserve stopwords and use smaller shingles for short texts.",
    )
    parser.add_argument("--short-text-token-limit", type=int, default=10)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--bands", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--jaccard-threshold",
        "--threshold",
        dest="jaccard_threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument("--simhash-max-distance", type=int, default=25)
    parser.add_argument("--simhash-ngram-size", type=int, default=1)
    parser.add_argument(
        "--simhash-mode",
        choices=["standard", "hybrid"],
        default="standard",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plagiarism-engine",
        description="Semantic duplicate and near-plagiarism detection engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="Compare two text files")
    compare.add_argument("--file-a", required=True)
    compare.add_argument("--file-b", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument(
        "--ensemble-model",
        help="Optional calibrated ensemble JSON produced by the calibrate command.",
    )
    _add_common_arguments(compare)
    compare.set_defaults(handler=command_compare)

    corpus = subparsers.add_parser("corpus", help="Find similar documents in a folder")
    corpus.add_argument("--data", required=True)
    corpus.add_argument("--output", required=True)
    corpus.add_argument(
        "--ensemble-model",
        help="Optional calibrated ensemble JSON for scoring LSH candidates.",
    )
    _add_common_arguments(corpus)
    corpus.set_defaults(handler=command_corpus)

    pairs = subparsers.add_parser("pairs", help="Evaluate on a labeled pair CSV")
    pairs.add_argument("--pairs", required=True)
    pairs.add_argument("--text-col-a", required=True)
    pairs.add_argument("--text-col-b", required=True)
    pairs.add_argument("--label-col", required=True)
    pairs.add_argument("--id-col")
    pairs.add_argument("--limit", type=int)
    pairs.add_argument("--output", required=True)
    pairs.add_argument("--predictions-output")
    pairs.add_argument(
        "--cosine-threshold",
        type=float,
        default=None,
        help="Enable TF-IDF cosine evaluation with the selected threshold.",
    )
    _add_common_arguments(pairs)
    pairs.set_defaults(handler=command_pairs)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="Train a calibrated lexical ensemble with train/validation/test splits",
    )
    calibrate.add_argument("--pairs", required=True)
    calibrate.add_argument("--text-col-a", required=True)
    calibrate.add_argument("--text-col-b", required=True)
    calibrate.add_argument("--label-col", required=True)
    calibrate.add_argument("--id-col")
    calibrate.add_argument("--limit", type=int)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--predictions-output")
    calibrate.add_argument("--model-output", required=True)
    calibrate.add_argument("--train-ratio", type=float, default=0.6)
    calibrate.add_argument("--validation-ratio", type=float, default=0.2)
    calibrate.add_argument("--ensemble-l2", type=float, default=10.0 / 3.0)
    calibrate.add_argument("--ensemble-max-iter", type=int, default=25)
    _add_common_arguments(calibrate)
    calibrate.set_defaults(handler=command_calibrate)

    bonus_eval = subparsers.add_parser("bonus-eval", help="Evaluate with bonus features")
    bonus_eval.add_argument("--pairs", required=True)
    bonus_eval.add_argument("--text-col-a", required=True)
    bonus_eval.add_argument("--text-col-b", required=True)
    bonus_eval.add_argument("--label-col", required=True)
    bonus_eval.add_argument("--id-col")
    bonus_eval.add_argument("--limit", type=int)
    bonus_eval.add_argument("--output", required=True)
    _add_common_arguments(bonus_eval)
    bonus_eval.set_defaults(handler=command_bonus_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_perm % args.bands != 0:
        parser.error("--num-perm must be divisible by --bands")
    try:
        return int(args.handler(args))
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
