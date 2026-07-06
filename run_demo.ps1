$ErrorActionPreference = "Stop"

python -m plagiarism_engine.cli compare `
  --file-a data/sample_corpus/doc_01.txt `
  --file-b data/sample_corpus/doc_02.txt `
  --output outputs/two_file_compare.json

python -m plagiarism_engine.cli corpus `
  --data data/sample_corpus `
  --threshold 0.25 `
  --shingle-size 3 `
  --output outputs/candidates.csv

python -m plagiarism_engine.cli pairs `
  --pairs data/processed/sample_pairs.csv `
  --text-col-a text_a `
  --text-col-b text_b `
  --label-col label `
  --id-col pair_id `
  --output outputs/metrics.csv `
  --predictions-output outputs/pair_predictions.csv

python -m unittest discover -s tests -v
