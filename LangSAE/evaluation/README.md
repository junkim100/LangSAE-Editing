# LangSAE Retrieval Evaluation

This package contains the BEIR-style retrieval evaluation code used for LangSAE Editing experiments.
It evaluates directories containing:

- `queries.jsonl`
- `corpus.jsonl`
- `qrels.jsonl`

Each output is a JSON file with BEIR `ndcg`, `map`, `recall`, and `precision` metrics at `[1, 3, 5, 10, 20, 100, 1000]`.

## Install dependencies

```bash
uv pip install -e .
# or
pip install -e .
```

## Base-model evaluation

```bash
python -m LangSAE.evaluation.base_eval \
  --model="intfloat/multilingual-e5-large" \
  --data_dirs='["data/Belebele/Belebele_test_en_to_all"]' \
  --results_root="results_base" \
  --batch_size=1024 \
  --max_seq_length=512
```

## LangSAE-edited evaluation

```bash
python -m LangSAE.evaluation.sae_eval \
  --model="intfloat/multilingual-e5-large" \
  --sae_path="checkpoints/exp128_k4096_lr5e-04/final_model.pt" \
  --mask_path="analysis/language_features_combined_mask.pt" \
  --data_dirs='["data/Belebele/Belebele_test_en_to_all"]' \
  --results_root="results_sae_eval" \
  --batch_size=128 \
  --max_seq_length=512 \
  --use_reconstruction=True
```

`--use_reconstruction=True` is the recommended default for published-style retrieval evaluation because it reconstructs edited sparse features back into the base embedding dimensionality before cosine search. Use `False` only when intentionally comparing sparse feature-space retrieval.
