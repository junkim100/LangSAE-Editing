"""BEIR-style retrieval evaluation for LangSAE-edited embeddings."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import fire
import torch

from LangSAE.inference import remove_language_features
from LangSAE.model import LangSAE

from .base_eval import (
    _require_beir_evaluator,
    _require_sentence_transformers,
    add_prefix_to_corpus,
    add_prefix_to_queries,
    get_safe_model_name,
    load_corpus_from_jsonl,
    load_qrels_from_jsonl,
    load_queries_from_jsonl,
    load_sentence_transformer,
)

logger = logging.getLogger(__name__)


def infer_sparsity_from_checkpoint(checkpoint: Any, sae_path: str, dict_size: int) -> int:
    """Infer top-k sparsity from checkpoint metadata or folder names."""
    sparsity = None
    if isinstance(checkpoint, dict):
        for key in ("sparsity", "k", "top_k", "topk"):
            if key in checkpoint:
                try:
                    sparsity = int(checkpoint[key])
                    break
                except Exception:
                    pass
    if sparsity is None:
        match = re.search(r"(?:^|_)k(\d+)(?:_|$)", Path(sae_path).parent.name)
        if match:
            sparsity = int(match.group(1))
    if sparsity is None:
        sparsity = 64
    return max(1, min(int(sparsity), dict_size - 1))


def load_langsae_from_checkpoint(sae_path: str, device: torch.device) -> LangSAE:
    """Load a LangSAE checkpoint saved as either a wrapper dict or a raw state dict."""
    checkpoint = torch.load(sae_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "encoder.weight" in checkpoint:
        state_dict = checkpoint
    else:
        raise ValueError(f"Could not parse checkpoint format from {sae_path}")

    encoder_weight = state_dict["encoder.weight"]
    input_dim = encoder_weight.shape[1]
    dict_size = encoder_weight.shape[0]
    expansion_factor = dict_size // input_dim
    sparsity = infer_sparsity_from_checkpoint(checkpoint, sae_path, dict_size)

    sae = LangSAE(input_dim=input_dim, expansion_factor=expansion_factor, sparsity=sparsity)
    sae.load_state_dict(state_dict)
    sae.to(device)
    sae.eval()
    return sae


class SAERetriever:
    """Retriever that evaluates embeddings after LangSAE language-feature removal."""

    def __init__(
        self,
        model,
        sae: LangSAE,
        mask: torch.Tensor,
        batch_size: int = 128,
        use_reconstruction: bool = True,
        show_progress_bar: bool = True,
        **_: Any,
    ) -> None:
        self.model = model
        self.sae = sae
        self.mask = mask
        self.batch_size = batch_size
        self.use_reconstruction = use_reconstruction
        self.show_progress_bar = show_progress_bar
        self.results: dict[str, dict[str, float]] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sae.to(self.device)
        self.sae.eval()
        self.mask = self.mask.to(self.device).bool()

    def apply_sae_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Mask language-specific SAE features and optionally reconstruct embeddings."""
        transformed = []
        with torch.no_grad():
            for start in range(0, len(embeddings), self.batch_size):
                batch = embeddings[start : start + self.batch_size].to(self.device)
                _, features, _, _, _ = self.sae(batch)
                features_agnostic = remove_language_features(features, self.mask)
                if self.use_reconstruction:
                    output = self.sae.decoder(features_agnostic)
                else:
                    output = features_agnostic
                transformed.append(output.detach().cpu())
        return torch.cat(transformed, dim=0)

    def search(
        self,
        corpus: dict[str, dict[str, Any]],
        queries: dict[str, str],
        top_k: int,
        score_function: str = "cos_sim",
        **_: Any,
    ) -> dict[str, dict[str, float]]:
        """Encode, edit, and retrieve top-k documents for each query."""
        _, cos_sim = _require_sentence_transformers()
        logger.info("Encoding queries...")
        query_ids = list(queries.keys())
        query_embeddings = self.model.encode(
            [queries[qid] for qid in query_ids],
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
            device=self.device,
        )
        query_embeddings = self.apply_sae_transformation(query_embeddings)

        logger.info("Encoding corpus...")
        corpus_ids = list(corpus.keys())
        corpus_texts = []
        for corpus_id in corpus_ids:
            doc = corpus[corpus_id]
            title = doc.get("title", "")
            text = doc.get("text", "")
            corpus_texts.append(f"{title} {text}".strip() if title else text.strip())
        corpus_embeddings = self.model.encode(
            corpus_texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
            device=self.device,
        )
        corpus_embeddings = self.apply_sae_transformation(corpus_embeddings)

        cos_scores = cos_sim(query_embeddings, corpus_embeddings)
        self.results = {}
        for idx, query_id in enumerate(query_ids):
            scores = cos_scores[idx]
            top_results = torch.topk(scores, k=min(top_k, len(corpus_ids)))
            self.results[query_id] = {}
            for score, corpus_idx in zip(top_results.values.tolist(), top_results.indices.tolist()):
                corpus_id = corpus_ids[corpus_idx]
                if corpus_id != query_id:
                    self.results[query_id][corpus_id] = float(score)
        return self.results

    def encode(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("File-based BEIR encoding is not implemented")

    def search_from_files(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("File-based BEIR search is not implemented")


def run_sae_eval(
    model: str,
    sae_path: str,
    mask_path: str,
    data_dirs: list[str],
    results_root: str = "./results_sae_eval",
    batch_size: int = 128,
    max_seq_length: int = 512,
    use_reconstruction: bool = True,
    mask_threshold: Optional[float] = None,
    overwrite: bool = False,
) -> None:
    """Evaluate LangSAE-edited embeddings on BEIR-style retrieval datasets."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    results_path = Path(results_root)
    results_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st_model = load_sentence_transformer(model, max_seq_length=max_seq_length)
    sae = load_langsae_from_checkpoint(sae_path, device=device)
    mask = torch.load(mask_path, map_location=device).bool()
    if mask.shape[0] != sae.dict_size:
        raise ValueError(f"Mask size {mask.shape[0]} does not match SAE dict_size {sae.dict_size}")

    masked_features = int(mask.sum().item())
    total_features = int(mask.shape[0])
    print(f"Mask Statistics: masked={masked_features}/{total_features} ({masked_features / total_features:.2%})")

    retriever_kwargs = {
        "model": st_model,
        "sae": sae,
        "mask": mask,
        "batch_size": batch_size,
        "use_reconstruction": use_reconstruction,
    }
    checkpoint_name = Path(sae_path).parent.name or Path(sae_path).stem
    mode_name = "reconstructed" if use_reconstruction else "features"
    model_safe_name = get_safe_model_name(model)
    if mask_threshold is not None:
        mask_str = f"mask{str(mask_threshold).replace('.', '_')}"
        output_dir = results_path / f"{model_safe_name}_{checkpoint_name}_{mask_str}"
    else:
        output_dir = results_path / f"{model_safe_name}_{checkpoint_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    EvaluateRetrieval = _require_beir_evaluator()
    for data_dir in data_dirs:
        dataset_name = os.path.basename(data_dir.rstrip("/"))
        output_file = output_dir / f"{dataset_name}_{mode_name}_results.json"
        if output_file.exists() and not overwrite:
            print(f"[SKIP] Results already exist at: {output_file}")
            continue

        print(f"Evaluating {model} + LangSAE on {dataset_name}")
        queries = add_prefix_to_queries(load_queries_from_jsonl(os.path.join(data_dir, "queries.jsonl")), model)
        corpus = add_prefix_to_corpus(load_corpus_from_jsonl(os.path.join(data_dir, "corpus.jsonl")), model)
        qrels = load_qrels_from_jsonl(os.path.join(data_dir, "qrels.jsonl"))

        retriever = SAERetriever(**retriever_kwargs)
        evaluator = EvaluateRetrieval(retriever=retriever)
        results = evaluator.retrieve(corpus, queries)
        ndcg, _map, recall, precision = evaluator.evaluate(
            qrels, results, k_values=[1, 3, 5, 10, 20, 100, 1000]
        )
        results_dict = {
            "model": model,
            "sae": sae_path,
            "mask": mask_path,
            "mask_threshold": mask_threshold,
            "checkpoint_folder": checkpoint_name,
            "dataset": dataset_name,
            "ndcg": ndcg,
            "map": _map,
            "recall": recall,
            "precision": precision,
            "use_reconstruction": use_reconstruction,
        }
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")


def cli() -> None:
    fire.Fire(run_sae_eval)


if __name__ == "__main__":
    cli()
