"""BEIR-style retrieval evaluation for base embedding models."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import fire
import torch

logger = logging.getLogger(__name__)


def _require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import cos_sim
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise ImportError(
            "Evaluation requires sentence-transformers. Install the evaluation dependencies "
            "documented in the README before running this module."
        ) from exc
    return SentenceTransformer, cos_sim


def _require_beir_evaluator():
    try:
        from beir.retrieval.evaluation import EvaluateRetrieval
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise ImportError(
            "Evaluation requires beir. Install the evaluation dependencies documented "
            "in the README before running this module."
        ) from exc
    return EvaluateRetrieval


def load_queries_from_jsonl(file_path: str) -> dict[str, str]:
    """Load BEIR-style queries from JSONL as ``{query_id: text}``."""
    queries: dict[str, str] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "_id" in data and "text" in data:
                queries[data["_id"]] = data["text"]
            else:
                queries.update(data)
    return queries


def load_corpus_from_jsonl(file_path: str) -> dict[str, dict[str, Any]]:
    """Load BEIR-style corpus entries from JSONL as ``{doc_id: doc}``."""
    corpus: dict[str, dict[str, Any]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            corpus.update(json.loads(line))
    return corpus


def load_qrels_from_jsonl(file_path: str) -> dict[str, dict[str, int]]:
    """Load BEIR-style qrels from JSONL as ``{query_id: {doc_id: relevance}}``."""
    qrels: dict[str, dict[str, int]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            qrels.update(json.loads(line))
    return qrels


def add_prefix_to_queries(queries: dict[str, str], model_name: str) -> dict[str, str]:
    """Apply model-specific query prompts used by common embedding models."""
    lower = model_name.lower()
    if "e5" in lower or "snowflake-arctic-embed" in lower:
        return {qid: f"query: {query}" for qid, query in queries.items()}
    if "qwen" in lower or "gte-qwen" in lower:
        return {
            qid: (
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
                f"Query: {query}"
            )
            for qid, query in queries.items()
        }
    if "jina" in lower:
        return {
            qid: f"Represent the query for retrieving evidence documents: {query}"
            for qid, query in queries.items()
        }
    if "embeddinggemma" in lower:
        return {qid: f"task: search result | query: {query}" for qid, query in queries.items()}
    if model_name == "nvidia/llama-nemotron-embed-1b-v2":
        return {qid: f"query: {query}" for qid, query in queries.items()}
    if "mxbai" in lower:
        return {
            qid: f"Represent this sentence for searching relevant passages: {query}"
            for qid, query in queries.items()
        }
    return queries


def add_prefix_to_corpus(
    corpus: dict[str, dict[str, Any]], model_name: str
) -> dict[str, dict[str, Any]]:
    """Apply model-specific document prompts while preserving metadata fields."""
    lower = model_name.lower()
    prefixed: dict[str, dict[str, Any]] = {}
    for doc_id, doc in corpus.items():
        new_doc = dict(doc)
        text = doc.get("text", "")
        if "e5" in lower:
            new_doc["text"] = f"passage: {text}"
        elif "jina" in lower:
            new_doc["text"] = f"Represent the document for retrieval: {text}"
        elif "embeddinggemma" in lower:
            new_doc["text"] = f"title: none | text: {text}"
        elif model_name == "nvidia/llama-nemotron-embed-1b-v2":
            new_doc["text"] = f"passage: {text}"
        else:
            new_doc["text"] = text
        prefixed[doc_id] = new_doc
    return prefixed


def load_sentence_transformer(model_name: str, max_seq_length: int):
    """Load a SentenceTransformer with model-specific precision/trust settings."""
    SentenceTransformer, _ = _require_sentence_transformers()
    lower = model_name.lower()
    if "gte" in lower:
        model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            tokenizer_kwargs={"fix_mistral_regex": True},
        )
        if "qwen" in lower:
            model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                model_kwargs={"dtype": torch.bfloat16},
            )
    elif "jina" in lower or "llama-nemotron-embed" in lower:
        model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            model_kwargs={"dtype": torch.bfloat16},
        )
    elif "qwen" in lower:
        model = SentenceTransformer(model_name, model_kwargs={"dtype": torch.bfloat16})
    elif "mxbai" in lower:
        model = SentenceTransformer(model_name, model_kwargs={"dtype": torch.float16})
    else:
        model = SentenceTransformer(model_name, tokenizer_kwargs={"fix_mistral_regex": True})

    model.max_seq_length = max_seq_length
    for module in model.modules():
        if hasattr(module, "config") and hasattr(module.config, "use_cache"):
            module.config.use_cache = False
    return model


class DenseRetriever:
    """Dense retriever wrapper compatible with BEIR's ``EvaluateRetrieval``."""

    def __init__(self, model, batch_size: int = 128, show_progress_bar: bool = True, **_: Any) -> None:
        self.model = model
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.results: dict[str, dict[str, float]] = {}

    def search(
        self,
        corpus: dict[str, dict[str, Any]],
        queries: dict[str, str],
        top_k: int,
        score_function: str = "cos_sim",
        **_: Any,
    ) -> dict[str, dict[str, float]]:
        """Encode queries/corpus and return top-k cosine-similarity results."""
        _, cos_sim = _require_sentence_transformers()
        logger.info("Encoding queries...")
        query_ids = list(queries.keys())
        query_embeddings = self.model.encode(
            [queries[qid] for qid in query_ids],
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
        )

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
        )

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


def get_safe_model_name(model_name: str) -> str:
    """Return a filesystem-safe model identifier for result directories."""
    name = model_name.rstrip("/")
    if os.path.exists(name):
        name = os.path.basename(name)
    return name.replace("/", "_").replace(" ", "_")


def _evaluate_one_dataset(model_name: str, data_dir: str, output_file: Path, retriever: DenseRetriever) -> None:
    queries = add_prefix_to_queries(load_queries_from_jsonl(os.path.join(data_dir, "queries.jsonl")), model_name)
    corpus = add_prefix_to_corpus(load_corpus_from_jsonl(os.path.join(data_dir, "corpus.jsonl")), model_name)
    qrels = load_qrels_from_jsonl(os.path.join(data_dir, "qrels.jsonl"))

    EvaluateRetrieval = _require_beir_evaluator()
    evaluator = EvaluateRetrieval(retriever=retriever)
    results = evaluator.retrieve(corpus, queries)
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels, results, k_values=[1, 3, 5, 10, 20, 100, 1000]
    )

    results_dict = {
        "model": model_name,
        "dataset": os.path.basename(data_dir.rstrip("/")),
        "ndcg": ndcg,
        "map": _map,
        "recall": recall,
        "precision": precision,
    }
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)


def run(
    model: str,
    data_dirs: list[str],
    results_root: str = "./results_base",
    batch_size: int = 1024,
    overwrite: bool = False,
    max_seq_length: int = 512,
) -> None:
    """Run BEIR-style evaluation for a base embedding model."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    results_path = Path(results_root)
    results_path.mkdir(parents=True, exist_ok=True)

    st_model = load_sentence_transformer(model, max_seq_length=max_seq_length)
    retriever = DenseRetriever(st_model, batch_size=batch_size)
    output_dir = results_path / get_safe_model_name(model)
    output_dir.mkdir(parents=True, exist_ok=True)

    for data_dir in data_dirs:
        dataset_name = os.path.basename(data_dir.rstrip("/"))
        output_file = output_dir / f"{dataset_name}_results.json"
        if output_file.exists() and not overwrite:
            print(f"[SKIP] Results already exist at: {output_file}")
            continue
        print(f"Evaluating {model} on {dataset_name}")
        _evaluate_one_dataset(model, data_dir, output_file, retriever)
        print(f"Results saved to: {output_file}")


def cli() -> None:
    fire.Fire(run)


if __name__ == "__main__":
    cli()
