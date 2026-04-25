import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "LangSAE" / "evaluation" / "base_eval.py"
SAE = ROOT / "LangSAE" / "evaluation" / "sae_eval.py"
README = ROOT / "LangSAE" / "evaluation" / "README.md"


def module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def function_names(path: Path) -> set[str]:
    return {node.name for node in ast.walk(module(path)) if isinstance(node, ast.FunctionDef)}


def class_names(path: Path) -> set[str]:
    return {node.name for node in ast.walk(module(path)) if isinstance(node, ast.ClassDef)}


def test_evaluation_package_exposes_base_and_sae_entrypoints():
    assert {"load_queries_from_jsonl", "load_corpus_from_jsonl", "load_qrels_from_jsonl", "run"} <= function_names(BASE)
    assert {"DenseRetriever"} <= class_names(BASE)
    assert {"load_langsae_from_checkpoint", "run_sae_eval"} <= function_names(SAE)
    assert {"SAERetriever"} <= class_names(SAE)


def test_sae_evaluation_defaults_to_reconstructed_embedding_space():
    tree = module(SAE)
    init = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    defaults = dict(zip([arg.arg for arg in init.args.args[-len(init.args.defaults):]], init.args.defaults))
    assert isinstance(defaults["use_reconstruction"], ast.Constant)
    assert defaults["use_reconstruction"].value is True

    run = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_sae_eval"
    )
    defaults = dict(zip([arg.arg for arg in run.args.args[-len(run.args.defaults):]], run.args.defaults))
    assert isinstance(defaults["use_reconstruction"], ast.Constant)
    assert defaults["use_reconstruction"].value is True


def test_model_specific_prompt_prefixes_are_present():
    source = BASE.read_text(encoding="utf-8")
    assert "query: {query}" in source
    assert "Instruct: Given a web search query" in source
    assert "Represent the query for retrieving evidence documents" in source
    assert "task: search result | query:" in source
    assert "passage: {text}" in source
    assert "title: none | text:" in source


def test_docs_warn_how_to_run_real_benchmark_evaluation():
    text = README.read_text(encoding="utf-8")
    assert "queries.jsonl" in text
    assert "corpus.jsonl" in text
    assert "qrels.jsonl" in text
    assert "python -m LangSAE.evaluation.base_eval" in text
    assert "python -m LangSAE.evaluation.sae_eval" in text
    assert "--use_reconstruction=True" in text
