"""
Utilities for:
- prompt obfuscation
- metrics: ROUGE-L (utility), normalized Levenshtein distance (privacy)
"""

from __future__ import annotations

import ast
import builtins
import io
import keyword
import re
import tokenize
from typing import Dict, Set

from rouge_score import rouge_scorer
from rapidfuzz.distance import Levenshtein


# -------------------------
# Globals
# -------------------------

_PY_KEYWORDS: Set[str] = set(keyword.kwlist)
_BUILTINS: Set[str] = set(dir(builtins))


# -------------------------
# Helpers
# -------------------------

def _strip_comments(code: str) -> str:
    """Remove # comments, keep everything else."""
    out_tokens = []
    try:
        tokgen = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok_type, tok_str, *_ in tokgen:
            if tok_type != tokenize.COMMENT:
                out_tokens.append((tok_type, tok_str))
        return tokenize.untokenize(out_tokens)
    except Exception:
        return re.sub(r"#.*", "", code)


def _normalize_docstring(ds: str) -> str:
    """
    Keep only the first sentence of a docstring.
    Removes examples, lists, and extra detail.
    """
    ds = ds.strip()
    # Cut at first example or doctest
    ds = ds.split(">>>")[0]
    # Take first sentence
    match = re.split(r"\.\s+", ds, maxsplit=1)
    return match[0].strip() + "." if match else ds


def _collect_body_names(func: ast.FunctionDef) -> Set[str]:
    """Collect identifiers used inside function body only."""
    names: Set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name):
            names.add(node.id)

        def visit_Attribute(self, node: ast.Attribute):
            self.visit(node.value)

    for stmt in func.body:
        Visitor().visit(stmt)

    return names


def _make_name_mapping(used_names: Set[str], mode: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    i = 0

    for name in sorted(used_names):
        if (
            name in _PY_KEYWORDS
            or name in _BUILTINS
            or name.startswith("__")
            or name in {"self", "cls"}
        ):
            continue

        new_name = f"v{i}" if mode == "low" else f"VAR_{i}"
        while new_name in used_names:
            i += 1
            new_name = f"v{i}" if mode == "low" else f"VAR_{i}"

        mapping[name] = new_name
        i += 1

    return mapping


# -------------------------
# AST transformers
# -------------------------

class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name):
        if node.id in self.mapping:
            return ast.copy_location(
                ast.Name(id=self.mapping[node.id], ctx=node.ctx), node
            )
        return node

    def visit_Attribute(self, node: ast.Attribute):
        node.value = self.visit(node.value)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Preserve API: name, args, returns, decorators
        node.body = [self.visit(stmt) for stmt in node.body]
        return node


class _LiteralMasker(ast.NodeTransformer):
    """Mask string and numeric literals."""

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="STR"), node)
        if isinstance(node.value, (int, float)):
            return ast.copy_location(ast.Constant(value=0), node)
        return node


class _DocstringNormalizer(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef):
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body[0].value.value = _normalize_docstring(
                node.body[0].value.value
            )
        return node


# -------------------------
# Public API
# -------------------------

def low_obfuscation(prompt: str) -> str:
    """
    Low obfuscation:
    - rename local variables only
    - preserve signature, types, docstring
    - normalize formatting via AST
    """
    try:
        tree = ast.parse(prompt)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                names = _collect_body_names(node)
                mapping = _make_name_mapping(names, mode="low")
                _Renamer(mapping).visit(node)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return prompt


def high_obfuscation(prompt: str) -> str:
    """
    High obfuscation:
    - strip # comments
    - normalize docstring
    - rename locals aggressively
    - mask string & numeric literals
    """
    try:
        stripped = _strip_comments(prompt)
        tree = ast.parse(stripped)

        _DocstringNormalizer().visit(tree)
        _LiteralMasker().visit(tree)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                names = _collect_body_names(node)
                mapping = _make_name_mapping(names, mode="high")
                _Renamer(mapping).visit(node)

        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return prompt


# -------------------------
# Metrics
# -------------------------

def privacy_score(prompt_variant: str, prompt_original: str) -> float:
    if not prompt_original:
        return 0.0
    dist = Levenshtein.distance(prompt_variant, prompt_original)
    return dist / max(len(prompt_original), len(prompt_variant), 1)


_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

def utility_score_rougeL(completion: str, canonical_solution: str) -> float:
    scores = _scorer.score(canonical_solution, completion)
    return float(scores["rougeL"].fmeasure)
