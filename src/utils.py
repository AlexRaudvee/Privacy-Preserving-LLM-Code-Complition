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
from dataclasses import dataclass
from typing import Dict, Set, Tuple, Optional

from rouge_score import rouge_scorer
from rapidfuzz.distance import Levenshtein


_PY_KEYWORDS: Set[str] = set(keyword.kwlist)
_BUILTINS: Set[str] = set(dir(builtins))


def _strip_comments_and_docstrings(code: str) -> str:
    """
    Removes:
      - # comments
      - standalone triple-quoted docstrings at module/function level (best-effort)
    Keeps string literals elsewhere.
    """
    # 1) Strip comments using tokenize (robust vs regex)
    out_tokens = []
    try:
        tokgen = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok_type, tok_str, start, end, line in tokgen:
            if tok_type == tokenize.COMMENT:
                continue
            out_tokens.append((tok_type, tok_str))
        code_no_comments = tokenize.untokenize(out_tokens)
    except Exception:
        code_no_comments = re.sub(r"#.*", "", code)

    # 2) Remove docstrings using AST (best-effort). If parse fails, fall back to regex.
    try:
        tree = ast.parse(code_no_comments)

        class DocstringRemover(ast.NodeTransformer):
            def _remove_docstring(self, node):
                if node.body and isinstance(node.body[0], ast.Expr):
                    expr = node.body[0].value
                    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
                        node.body = node.body[1:]
                return node

            def visit_Module(self, node: ast.Module):
                node = self.generic_visit(node)
                return self._remove_docstring(node)

            def visit_FunctionDef(self, node: ast.FunctionDef):
                node = self.generic_visit(node)
                return self._remove_docstring(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                node = self.generic_visit(node)
                return self._remove_docstring(node)

        tree = DocstringRemover().visit(tree)
        ast.fix_missing_locations(tree)
        # ast.unparse exists in 3.9+, but we can still rely on it for most environments.
        return ast.unparse(tree)
    except Exception:
        # Simple regex removal of leading docstring after "def ..."
        return re.sub(r'^\s*(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\')\s*', '', code_no_comments, flags=re.DOTALL)


def _collect_defined_names(tree: ast.AST) -> Set[str]:
    names: Set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name):
            names.add(node.id)

        def visit_arg(self, node: ast.arg):
            names.add(node.arg)

        def visit_FunctionDef(self, node: ast.FunctionDef):
            names.add(node.name)
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef):
            names.add(node.name)
            self.generic_visit(node)

    Visitor().visit(tree)
    return names


def _make_name_mapping(used_names: Set[str], mode: str) -> Dict[str, str]:
    """
    mode:
      - "low": keep some readability (v1, v2, ...)
      - "high": placeholders (VAR_0, VAR_1, ...)
    """
    mapping: Dict[str, str] = {}
    i = 0
    for name in sorted(used_names):
        if (
            name in _PY_KEYWORDS
            or name in _BUILTINS
            or name.startswith("__")
        ):
            continue
        # Avoid renaming very common semantic names you might want to preserve (tune as desired)
        if mode == "low" and name in {"self", "cls"}:
            continue

        new_name = (f"v{i}" if mode == "low" else f"VAR_{i}")
        # ensure we don't collide with existing names
        while new_name in used_names or new_name in _PY_KEYWORDS or new_name in _BUILTINS:
            i += 1
            new_name = (f"v{i}" if mode == "low" else f"VAR_{i}")
        mapping[name] = new_name
        i += 1
    return mapping


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name):
        if node.id in self.mapping:
            return ast.copy_location(ast.Name(id=self.mapping[node.id], ctx=node.ctx), node)
        return node

    def visit_arg(self, node: ast.arg):
        if node.arg in self.mapping:
            node.arg = self.mapping[node.arg]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Do not rename the function name itself in this assignment (keep prompt anchored)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node: ast.Attribute):
        # Avoid renaming attribute names (e.g., obj.foo) because it can break code structure.
        self.generic_visit(node)
        return node


def low_obfuscation(prompt: str) -> str:
    """
    "Low" obfuscation: variable renaming only (best-effort).
    """
    try:
        tree = ast.parse(prompt)
        used = _collect_defined_names(tree)
        mapping = _make_name_mapping(used, mode="low")
        tree2 = _Renamer(mapping).visit(tree)
        ast.fix_missing_locations(tree2)
        return ast.unparse(tree2)
    except Exception:
        # If parsing fails, do a conservative regex rename of simple identifiers.
        return prompt


def high_obfuscation(prompt: str) -> str:
    """
    "High" obfuscation:
      - strip comments and docstrings
      - rename identifiers more aggressively
    """
    stripped = _strip_comments_and_docstrings(prompt)
    try:
        tree = ast.parse(stripped)
        used = _collect_defined_names(tree)
        mapping = _make_name_mapping(used, mode="high")
        tree2 = _Renamer(mapping).visit(tree)
        ast.fix_missing_locations(tree2)
        return ast.unparse(tree2)
    except Exception:
        return stripped


def privacy_score(prompt_variant: str, prompt_original: str) -> float:
    """
    Normalized Levenshtein distance in [0,1]:
      0 = identical to original prompt (low privacy)
      1 = maximally different (high privacy, by this proxy)
    """
    if not prompt_original:
        return 0.0
    dist = Levenshtein.distance(prompt_variant, prompt_original)
    return dist / max(len(prompt_original), len(prompt_variant), 1)


_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

def utility_score_rougeL(completion: str, canonical_solution: str) -> float:
    """
    ROUGE-L F1 between completion and canonical solution in [0,1].
    """
    scores = _scorer.score(canonical_solution, completion)
    return float(scores["rougeL"].fmeasure)
