"""test_model_dump_guard.py.

Coverage: 22%. Missing: 13-18, 23-29, 33-37, 50-74, 77-78
Last Update: 10 December, 2025
"""

import ast
from pathlib import Path


def _contains_record_is_parsed_in_test(test_node: ast.AST) -> bool:
    """Recursively check if the AST test contains attribute 'record.is_parsed'."""
    for node in ast.walk(test_node):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "record":
                if node.attr == "is_parsed":
                    return True
    return False


def _is_guarded_by_is_parsed(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Walk up parents to see if node is inside an if guarded by 'record.is_parsed'."""
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.If):
            if _contains_record_is_parsed_in_test(current.test):
                return True
    return False


def _iter_ast_parents(tree: ast.AST):
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def test_no_unguarded_record_model_dump_calls():
    """Scan src/ for instances where 'record.model_dump' is called without an 'exclude' keyword and without being guarded by 'if record.is_parsed'.

    Fail the test if any such occurrences are found.
    """
    src_dir = Path("src")
    issues = []

    for py_file in src_dir.rglob("*.py"):
        # Skip test files and migrations
        if py_file.match("**/tests/**"):
            continue
        code = py_file.read_text()
        try:
            tree = ast.parse(code, filename=str(py_file))
        except SyntaxError:
            # Ignore parse failures in non-Python files or generated code
            continue
        parents = _iter_ast_parents(tree)

        for node in ast.walk(tree):
            # Look for calls like 'record.model_dump(...)'
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                func = node.func
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id == "record"
                    and func.attr == "model_dump"
                ):
                    # Check if 'exclude' keyword present
                    has_exclude = any((kw.arg == "exclude") for kw in node.keywords)
                    # Check for guarding if 'record.is_parsed' above this call
                    guarded = _is_guarded_by_is_parsed(node, parents)
                    if not has_exclude and not guarded:
                        issues.append((py_file, node.lineno))

    if issues:
        msg_lines = [f"Ungarded record.model_dump() in: {p}:{ln}" for p, ln in issues]
        assert False, "\n".join(msg_lines)
