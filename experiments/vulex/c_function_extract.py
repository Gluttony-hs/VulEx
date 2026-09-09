import re
from dataclasses import dataclass

from tree_sitter import Language, Parser
import tree_sitter_c

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class FunctionSpan:
    name: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class FunctionBody:
    name: str
    body: str


def patch_old_target_lines(patch: str) -> list[int]:
    """Extract old-side target line numbers from a unified diff; prefer deletions and use the hunk start for insertion-only hunks."""
    targets: list[int] = []
    old_line: int | None = None
    hunk_start: int | None = None
    hunk_has_delete = False
    for line in patch.splitlines():
        match = HUNK_RE.match(line)
        if match:
            if hunk_start is not None and not hunk_has_delete:
                targets.append(hunk_start)
            old_line = int(match.group(1))
            hunk_start = old_line
            hunk_has_delete = False
            continue
        if old_line is None:
            continue
        if line.startswith("-") and not line.startswith("---"):
            targets.append(old_line)
            hunk_has_delete = True
            old_line += 1
        elif line.startswith(" "):
            old_line += 1
        elif line.startswith("+") and not line.startswith("+++"):
            continue
    if hunk_start is not None and not hunk_has_delete:
        targets.append(hunk_start)
    return list(dict.fromkeys(targets))


def _parser() -> Parser:
    """Construct the C parser inside the function so import-time initialization errors remain visible."""
    return Parser(Language(tree_sitter_c.language()))


def _node_text(source: bytes, node) -> str:
    """Extract a source fragment using a tree-sitter byte span."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _function_name(source: bytes, function_node) -> str | None:
    """Extract the C function name from a function_definition declarator."""
    declarator = function_node.child_by_field_name("declarator")
    stack = [declarator] if declarator is not None else []
    while stack:
        node = stack.pop()
        if node.type == "function_declarator":
            name_node = node.child_by_field_name("declarator")
            while name_node is not None and name_node.type in {"parenthesized_declarator", "pointer_declarator"}:
                name_node = name_node.child_by_field_name("declarator")
            if name_node is not None and name_node.type == "identifier":
                return _node_text(source, name_node)
        stack.extend(reversed(node.children))
    return None


def _c_function_body_spans(code: str) -> list[tuple[int, int, FunctionBody]]:
    """Scan C source text and return spans of parseable function bodies in the original text."""
    source = code.encode("utf-8", errors="replace")
    tree = _parser().parse(source)
    bodies: list[tuple[int, int, FunctionBody]] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "function_definition":
            name = _function_name(source, node)
            if name:
                bodies.append(
                    (
                        node.start_point[0] + 1,
                        node.end_point[0] + 1,
                        FunctionBody(name, _node_text(source, node)),
                    )
                )
            continue
        stack.extend(reversed(node.children))
    return bodies


def c_function_spans(code_before: str) -> list[FunctionSpan]:
    """Parse C source with tree-sitter-c and return line ranges for all function definitions."""
    return [
        FunctionSpan(body.name, start_line, end_line)
        for start_line, end_line, body in _c_function_body_spans(code_before)
    ]


def c_function_bodies(code: str) -> list[FunctionBody]:
    """Use tree-sitter-c to return C function names and complete function bodies."""
    return [body for _start_line, _end_line, body in _c_function_body_spans(code)]


def enclosing_function_body_from_code_before_patch(code_before: str, patch: str) -> FunctionBody | None:
    """Use a patch old-side line number to locate the first enclosing C function body in code_before."""
    bodies = _c_function_body_spans(code_before)
    if not bodies:
        return None
    for line in patch_old_target_lines(patch):
        hits = [
            (start_line, end_line, body)
            for start_line, end_line, body in bodies
            if start_line <= line <= end_line
        ]
        if not hits:
            continue
        hits.sort(key=lambda hit: (hit[1] - hit[0], hit[0]))
        return hits[0][2]
    return None


def enclosing_function_from_code_before_patch(code_before: str, patch: str) -> str | None:
    """Use a patch old-side line number to locate the first enclosing C function in code_before."""
    body = enclosing_function_body_from_code_before_patch(code_before, patch)
    return body.name if body else None
