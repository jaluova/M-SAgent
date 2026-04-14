#!/usr/bin/env python3

from __future__ import annotations

import ast
import importlib
import importlib.util
import py_compile
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TESTS_ROOT = REPO_ROOT / "tests"
FIRST_PARTY_ROOTS = {"msagent", "tests"}
STDLIB_MODULES = set(sys.stdlib_module_names)
FORBIDDEN_IMPORT_MODULES = {
    "tomllib",
}
FORBIDDEN_IMPORT_NAMES = {
    "enum": {
        "CONFORM",
        "CONTINUOUS",
        "EJECT",
        "EnumCheck",
        "FlagBoundary",
        "KEEP",
        "NAMED_FLAGS",
        "ReprEnum",
        "STRICT",
        "StrEnum",
        "global_enum",
        "member",
        "nonmember",
        "property",
        "show_flag_values",
        "verify",
    },
    "itertools": {
        "batched",
    },
    "math": {
        "cbrt",
        "exp2",
        "sumprod",
    },
    "pathlib": {
        "UnsupportedOperation",
    },
    "typing": {
        "Never",
        "NotRequired",
        "Required",
        "Self",
        "TypeAliasType",
        "Unpack",
        "assert_never",
        "assert_type",
        "clear_overloads",
        "dataclass_transform",
        "get_overloads",
        "override",
        "reveal_type",
    },
    "warnings": {
        "deprecated",
    },
}
ALLOWED_TOP_LEVEL_EXTERNAL_IMPORTS = {
    "__future__",
}


@dataclass(slots=True)
class GuardFailure:
    path: Path
    message: str


def iter_python_files() -> list[Path]:
    roots = (SRC_ROOT, TESTS_ROOT)
    return sorted(
        path
        for root in roots
        for path in root.rglob("*.py")
    )


def compile_all(paths: list[Path]) -> list[GuardFailure]:
    failures: list[GuardFailure] = []
    for path in paths:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(GuardFailure(path=path, message=str(exc)))
    return failures


def scan_forbidden_stdlib_usage(paths: list[Path]) -> list[GuardFailure]:
    failures: list[GuardFailure] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_aliases: dict[str, str] = {}
        for node, importerror_guarded in walk_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    module_aliases[alias.asname or root] = root
                    if root in FORBIDDEN_IMPORT_MODULES and not importerror_guarded:
                        failures.append(
                            GuardFailure(
                                path=path,
                                message=f"imports stdlib module '{root}' that is unavailable on Python 3.10",
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.level != 0 or node.module is None:
                    continue
                root = node.module.split(".", 1)[0]
                if root in FORBIDDEN_IMPORT_MODULES and not importerror_guarded:
                    failures.append(
                        GuardFailure(
                            path=path,
                            message=f"imports from stdlib module '{root}' that is unavailable on Python 3.10",
                        )
                    )
                forbidden_names = FORBIDDEN_IMPORT_NAMES.get(root, set())
                for alias in node.names:
                    if alias.name in forbidden_names and not importerror_guarded:
                        failures.append(
                            GuardFailure(
                                path=path,
                                message=(
                                    f"imports stdlib symbol '{root}.{alias.name}' "
                                    "that is unavailable on Python 3.10"
                                ),
                            )
                        )
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                root = module_aliases.get(node.value.id)
                if root is None:
                    continue
                if node.attr in FORBIDDEN_IMPORT_NAMES.get(root, set()):
                    failures.append(
                        GuardFailure(
                            path=path,
                            message=(
                                f"references stdlib symbol '{root}.{node.attr}' "
                                "that is unavailable on Python 3.10"
                            ),
                        )
                    )
                elif root in FORBIDDEN_IMPORT_MODULES:
                    failures.append(
                        GuardFailure(
                            path=path,
                            message=f"references stdlib module '{root}' that is unavailable on Python 3.10",
                        )
                    )
    return failures


def walk_nodes(tree: ast.AST) -> list[tuple[ast.AST, bool]]:
    nodes: list[tuple[ast.AST, bool]] = []

    def visit(node: ast.AST, *, importerror_guarded: bool) -> None:
        nodes.append((node, importerror_guarded))
        if isinstance(node, ast.Try):
            guarded = catches_import_error(node)
            for child in node.body:
                visit(child, importerror_guarded=importerror_guarded or guarded)
            for child in node.handlers:
                visit(child, importerror_guarded=importerror_guarded)
            for child in node.orelse:
                visit(child, importerror_guarded=importerror_guarded)
            for child in node.finalbody:
                visit(child, importerror_guarded=importerror_guarded)
            return

        for child in ast.iter_child_nodes(node):
            visit(child, importerror_guarded=importerror_guarded)

    visit(tree, importerror_guarded=False)
    return nodes


def catches_import_error(node: ast.Try) -> bool:
    for handler in node.handlers:
        if handler.type is None:
            continue
        if isinstance(handler.type, ast.Name) and handler.type.id == "ImportError":
            return True
        if isinstance(handler.type, ast.Tuple):
            for element in handler.type.elts:
                if isinstance(element, ast.Name) and element.id == "ImportError":
                    return True
    return False


def top_level_external_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if is_external_import_root(root):
                    imports.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module is None:
                continue
            root = node.module.split(".", 1)[0]
            if is_external_import_root(root):
                imports.add(root)
    return imports


def is_external_import_root(root: str) -> bool:
    if root in ALLOWED_TOP_LEVEL_EXTERNAL_IMPORTS:
        return False
    if root in FIRST_PARTY_ROOTS:
        return False
    return root not in STDLIB_MODULES


def module_name_for_path(path: Path) -> str:
    if path.is_relative_to(SRC_ROOT):
        return ".".join(path.relative_to(SRC_ROOT).with_suffix("").parts)
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    return "_py310_smoke." + ".".join(relative.parts)


def import_target(path: Path) -> None:
    if path.is_relative_to(SRC_ROOT):
        importlib.invalidate_caches()
        importlib.import_module(module_name_for_path(path))
        return

    spec = importlib.util.spec_from_file_location(module_name_for_path(path), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise


class StubModule(ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__path__ = []

    def __getattr__(self, attr: str) -> object:
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        value = make_placeholder_type(f"{self.__name__}.{attr}")
        setattr(self, attr, value)
        return value


def make_placeholder_type(name: str) -> type:
    return type(name.rsplit(".", 1)[-1], (), {})


def install_stub_modules(module_names: set[str]) -> tuple[dict[str, ModuleType], set[str]]:
    created: dict[str, ModuleType] = {}
    created_roots: set[str] = set()
    for module_name in sorted(module_names):
        parts = module_name.split(".")
        for index in range(1, len(parts) + 1):
            partial_name = ".".join(parts[:index])
            if partial_name in sys.modules:
                module = sys.modules[partial_name]
            else:
                module = StubModule(partial_name)
                sys.modules[partial_name] = module
                created[partial_name] = module
                if index == 1:
                    created_roots.add(partial_name)
            if index > 1:
                parent_name = ".".join(parts[: index - 1])
                child_name = parts[index - 1]
                setattr(sys.modules[parent_name], child_name, module)
    return created, created_roots


def remove_stub_modules(created_modules: dict[str, ModuleType], created_roots: set[str]) -> None:
    for root_name in created_roots:
        sys.modules.pop(root_name, None)
    for module_name in sorted(created_modules, reverse=True):
        sys.modules.pop(module_name, None)


def import_target_with_stubbed_external_deps(path: Path, external_imports: set[str]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in external_imports:
                    module_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module is None:
                continue
            root = node.module.split(".", 1)[0]
            if root in external_imports:
                module_names.add(node.module)

    created_modules, created_roots = install_stub_modules(module_names or external_imports)
    try:
        import_target(path)
    finally:
        remove_stub_modules(created_modules, created_roots)


def run_import_smoke(paths: list[Path]) -> tuple[list[GuardFailure], list[str]]:
    failures: list[GuardFailure] = []
    skipped: list[str] = []
    for path in paths:
        external_imports = top_level_external_imports(path)
        try:
            import_target(path)
        except ModuleNotFoundError as exc:
            missing_root = exc.name.split(".", 1)[0] if exc.name else None
            if external_imports and missing_root in external_imports:
                try:
                    import_target_with_stubbed_external_deps(path, external_imports)
                    skipped.append(
                        f"{path.relative_to(REPO_ROOT)} (external import smoke used stubs for: {', '.join(sorted(external_imports))})"
                    )
                    continue
                except Exception as stub_exc:  # pragma: no cover - smoke error reporting path
                    failures.append(
                        GuardFailure(
                            path=path,
                            message=(
                                "import smoke failed even with external dependency stubs: "
                                f"{stub_exc!r}"
                            ),
                        )
                    )
                    continue
            failures.append(GuardFailure(path=path, message=f"import smoke failed: {exc!r}"))
        except Exception as exc:  # pragma: no cover - smoke error reporting path
            failures.append(GuardFailure(path=path, message=f"import smoke failed: {exc!r}"))
    return failures, skipped


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(SRC_ROOT))

    paths = iter_python_files()

    print("Running Python 3.10 compile smoke check")
    compile_failures = compile_all(paths)
    if compile_failures:
        report_failures(compile_failures)
        return 1

    print("Scanning for known Python 3.11+ stdlib usage")
    stdlib_failures = scan_forbidden_stdlib_usage(paths)
    if stdlib_failures:
        report_failures(stdlib_failures)
        return 1

    print("Running Python 3.10 import smoke")
    import_failures, skipped = run_import_smoke(paths)
    if skipped:
        print("Import smoke used reduced external-dependency coverage for:")
        for item in skipped:
            print(f"  - {item}")
    if import_failures:
        report_failures(import_failures)
        return 1

    print("Python 3.10 compatibility guardrails passed")
    return 0


def report_failures(failures: list[GuardFailure]) -> None:
    for failure in failures:
        rel_path = failure.path.relative_to(REPO_ROOT)
        print(f"{rel_path}: {failure.message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
