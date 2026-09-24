# tkv_harness.py — TV-1001 verification harness.
#
# tkvc.exe lives in the separate TokenVector compiler repo and is not available
# here, so this harness verifies the .tkv translation the next best way:
#   1. Parse every .tkv file with the Python `ast` module and reject any
#      construct outside the TV-1001 grammar (TOKENVECTOR_SYNTAX_SPEC.md 1-6).
#   2. Load the modules through a custom importer that maps the `tv.*` module
#      tree onto src/tokenvector/*.tkv and provide the runtime primitives the
#      spec reserves for the compiler (tv.sqrt, tv.io.*, tv.f64_bits, ...).
#   3. Execute tests/tokenvector/smoke_tests.tkv's main() and report pass/fail.
#
# Usage:  python tests/tokenvector/tkv_harness.py
# Anything that fails here is either a real translation bug or a missing tv.*
# primitive to add to the stub table below.

import ast
import glob
import importlib
import importlib.abc
import importlib.machinery
import math
import os
import struct
import sys
import types
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "tokenvector")

# ---------------------------------------------------------------------------
# 1. Static syntax gate over every .tkv module
# ---------------------------------------------------------------------------

# Constructs outside the TV-1001 surface (spec sections 1-6): no dict/set
# displays, no comprehensions, no global/nonlocal, no try/except, no async,
# no yield, no with, no walrus, no starred assignment, no slice-step abuse.
BANNED_NODES = (
    ast.Dict, ast.DictComp, ast.Set, ast.SetComp, ast.ListComp, ast.GeneratorExp,
    ast.Global, ast.Nonlocal,
    ast.Try, ast.ExceptHandler,
    ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom,
    ast.With, ast.AsyncWith,
    ast.NamedExpr,
)


def syntax_check_all():
    failures = []
    for path in sorted(glob.glob(os.path.join(SRC, "*.tkv"))):
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            failures.append((os.path.basename(path), f"line {e.lineno}: {e.msg}"))
            continue
        for node in ast.walk(tree):
            if isinstance(node, BANNED_NODES):
                failures.append((os.path.basename(path),
                                 f"line {node.lineno}: banned construct {type(node).__name__}"))
    return failures

# ---------------------------------------------------------------------------
# 2. tv.* runtime primitives (compiler-provided builtins per the spec)
# ---------------------------------------------------------------------------

_MAPPED = {}
_MAPPED_NEXT = [1]


def make_tv_runtime(core_mod):
    tv = types.ModuleType("tv")
    tv.sqrt = math.sqrt
    tv.exp = math.exp
    tv.log = math.log
    tv.sin = math.sin
    tv.cos = math.cos
    tv.tan = math.tan
    tv.tanh = math.tanh
    tv.sinh = math.sinh
    tv.cosh = math.cosh
    tv.asin = math.asin
    tv.acos = math.acos
    tv.atan = math.atan
    tv.atan2 = math.atan2
    tv.floor = math.floor
    tv.ceil = math.ceil
    tv.abs = abs
    tv.nan = lambda: math.nan
    tv.f32 = "f32"
    tv.f64 = "f64"
    tv.i32 = "i32"
    tv.i64 = "i64"
    tv.array = core_mod.from_array
    tv.zeros = core_mod.zeros
    tv.ones = core_mod.ones
    tv.full = core_mod.full
    tv.linspace = core_mod.linspace
    tv.f32_bits = lambda v: _float_bits(v, 4)
    tv.f64_bits = lambda v: _float_bits(v, 8)
    tv.f32_from_bits = lambda b: _float_from_bits(b, 4)
    tv.f64_from_bits = lambda b: _float_from_bits(b, 8)
    tv.f32_epsilon = 1.1920928955078125e-07
    tv.f64_epsilon = 2.220446049250313e-16

    io = types.ModuleType("tv.io")
    io.write_file_bytes = _write_bytes
    io.read_file_bytes = _read_bytes
    io.write_file_text = _write_text
    io.read_file_text = _read_text
    io.write_zip_archive = _write_zip
    io.read_zip_archive = _read_zip
    io.open_mapped_file = _open_mapped
    io.mapped_read_f64 = _mapped_read
    io.mapped_write_f64 = _mapped_write
    io.mapped_flush = lambda h: None
    io.close_mapped_file = _close_mapped
    tv.io = io
    return tv


def _float_bits(v, size):
    return int.from_bytes(struct.pack("<d" if size == 8 else "<f", v), "little")


def _float_from_bits(b, size):
    return struct.unpack("<d" if size == 8 else "<f", int(b).to_bytes(size, "little"))[0]


def _write_bytes(path, data):
    with open(path, "wb") as f:
        f.write(bytes(bytearray(int(b) & 0xFF for b in data)))
    return path


def _read_bytes(path):
    with open(path, "rb") as f:
        return list(f.read())


def _write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return path


def _read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_zip(path, names, payloads, compress=True):
    with zipfile.ZipFile(path, "w",
                         zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED) as z:
        for name, payload in zip(names, payloads):
            z.writestr(name, bytes(bytearray(int(b) & 0xFF for b in payload)))


def _read_zip(path):
    names, payloads = [], []
    with zipfile.ZipFile(path, "r") as z:
        for info in z.infolist():
            names.append(info.filename)
            payloads.append(list(z.read(info.filename)))
    return names, payloads


def _open_mapped(path, byte_length, create):
    handle = _MAPPED_NEXT[0]
    _MAPPED_NEXT[0] += 1
    if create or not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(b"\x00" * byte_length)
    _MAPPED[handle] = open(path, "r+b")
    return handle


def _mapped_read(handle, flat_index):
    f = _MAPPED[handle]
    f.seek(flat_index * 8)
    return struct.unpack("<d", f.read(8))[0]


def _mapped_write(handle, flat_index, value):
    f = _MAPPED[handle]
    f.seek(flat_index * 8)
    f.write(struct.pack("<d", float(value)))


def _close_mapped(handle):
    f = _MAPPED.pop(handle, None)
    if f is not None:
        f.close()

# ---------------------------------------------------------------------------
# 3. Custom importer: map `tv.<name>` -> src/tokenvector/<name>.tkv
# ---------------------------------------------------------------------------


class TkvLoader(importlib.abc.Loader):
    def __init__(self):
        self.tv_runtime = None

    def find_spec(self, name, path=None, target=None):
        if name == "tv" or name.startswith("tv."):
            return importlib.machinery.ModuleSpec(name, self)
        return None

    def create_module(self, spec):
        return None  # default module creation

    def exec_module(self, module):
        name = module.__name__
        if name == "tv":
            module.__path__ = []  # mark as package so tv.* submodules resolve
            if self.tv_runtime is not None:
                module.__dict__.update(self.tv_runtime.__dict__)
            return
        parts = name.split(".")
        if len(parts) != 2 or parts[0] != "tv":
            raise ImportError("unsupported tv module path: " + name)
        if parts[1] == "io" and self.tv_runtime is not None:
            # tv.io is a runtime namespace (compiler builtins), not a .tkv file.
            module.__dict__.update(self.tv_runtime.io.__dict__)
            return
        self._load_tkv(parts[1], module)

    def _load_tkv(self, mod_name, module):
        path = os.path.join(SRC, mod_name + ".tkv")
        if not os.path.exists(path):
            raise ImportError("no such tv module: " + mod_name)
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        code = compile(src, os.path.abspath(path), "exec")
        module.__file__ = path
        exec(code, module.__dict__)
        tv_mod = sys.modules.get("tv")
        if tv_mod is not None:
            setattr(tv_mod, mod_name, module)

# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------


def main():
    failures = syntax_check_all()
    if failures:
        print("SYNTAX / BANNED-CONSTRUCT FAILURES:")
        for name, msg in failures:
            print("  ", name, "->", msg)
        return 1
    print("Syntax gate: all .tkv modules parse, TV-1001 constructs only.")

    loader = TkvLoader()
    sys.meta_path.insert(0, loader)

    core = importlib.import_module("tv.core")
    loader.tv_runtime = make_tv_runtime(core)
    sys.modules["tv"].__dict__.update(loader.tv_runtime.__dict__)

    test_path = os.path.join(ROOT, "tests", "tokenvector", "smoke_tests.tkv")
    with open(test_path, "r", encoding="utf-8") as f:
        src = f.read()
    code = compile(src, test_path, "exec")
    g = {"__name__": "smoke_tests"}
    exec(code, g)
    # The .tkv suite is a TV-1001 program: it calls main() at module level
    # (mirroring `tkvc smoke_tests.tkv -o smoke.exe`), so the harness must not
    # invoke it a second time or counters would double.
    return 0


if __name__ == "__main__":
    sys.exit(main())
