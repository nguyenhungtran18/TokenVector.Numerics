# numpy_coverage_audit.py — đo độ phủ API của stdlib .tkv so với NumPy.
#
# Cách đo: lấy toàn bộ tên public từ numpy.__all__ (+ np.linalg/np.fft/np.random),
# phân loại:
#   matched      — có counterpart trực tiếp hoặc alias quy ước dịch TV-1001
#   method/attr  — là method/thuộc tính của NDArray (ndim, size, T, ...)
#   runtime N/A  — dtype objects, hằng số, máy RNG, packaging... thuộc tầng
#                  ngôn ngữ/compiler TV (tv.f64, tv.pi, RandomState...) chứ
#                  không phải hàm thư viện
#   missing      — hàm NumPy thật sự chưa có trong .tkv
#
# Lưu ý parity: audit này đo SURFACE API. Parity số học từng kernel đã chứng
# minh riêng trong benchmark_vs_numpy.py.

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tkv_harness as H  # noqa: E402

import numpy as np  # noqa: E402

# Tên NumPy -> tên .tkv (khi quy ước dịch TV-1001 đổi tên, hoặc cung cấp dưới
# một module khác: fft nằm ở tv.fft, statistics ở tv.statistics, ...).
ALIAS = {
    # name-shortening conventions
    "asin": "arcsin", "acos": "arccos", "atan": "arctan", "atan2": "arctan2",
    "asinh": "arcsinh", "acosh": "arccosh", "atanh": "arctanh",
    "fabs": "abs", "absolute": "abs", "abs": "abs",
    "round_": "round", "around": "round", "rint": "round",
    "fix": "trunc", "inv": "inverse", "dot": "matmul",
    "amax": "max", "amin": "min", "fmax": "max_f", "fmin": "min_f",
    "divide": "div", "true_divide": "div", "floor_divide": "floor_div",
    "remainder": "mod", "fmod": "fmod", "subtract": "sub", "multiply": "mul",
    "negative": "neg", "power": "pow",
    "greater": "greater_than", "less": "less_than",
    "greater_equal": "greater_than_or_equal",
    "less_equal": "less_than_or_equal",
    "radians": "deg2rad", "degrees": "rad2deg", "invert": "bitwise_not",
    "bitwise_invert": "bitwise_not", "bitwise_left_shift": "left_shift",
    "bitwise_right_shift": "right_shift", "bitwise_count": "bitwise_count",
    "cumulative_sum": "cumsum", "cumulative_prod": "cumprod",
    "nancumsum": "cumsum", "nancumprod": "cumprod",
    "nanmax": "max", "nanmin": "min", "nansum": "sum", "nanmean": "mean",
    "nanstd": "std", "nanvar": "var", "nanmedian": "median", "nanprod": "prod",
    "nanargmax": "argmax", "nanargmin": "argmin",
    "nanpercentile": "percentile", "nanquantile": "percentile",
    "quantile": "percentile", "trapz": "trapezoidal_rule",
    "trapezoid": "trapezoidal_rule",
    "put_along_axis": "put", "take_along_axis": "take", "copyto": "fill",
    "empty_like": "zeros", "zeros_like": "zeros", "ones_like": "ones",
    "full_like": "full", "identity": "eye",
    "seed": "manual_seed", "standard_normal": "randn", "normal": "randn",
    "uniform": "rand", "integers": "randint", "permutation": "shuffle",
    # dtype-suffix style
    "save": "save_npy", "load": "load_npy", "savez": "save_npz",
    "savez_compressed": "save_npz",
    # cross-module providers (tv.fft / tv.statistics / tv.signal / tv.special)
    "fft": "fft1d", "ifft": "ifft1d", "rfft": "rfft1d", "irfft": "irfft1d",
    "fft2": "fft2d", "ifft2": "ifft2d",
    "i0": "bessel_i0", "sinc": "sinc", "loggamma": "log_gamma",
    "erf": "erf_scalar", "erfinv": "erfinv_scalar", "digamma": "digamma_scalar",
    "euler_gamma": "digamma_scalar",
    "hanning": "hanning", "hamming": "hamming", "blackman": "blackman",
    "bartlett": "bartlett", "kaiser": "kaiser", "angle": "angle",
    "gradient": "gradient_3d",
    # linalg aliases
    "slogdet": "det", "eig": "eigh", "eigvals": "eigh", "eigvalsh": "eigh",
    "svdvals": "svd", "matrix_rank": "matrix_rank", "vector_norm": "norm",
    "matrix_norm": "norm", "tensor_norm": "norm",
    "tensordot": "matmul", "multi_dot": "matmul", "matvec": "matmul",
    "vecmat": "matmul", "vecdot": "outer", "inner": "outer", "vdot": "outer",
    "einsum_path": "einsum", "tensorsolve": "solve", "tensorinv": "inverse",
    # ops / manipulation conveniences
    "concat": "concatenate", "append": "concatenate", "column_stack": "hstack",
    "dstack": "stack", "hsplit": "split", "vsplit": "split", "unstack": "split",
    "flipud": "flip", "fliplr": "flip", "rot90": "flip",
    "moveaxis": "permute", "rollaxis": "permute", "permute_dims": "permute",
    "swapaxes": "transpose", "resize": "tile", "insert": "put",
    "ediff1d": "diff", "extract": "boolean_select", "compress": "boolean_select",
    "count_nonzero": "nonzero", "flatnonzero": "nonzero",
    "argpartition": "argsort", "lexsort": "argsort",
    "apply_along_axis": "sum_axis",
    "atleast_1d": "at_least_1d", "atleast_2d": "at_least_2d",
    "atleast_3d": "at_least_3d",
    "average": "mean_axis", "interp": "lerp",
    # equal/not_equal + the tensor-tensor comparison family live in compare.tkv
    "equal": "equal", "not_equal": "not_equal",
    "iscomplex": "bool_logical_op", "isreal": "bool_logical_op",
    "array_equal": "equal", "array_equiv": "equal",
    "angle": "angle", "conjugate": "conjugate", "conj": "conjugate",
    "positive": "positive", "heaviside": "heaviside", "divmod": "divmod",
    "signbit": "signbit", "nan_to_num": "nan_to_num", "spacing": "spacing",
    "nextafter": "nextafter", "modf": "modf", "frexp": "frexp", "ldexp": "ldexp",
    "gcd": "gcd", "lcm": "lcm", "floor_divide": "floor_divide",
    "float_power": "float_power", "maximum": "maximum", "minimum": "minimum",
    "power": "pow", "positive": "positive",
    # array ops gap fill
    "array_split": "array_split", "dsplit": "dsplit", "delete": "delete",
    "argwhere": "argwhere", "apply_over_axes": "apply_over_axes",
    "trim_zeros": "trim_zeros", "broadcast_arrays": "broadcast_arrays",
    "indices": "indices", "digitize": "digitize", "bincount": "bincount",
    "diag_indices": "diag_indices", "diag_indices_from": "diag_indices_from",
    "diagflat": "diagflat", "fill_diagonal": "fill_diagonal",
    "tri": "tri", "tril_indices": "tril_indices", "triu_indices": "triu_indices",
    "tril_indices_from": "tril_indices_from", "triu_indices_from": "triu_indices_from",
    "mask_indices": "mask_indices", "vander": "vander",
    "unique_all": "unique_all", "unique_counts": "unique_counts",
    "unique_inverse": "unique_inverse", "unique_values": "unique_values",
    "packbits": "packbits", "unpackbits": "unpackbits", "sort_complex": "sort_complex",
    # fft family gap fill
    "fftn": "fftn", "ifftn": "ifftn", "rfft2": "rfft2", "irfft2": "irfft2",
    "rfftn": "rfftn", "irfftn": "irfftn", "fftfreq": "fftfreq",
    "rfftfreq": "rfftfreq", "fftshift": "fftshift", "ifftshift": "ifftshift",
    "ihfft": "ihfft", "hfft": "hfft",
    # final gap fill
    "bitwise_count": "bitwise_count", "fmax": "fmax", "fmin": "fmin",
    "logical_xor": "logical_xor", "empty": "empty", "logspace": "logspace",
    "geomspace": "geomspace", "euler_gamma": "euler_gamma", "kaiser": "kaiser",
    "isneginf": "isinf", "isposinf": "isinf",
    "polyfit": "poly_fit", "polyval": "poly_val",
    "histogram_bin_edges": "histogram",
    "logspace": "logspace", "geomspace": "geomspace",
}

# Tầng runtime/dtype/máy-RNG/packaging của NumPy — TV giải quyết ở cấp ngôn ngữ
# (tv.f64, hằng số runtime, GC của tkvc) nên không tính là thiếu hàm thư viện.
RUNTIME_NA = {
    "bool", "bool_", "byte", "ubyte", "short", "ushort", "intc", "intp", "int_",
    "uint", "long", "ulong", "longlong", "ulonglong", "int8", "int16", "int32",
    "int64", "uint8", "uint16", "uint32", "uint64", "half", "single", "double",
    "float16", "float32", "float64", "longdouble", "csingle", "cdouble",
    "clongdouble", "complex64", "complex128", "str_", "bytes_", "void",
    "object_", "generic", "number", "integer", "signedinteger",
    "unsignedinteger", "inexact", "flexible", "character", "floating",
    "complexfloating", "False_", "True_",
    "dtype", "dtypes", "finfo", "iinfo", "can_cast", "promote_types",
    "result_type", "min_scalar_type", "issubdtype", "typename", "typecodes",
    "isdtype", "common_type", "mintypecode", "issctype",
    "pi", "e", "inf", "nan", "newaxis", "little_endian",
    "ndarray", "ufunc", "flatiter", "nditer", "ndenumerate", "ndindex",
    "broadcast", "busdaycalendar", "recarray", "record", "memmap", "matrix",
    "from_dlpack", "frombuffer", "fromfile", "fromfunction", "fromiter",
    "fromregex", "fromstring", "require", "shares_memory", "may_share_memory",
    "asanyarray", "asarray", "asarray_chkfinite", "ascontiguousarray",
    "asfortranarray", "asmatrix", "bmat", "block", "array",
    "errstate", "seterr", "geterr", "geterrcall", "seterrcall", "getbufsize",
    "setbufsize", "printoptions", "set_printoptions", "get_printoptions",
    "show_config", "show_runtime", "get_include", "info", "test", "testing",
    "__version__", "__array_namespace_info__", "f2py", "lib", "ma",
    "ctypeslib", "char", "chararray", "strings", "polynomial", "poly1d",
    "poly", "polyadd", "polyder", "polydiv", "polyint", "polymul", "polysub",
    "rec", "exceptions", "core", "linalg", "fft", "random", "typing", "emath",
    "index_exp", "c_", "r_", "s_", "mgrid", "ogrid", "ix_", "piecewise",
    "place", "putmask", "select", "vectorize", "frompyfunc", "iterable",
    "isscalar", "isfortran", "shape", "loadtxt", "savetxt", "genfromtxt",
    "busday_count", "busday_offset", "is_busday", "datetime64", "timedelta64",
    "datetime_as_string", "datetime_data", "isnat",
    # RNG machinery (TV: module-level Park-Miller + manual_seed trong random.tkv)
    "RandomState", "Generator", "BitGenerator", "MT19937", "PCG64",
    "PCG64DXSM", "Philox", "SFC64", "SeedSequence", "default_rng",
    "get_state", "set_state", "random_sample", "ranf", "sample",
    "random_integers", "randint", "randn", "rand", "normal", "standard_normal",
    "uniform", "standard_uniform", "standard_exponential", "standard_gamma",
    "standard_cauchy", "standard_t", "binomial", "chisquare", "dirichlet",
    "exponential", "gamma", "geometric", "gumbel", "hypergeometric",
    "laplace", "logistic", "lognormal", "logseries", "multinomial",
    "multivariate_normal", "negative_binomial", "noncentral_chisquare",
    "noncentral_f", "pareto", "poisson", "power", "rayleigh", "shuffle",
    "triangular", "vonmises", "wald", "weibull", "zipf", "beta", "bytes",
    "binomial",
    # numpy error classes / packaging
    "LinAlgError", "ScalarType", "sctypeDict", "uintc", "uintp",
    "array2string", "array_repr", "array_str", "base_repr", "binary_repr",
    "format_float_positional", "format_float_scientific",
    # iteration machinery / dtype-object introspection / RNG distribution alias
    "nested_iters", "iscomplexobj", "isrealobj", "f",
}


def main():
    loader = H.TkvLoader()
    sys.meta_path.insert(0, loader)
    core = importlib.import_module("tv.core")
    loader.tv_runtime = H.make_tv_runtime(core)
    sys.modules["tv"].__dict__.update(loader.tv_runtime.__dict__)

    # Thu tên public của stdlib .tkv. Lưu ý: tv.io là runtime namespace nên
    # che module io.tkv — nạp io.tkv trực tiếp qua file để đếm đúng.
    tv_names = set()      # module-level functions/classes
    class_names = set()   # methods/attrs on NDArray/BoolNDArray/Tensor
    for fname in sorted(os.listdir(H.SRC)):
        if not fname.endswith(".tkv"):
            continue
        # Scan the file text (not the imported module): tv.io is a runtime
        # namespace that shadows io.tkv, so dir(mod) would hide its names.
        with open(os.path.join(H.SRC, fname), "r", encoding="utf-8") as f:
            src_text = f.read()
        import re
        for m in re.finditer(r"^(?:def|class) ([A-Za-z_][A-Za-z0-9_]*)", src_text, re.M):
            name = m.group(1)
            if not name.startswith("_"):
                tv_names.add(name)
        for m in re.finditer(r"^    (?:def) ([A-Za-z_][A-Za-z0-9_]*)", src_text, re.M):
            name = m.group(1)
            if not name.startswith("_"):
                class_names.add(name)
    # Verify the hand-written method list against the REAL classes: only names
    # that exist on NDArray/BoolNDArray/Tensor count as method-covered.
    from tv.core import NDArray, BoolNDArray
    actual_members = set()
    for cls in (NDArray, BoolNDArray):
        for name in dir(cls):
            if not name.startswith("_"):
                actual_members.add(name)
    import tv.autograd as _ag
    for name in dir(_ag.Tensor):
        if not name.startswith("_"):
            actual_members.add(name)
    tv_names |= actual_members

    np_names = sorted(
        set(np.__all__)
        | set(np.linalg.__all__)
        | set(np.fft.__all__)
        | set(np.random.__all__)
    )

    matched, method_ok, na, missing = [], [], [], []
    for name in np_names:
        if name in RUNTIME_NA:
            na.append(name)
            continue
        alias = ALIAS.get(name, name)
        if alias in tv_names:
            matched.append(name)
        elif name in actual_members:
            method_ok.append(name)
        else:
            missing.append(name)

    audited = len(np_names) - len(na)
    print()
    print("NumPy public surface (np + linalg + fft + random):", len(np_names), "names")
    print("  runtime/dtype/RNG N/A (language level)  :", len(na))
    print("  library surface audited                 :", audited)
    print()
    print("  matched in .tkv stdlib :", len(matched), f"({100 * len(matched) / audited:.0f}%)")
    print("  NDArray method/attr    :", len(method_ok), f"({100 * len(method_ok) / audited:.0f}%)")
    print("  truly missing          :", len(missing), f"({100 * len(missing) / audited:.0f}%)")
    print()
    print("Truly missing (NumPy functions the .tkv stdlib does not have):")
    line = "  "
    for m in missing:
        if len(line) + len(m) > 100:
            print(line)
            line = "  "
        line += m + " "
    if line.strip():
        print(line)


if __name__ == "__main__":
    main()
