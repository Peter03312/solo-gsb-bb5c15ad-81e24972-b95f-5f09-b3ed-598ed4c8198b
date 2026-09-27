"""公式白名单解析、量纲校验与带单位数值求值。

白名单（按 GUM 校准常用运算选取）：
  四则运算 + - * / （含一元正负号）
  幂运算 **（幂次只能是无量纲常数），同时支持 ^
  sqrt, exp, log/ln, log10, log2
  sin, cos, tan, asin, acos, atan, abs
  常数 pi, E

量纲规则（静态校验，先于数值计算）：
  加减：两侧量纲必须一致；
  幂：base ** k，结果量纲为 base 量纲的 k 次方（sqrt 为 1/2 次方）；
  exp/log/三角与反三角函数：参数必须无量纲（角度用 rad/deg，弧度即无量纲）；
  绝对值：保留参数量纲。
"""
from __future__ import annotations

import ast
import math

import sympy as sp
from pint import DimensionalityError, OffsetUnitCalculusError

from .errors import UnprocessableError
from .units import ureg

Quantity = ureg.Quantity

# -- 白名单定义 -----------------------------------------------------------


class Log10(sp.Function):
    """以 10 为底的对数，自定义求导规则供符号偏导使用。"""

    def fdiff(self, argindex: int = 1) -> sp.Expr:
        return 1.0 / (self.args[0] * sp.log(10))


class Log2(sp.Function):
    """以 2 为底的对数。"""

    def fdiff(self, argindex: int = 1) -> sp.Expr:
        return 1.0 / (self.args[0] * sp.log(2))


# 名称 -> SymPy 调用对象（parse 时使用）。
ALLOWED_FUNCS: dict[str, sp.Expr] = {
    "sqrt": sp.sqrt,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "log10": Log10,
    "log2": Log2,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "abs": sp.Abs,
}

# SymPy 函数类 -> 规范名（区分 Log10/Log2）。
FUNC_CANON: dict[type, str] = {
    sp.exp: "exp",
    sp.log: "log",
    sp.sqrt: "sqrt",
    sp.sin: "sin",
    sp.cos: "cos",
    sp.tan: "tan",
    sp.asin: "asin",
    sp.acos: "acos",
    sp.atan: "atan",
    sp.Abs: "abs",
    sp.sign: "sign",  # 仅由 abs 的符号偏导内部产生
    Log10: "log10",
    Log2: "log2",
}

# 参数必须无量纲、结果也无量纲的函数。
DIMENSIONLESS_FUNCS = {
    "exp", "log", "log10", "log2",
    "sin", "cos", "tan", "asin", "acos", "atan", "sign",
}

CONSTANTS = {"pi": sp.pi, "E": sp.E, "e": sp.E}

RESERVED_NAMES = set(ALLOWED_FUNCS) | set(CONSTANTS)

WHITELIST_DISPLAY = [
    "+", "-", "*", "/", "** (^)",
    "sqrt", "exp", "log", "ln", "log10", "log2",
    "sin", "cos", "tan", "asin", "acos", "atan", "abs",
    "pi", "E",
]

_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.BitXor)  # ^ 经 convert_xor 转为幂
_UNARY_OPS = (ast.UAdd, ast.USub)


# -- AST 白名单校验 ---------------------------------------------------------


def _check_ast(tree: ast.AST) -> None:
    """遍历 AST，拒绝白名单以外的任何语法结构。"""
    allowed_funcs = ", ".join(sorted(set(ALLOWED_FUNCS) - {"ln"}))
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression):
            continue
        # BinOp/UnaryOp 的运算符节点本身、Name.ctx 也会被 walk 遍历到，直接放行。
        if isinstance(node, (ast.operator, ast.unaryop, ast.expr_context)):
            continue
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise UnprocessableError(
                    f"不允许的常量 {node.value!r}，仅允许数值常量",
                    ["formula"],
                    code="not_whitelisted",
                )
            continue
        if isinstance(node, ast.Name):
            # 函数名/常量名放行；输入量在解析后按声明清单校验。
            if node.id in ALLOWED_FUNCS or node.id in CONSTANTS or node.id.isidentifier():
                continue
            raise UnprocessableError(
                f"名称 {node.id!r} 不合法", ["formula"], code="invalid_name"
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, _BIN_OPS):
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _UNARY_OPS):
            continue
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise UnprocessableError(
                    "仅允许调用白名单内的函数", ["formula"], code="not_whitelisted"
                )
            fname = node.func.id
            if fname not in ALLOWED_FUNCS:
                raise UnprocessableError(
                    f"函数 {fname!r} 不在白名单中；允许的函数：{allowed_funcs}",
                    ["formula"],
                    code="not_whitelisted",
                )
            if len(node.args) != 1 or node.keywords:
                raise UnprocessableError(
                    f"函数 {fname} 仅接受 1 个位置参数",
                    ["formula"],
                    code="bad_arity",
                )
            continue
        raise UnprocessableError(
            f"公式中不允许的语法结构：{type(node).__name__}",
            ["formula"],
            code="not_whitelisted",
        )


def parse_formula(formula: str, input_names: set[str]) -> sp.Expr:
    """解析公式并返回 SymPy 表达式。

    校验：语法合法、仅白名单运算、出现的符号全部是已声明的输入量、
    幂次只能是数值常量（不允许 x**y 这种量纲无法静态确定的写法）。
    """
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise UnprocessableError(
            f"公式语法错误：{exc.msg}（第 {exc.lineno} 行第 {exc.offset} 列）",
            ["formula"],
            code="syntax_error",
        ) from exc

    _check_ast(tree)

    local_dict: dict[str, sp.Expr] = {name: sp.Symbol(name) for name in input_names}
    local_dict.update(CONSTANTS)
    local_dict.update(ALLOWED_FUNCS)

    # parse_expr 内部构造数字字面量/符号时需要这些原子类；
    # 此处不提供任何函数，函数调用已在 AST 层用白名单限定。
    global_dict = {
        "Integer": sp.Integer,
        "Float": sp.Float,
        "Rational": sp.Rational,
        "Symbol": sp.Symbol,
    }
    try:
        expr = sp.parse_expr(
            formula,
            local_dict=local_dict,
            global_dict=global_dict,
            transformations=(
                sp.parsing.sympy_parser.standard_transformations
                + (sp.parsing.sympy_parser.convert_xor,)
            ),
            evaluate=True,
        )
    except Exception as exc:  # parse_expr 可能抛多种异常
        raise UnprocessableError(
            f"公式解析失败：{exc}", ["formula"], code="parse_error"
        ) from exc

    for atom in expr.free_symbols:
        if atom.name not in input_names:
            raise UnprocessableError(
                f"公式引用了未声明的输入量 {atom.name!r}",
                ["formula"],
                code="unknown_symbol",
            )

    for pow_node in expr.atoms(sp.Pow):
        if not pow_node.exp.is_number:
            raise UnprocessableError(
                f"幂次 {pow_node.exp} 不是数值常量；幂运算的指数必须为无量纲常数"
                f"（变量请改用乘除或 sqrt）",
                ["formula"],
                code="non_constant_exponent",
            )

    return expr


# -- 量纲静态校验 -----------------------------------------------------------

_NO_DIM: dict = ureg.dimensionless.dimensionality


def _canon_func_name(func: sp.Expr) -> str:
    name = FUNC_CANON.get(func)
    if name is None:
        raise UnprocessableError(
            f"函数 {getattr(func, '__name__', func)!r} 不在白名单中",
            ["formula"],
            code="not_whitelisted",
        )
    return name


def _scale_dim(dim: dict, exponent: float) -> dict:
    return {k: v * exponent for k, v in dim.items() if abs(v * exponent) > 1e-12}


def _mul_dims(a: dict, b: dict) -> dict:
    result = dict(a)
    for key, value in b.items():
        result[key] = result.get(key, 0) + value
    return {k: v for k, v in result.items() if abs(v) > 1e-12}


def dimension_of(expr: sp.Expr, symbol_dims: dict[sp.Symbol, dict]) -> dict:
    """递归求表达式的 Pint 量纲（dimensionality 字典），并做量纲一致性校验。"""
    if isinstance(expr, sp.Symbol):
        return symbol_dims[expr]
    if expr.is_number and not expr.free_symbols:
        return _NO_DIM
    if isinstance(expr, sp.Pow):
        base_dim = dimension_of(expr.base, symbol_dims)
        if not expr.exp.is_number:
            raise UnprocessableError(
                "幂次必须为无量纲常数", ["formula"], code="non_constant_exponent"
            )
        return _scale_dim(base_dim, float(expr.exp))
    if isinstance(expr, sp.Add):
        ref = dimension_of(expr.args[0], symbol_dims)
        for arg in expr.args[1:]:
            dim = dimension_of(arg, symbol_dims)
            if dim != ref:
                raise UnprocessableError(
                    f"量纲不一致：无法将量纲 {ref} 与 {dim} 的量相加减"
                    f"（子表达式 {arg}）",
                    ["formula"],
                    code="dimension_mismatch",
                )
        return ref
    if isinstance(expr, sp.Mul):
        result = dict(_NO_DIM)
        for arg in expr.args:
            result = _mul_dims(result, dimension_of(arg, symbol_dims))
        return result
    if expr.is_Function:
        name = _canon_func_name(expr.func)
        arg_dim = dimension_of(expr.args[0], symbol_dims)
        if name in DIMENSIONLESS_FUNCS:
            if arg_dim != _NO_DIM:
                raise UnprocessableError(
                    f"函数 {name} 的参数必须无量纲，实际量纲为 {arg_dim}"
                    f"（角度请使用 rad 或 deg）",
                    ["formula"],
                    code="dimension_mismatch",
                )
            return _NO_DIM
        if name == "sqrt":
            return _scale_dim(arg_dim, 0.5)
        if name == "abs":
            return arg_dim
        raise UnprocessableError(
            f"函数 {name} 不在白名单中", ["formula"], code="not_whitelisted"
        )
    raise UnprocessableError(
        f"无法确定表达式 {expr} 的量纲", ["formula"], code="unsupported_expression"
    )


# -- 带单位数值求值 ---------------------------------------------------------


def _as_constant(expr: sp.Expr) -> Quantity:
    return Quantity(float(expr), ureg.dimensionless)


def _to_radians_number(q: Quantity, name: str) -> float:
    """取无量纲量的数值；角度单位（rad/deg）统一换算为弧度。"""
    try:
        return float(q.to(ureg.radian).magnitude)
    except DimensionalityError as exc:
        raise UnprocessableError(
            f"函数 {name} 的参数必须无量纲，实际为 {q.units:~}",
            ["formula"],
            code="dimension_mismatch",
        ) from exc


def _to_dimless_number(q: Quantity, name: str) -> float:
    try:
        return float(q.to(ureg.dimensionless).magnitude)
    except DimensionalityError as exc:
        raise UnprocessableError(
            f"函数 {name} 的参数必须无量纲，实际为 {q.units:~}",
            ["formula"],
            code="dimension_mismatch",
        ) from exc


def _pow_quantity(base: Quantity, exponent: float) -> Quantity:
    try:
        return base ** exponent
    except DimensionalityError as exc:
        raise UnprocessableError(
            f"对带单位的量取 {exponent} 次幂不满足量纲自洽性：{exc}",
            ["formula"],
            code="dimension_mismatch",
        ) from exc
    except (ZeroDivisionError, ValueError, OffsetUnitCalculusError) as exc:
        raise UnprocessableError(
            "公式求值出现除零、零的负次幂或偏移单位（如 degC）上的非法运算",
            ["formula"],
            code="evaluation_error" if not isinstance(exc, OffsetUnitCalculusError) else "offset_unit",
        ) from exc


def _eval_function(expr: sp.Expr, quantities: dict[sp.Symbol, Quantity]) -> Quantity:
    name = _canon_func_name(expr.func)
    arg = evaluate(expr.args[0], quantities)

    if name == "sqrt":
        try:
            return _pow_quantity(arg, 0.5)
        except ValueError as exc:
            raise UnprocessableError(
                "sqrt 的参数不能为负", ["formula"], code="evaluation_error"
            ) from exc
    if name == "abs":
        return Quantity(abs(float(arg.magnitude)), arg.units)

    if name in ("sin", "cos", "tan", "asin", "acos", "atan"):
        trig = {
            "sin": (math.sin, True, False),
            "cos": (math.cos, True, False),
            "tan": (math.tan, True, False),
            "asin": (math.asin, False, True),
            "acos": (math.acos, False, True),
            "atan": (math.atan, False, True),
        }
        fn, takes_angle, returns_angle = trig[name]
        try:
            if takes_angle:
                x = _to_radians_number(arg, name)
            else:
                x = _to_dimless_number(arg, name)
            y = fn(x)
        except ValueError as exc:
            raise UnprocessableError(
                f"函数 {name} 的参数超出定义域", ["formula"], code="evaluation_error"
            ) from exc
        return Quantity(y, ureg.radian if returns_angle else ureg.dimensionless)

    # exp / 对数族。
    try:
        x = _to_dimless_number(arg, name)
        if name == "exp":
            y = math.exp(x)
        elif name == "log":
            y = math.log(x)
        elif name == "log10":
            y = math.log10(x)
        elif name == "log2":
            y = math.log2(x)
        else:  # sign（abs 的偏导内部产生）
            y = 0.0 if x == 0.0 else math.copysign(1.0, x)
    except ValueError as exc:
        raise UnprocessableError(
            f"函数 {name} 的参数超出定义域（需为正数）",
            ["formula"],
            code="evaluation_error",
        ) from exc
    return Quantity(y, ureg.dimensionless)


def evaluate(expr: sp.Expr, quantities: dict[sp.Symbol, Quantity]) -> Quantity:
    """以 Pint 量为操作数递归求值，全程保持量纲一致。"""
    if isinstance(expr, sp.Symbol):
        return quantities[expr]
    if expr.is_number and not expr.free_symbols:
        return _as_constant(expr)
    if expr.is_Function:
        return _eval_function(expr, quantities)
    if isinstance(expr, sp.Pow):
        base = evaluate(expr.base, quantities)
        return _pow_quantity(base, float(expr.exp))
    if isinstance(expr, sp.Mul):
        result = Quantity(1.0, ureg.dimensionless)
        try:
            for arg in expr.args:
                result = result * evaluate(arg, quantities)
        except OffsetUnitCalculusError as exc:
            raise UnprocessableError(
                f"偏移单位（如摄氏度 degC）不允许直接乘除：{exc}；请先换算为 K",
                ["formula"],
                code="offset_unit",
            ) from exc
        return result
    if isinstance(expr, sp.Add):
        terms = [evaluate(arg, quantities) for arg in expr.args]
        result = terms[0]
        try:
            for term in terms[1:]:
                result = result + term
        except DimensionalityError as exc:
            raise UnprocessableError(
                f"量纲不一致，无法相加：{exc}", ["formula"], code="dimension_mismatch"
            ) from exc
        except OffsetUnitCalculusError as exc:
            raise UnprocessableError(
                f"涉及偏移单位（如摄氏度 degC）的运算存在歧义：{exc}；"
                f"请先换算为开尔文 K 等绝对单位后再计算",
                ["formula"],
                code="offset_unit",
            ) from exc
        return result
    raise UnprocessableError(
        f"暂不支持的表达式结构：{expr}", ["formula"], code="unsupported_expression"
    )
