"""不确定度传播主流程：符号偏导 + 协方差矩阵（GUM 一阶近似）。

对测量模型 y = f(x1, ..., xn)，合成标准不确定度：

    u_c^2 = Σ_i c_i^2 u(x_i)^2
          + 2 Σ_{i<j} c_i c_j u(x_i) u(x_j) r(x_i, x_j)

其中 c_i = ∂f/∂x_i 在输入估计值处取值。

``propagate`` 直接给出合成结果；``evaluate_model`` /
``variance_from_cu`` 供不确定度逆分配（allocation）复用同一条公式、
同一套单位与相关矩阵校验。
"""
from __future__ import annotations

import math

import sympy as sp

from .correlation import validate_correlation_matrix
from .errors import UnprocessableError
from .formula import RESERVED_NAMES, dimension_of, evaluate, parse_formula
from .units import (
    Quantity,
    convert_magnitude,
    parse_unit,
    preferred_unit,
    same_dimensionality,
    unit_str,
)

WHITELIST = [
    "+", "-", "*", "/", "** (^)",
    "sqrt", "exp", "log", "ln", "log10", "log2",
    "sin", "cos", "tan", "asin", "acos", "atan", "abs",
    "pi", "E",
]


def _prepare_inputs(req, allow_missing_uncertainty: bool = False) -> list[dict]:
    """解析每个输入量的单位与不确定度，返回内部表示列表。

    ``allow_missing_uncertainty=True`` 时（逆分配请求），允许不确定度缺省，
    对应项 ``u_magnitude`` 为 ``None``，由调用方决定其取值。
    """
    seen: set[str] = set()
    prepared: list[dict] = []

    for index, item in enumerate(req.inputs):
        loc = ["inputs", index]
        if item.name in seen:
            raise UnprocessableError(
                f"输入量名称 {item.name!r} 重复", loc + ["name"], code="duplicate_name"
            )
        if item.name in RESERVED_NAMES:
            raise UnprocessableError(
                f"输入量名称 {item.name!r} 是白名单保留名称，请更换",
                loc + ["name"],
                code="reserved_name",
            )
        seen.add(item.name)

        unit = parse_unit(item.unit, loc + ["unit"])
        q = Quantity(float(item.value), unit)

        su = item.standard_uncertainty
        if su is None:
            if not allow_missing_uncertainty:
                # 常规传播请求中 schema 已保证非空；此处仅作防御。
                raise UnprocessableError(
                    f"输入量 {item.name!r} 缺少标准不确定度",
                    loc + ["standard_uncertainty"],
                    code="missing_uncertainty",
                )
            u_mag = None
        elif isinstance(su, (int, float)):
            u_mag, u_unit = float(su), unit
        else:
            if su.unit is None:
                u_mag, u_unit = float(su.value), unit
            else:
                u_unit = parse_unit(su.unit, loc + ["standard_uncertainty", "unit"])
                if not same_dimensionality(u_unit, unit):
                    raise UnprocessableError(
                        f"不确定度单位 {u_unit:~} 与输入量单位 {unit:~} 量纲不一致",
                        loc + ["standard_uncertainty", "unit"],
                        code="incompatible_unit",
                    )
                u_mag = convert_magnitude(
                    Quantity(float(su.value), u_unit),
                    unit,
                    loc + ["standard_uncertainty", "value"],
                )

        prepared.append(
            {
                "name": item.name,
                "symbol": sp.Symbol(item.name),
                "value": q,
                "magnitude": float(item.value),
                "unit": unit,
                "u_magnitude": u_mag,  # 以输入量自身单位表示
            }
        )

    return prepared


def evaluate_model(inputs: list[dict], formula: str) -> dict:
    """复用全部既有校验，求解模型值、单位与各输入量的灵敏度系数。

    返回字典包含：result_q（结果带单位量）、display_unit、expr（解析后的
    SymPy 表达式）、sensitivities（[{input,value,unit}]）与 c_mags（与
    inputs 同序的灵敏度数值，以结果单位/输入单位表示）。相关矩阵的校验在
    调用前完成。
    """
    names = {p["name"] for p in inputs}
    expr = parse_formula(formula, names)

    # 量纲静态校验。
    symbol_dims = {p["symbol"]: p["unit"].dimensionality for p in inputs}
    dimension_of(expr, symbol_dims)

    quantities = {p["symbol"]: p["value"] for p in inputs}

    # 结果值（带单位求值），并把 A·V 这类单位乘积规约为常用导出单位（W）。
    result_q = evaluate(expr, quantities)
    if not math.isfinite(float(result_q.magnitude)):
        raise UnprocessableError(
            "公式求值结果不是有限实数，请检查输入与公式",
            ["formula"],
            code="evaluation_error",
        )
    display_unit = preferred_unit(result_q.units)
    result_value = convert_magnitude(result_q, display_unit, ["formula"])
    result_q = Quantity(result_value, display_unit)

    # 符号偏导并在输入估计值处求值。
    sensitivities: list[dict] = []
    c_mags: list[float] = []
    for p in inputs:
        derivative = sp.diff(expr, p["symbol"])
        deriv_q = evaluate(derivative, quantities)
        # c_i 单位 = 结果单位 / 输入单位（量纲已由静态校验保证）。
        deriv_unit = result_q.units / p["unit"]
        if float(deriv_q.magnitude) == 0.0:
            # 偏导恒为零（输入量未出现在公式中）：零可带任意单位。
            c_mag = 0.0
        else:
            c_mag = convert_magnitude(deriv_q, deriv_unit, ["formula"])
        if not math.isfinite(c_mag):
            raise UnprocessableError(
                f"对 {p['name']} 的偏导在给定输入处不是有限实数，"
                "无法进行一阶不确定度传播",
                ["formula"],
                code="evaluation_error",
            )
        c_mags.append(c_mag)
        sensitivities.append(
            {
                "input": p["name"],
                "value": c_mag,
                "unit": unit_str(deriv_unit),
            }
        )

    return {
        "result_q": result_q,
        "display_unit": display_unit,
        "expr": expr,
        "sensitivities": sensitivities,
        "c_mags": c_mags,
    }


def variance_from_cu(cu: list[float], corr) -> tuple[float, list[list[float]]]:
    """协方差矩阵传播：C_y = c_u^T R c_u，返回（总方差, 逐元素矩阵）。"""
    n = len(cu)
    variance_matrix = [[cu[i] * corr[i, j] * cu[j] for j in range(n)] for i in range(n)]
    total_variance = sum(sum(row) for row in variance_matrix)
    if total_variance < 0 and total_variance > -1e-12:
        total_variance = 0.0
    if total_variance < 0:
        raise UnprocessableError(
            f"传播后方差为负（{total_variance:.6g}），相关结构数值不自洽",
            ["correlation_matrix"],
            code="not_positive_semidefinite",
        )
    return total_variance, variance_matrix


def _build_contributions(
    inputs: list[dict],
    cu: list[float],
    corr,
    variance_matrix: list[list[float]],
    total_variance: float,
    variance_display_unit: str,
) -> list[dict]:
    """按输入量分解方差贡献：u_i^2 对角项 + 相关项在 i、j 上分摊一半。"""
    shares = [cu[i] * cu[i] for i in range(len(inputs))]
    for i in range(len(inputs)):
        for j in range(i + 1, len(inputs)):
            pair = 2.0 * variance_matrix[i][j]
            if abs(pair) < 1e-300:
                pair = 0.0
            shares[i] += pair / 2.0
            shares[j] += pair / 2.0

    contributions: list[dict] = []
    for i, p in enumerate(inputs):
        corr_terms = []
        for j in range(len(inputs)):
            if i == j:
                continue
            pair = 2.0 * variance_matrix[i][j]
            if abs(pair) < 1e-300:
                pair = 0.0
            corr_terms.append(
                {
                    "input": inputs[j]["name"],
                    "correlation": float(corr[i, j]),
                    "covariance_contribution": pair,
                    "unit": variance_display_unit,
                }
            )
        share = shares[i] / total_variance if total_variance > 0 else 0.0
        contributions.append(
            {
                "input": p["name"],
                "variance_contribution": shares[i],
                "unit": variance_display_unit,
                "share": share,
                "correlation_terms": corr_terms,
            }
        )
    return contributions


def propagate(req) -> dict:
    """执行完整传播计算，返回可直接序列化的结果字典。"""
    inputs = _prepare_inputs(req)
    n = len(inputs)

    corr = validate_correlation_matrix(req.correlation_matrix, n)
    model = evaluate_model(inputs, req.formula)
    result_q = model["result_q"]
    display_unit = model["display_unit"]

    cu = [model["c_mags"][i] * inputs[i]["u_magnitude"] for i in range(n)]
    total_variance, variance_matrix = variance_from_cu(cu, corr)
    uc = math.sqrt(total_variance)

    variance_display_unit = unit_str(display_unit ** 2)
    contributions = _build_contributions(
        inputs, cu, corr, variance_matrix, total_variance, variance_display_unit
    )

    return {
        "result": {
            "value": float(result_q.magnitude),
            "unit": unit_str(result_q.units),
            "standard_uncertainty": uc,
        },
        "number_of_inputs": n,
        "correlation_matrix": [[float(x) for x in row] for row in corr],
        "sensitivity_coefficients": model["sensitivities"],
        "variance_contributions": contributions,
        "total_variance": total_variance,
        "total_variance_unit": variance_display_unit,
        "formula": req.formula,
        "whitelist": list(WHITELIST),
    }
