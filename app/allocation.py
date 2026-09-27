"""不确定度逆分配（uncertainty budget allocation）。

在常规 GUM 一阶传播的基础上，把某个输入量 k 的标准不确定度 u_k 视为
**未知非负量**，完整保留相关项，则总方差是关于 u_k（以该输入量自身单位
计）的二次函数：

    Var(u_k) = c_k^2 u_k^2
             + 2 c_k u_k Σ_{j≠k} c_j u_j r_kj
             + Σ_{i≠k} Σ_{j≠k} c_i c_j u_i u_j r_ij
             = a u_k^2 + b u_k + c

相关系数矩阵对角元恒为 1，故 a = c_k^2 ≥ 0。目标合成标准不确定度 U_t
对应二次不等式：

    a u_k^2 + b u_k + (c − U_t^2) ≤ 0，  u_k ≥ 0

求解其最大非负根即“满足目标的最大标准不确定度”。目标不可达
（最小值仍高于目标）、灵敏度为零（c_k = 0）或无有限上界时返回 422。

公式白名单、量纲校验、单位换算、相关矩阵校验全部复用传播模块。
"""
from __future__ import annotations

import math

from .correlation import validate_correlation_matrix
from .errors import UnprocessableError
from .propagation import (
    WHITELIST,
    _build_contributions,
    _prepare_inputs,
    evaluate_model,
    variance_from_cu,
)
from .units import (
    Quantity,
    convert_magnitude,
    parse_unit,
    same_dimensionality,
    unit_str,
)

_TARGET_TOL = 1e-10


def _resolve_allocate_index(req, inputs: list[dict]) -> int:
    """定位待分配输入量，并确保其余输入量都给出了标准不确定度。"""
    index = next(
        (i for i, p in enumerate(inputs) if p["name"] == req.allocate_input),
        None,
    )
    if index is None:
        raise UnprocessableError(
            f"待分配输入量 {req.allocate_input!r} 未在 inputs 中声明",
            ["allocate_input"],
            code="unknown_allocate_input",
        )
    for i, p in enumerate(inputs):
        if i != index and p["u_magnitude"] is None:
            raise UnprocessableError(
                f"输入量 {p['name']!r} 未给出标准不确定度；"
                "仅待分配输入量允许省略标准不确定度",
                ["inputs", i, "standard_uncertainty"],
                code="missing_uncertainty",
            )
    return index


def _parse_target(req, result_q, display_unit) -> float:
    """解析目标合成标准不确定度并换算到结果单位，返回其数值。"""
    loc = ["target_combined_uncertainty"]
    target_unit = parse_unit(req.target_combined_uncertainty.unit, loc + ["unit"])
    if not same_dimensionality(target_unit, display_unit):
        raise UnprocessableError(
            f"目标合成标准不确定度单位 {target_unit:~} 与公式输出单位 "
            f"{display_unit:~} 量纲不一致",
            loc + ["unit"],
            code="incompatible_unit",
        )
    return convert_magnitude(
        Quantity(float(req.target_combined_uncertainty.value), target_unit),
        display_unit,
        loc + ["value"],
    )


def allocate(req) -> dict:
    """执行逆分配计算，返回可直接序列化的结果字典。"""
    inputs = _prepare_inputs(req, allow_missing_uncertainty=True)
    n = len(inputs)

    corr = validate_correlation_matrix(req.correlation_matrix, n)
    k = _resolve_allocate_index(req, inputs)

    model = evaluate_model(inputs, req.formula)
    result_q = model["result_q"]
    display_unit = model["display_unit"]
    c_mags = model["c_mags"]

    target = _parse_target(req, result_q, display_unit)
    target_variance = target * target

    # 未知量取 0 时，其余输入量的 c_j u_j（统一以结果单位计）。
    cu0 = [
        0.0 if i == k else c_mags[i] * inputs[i]["u_magnitude"]
        for i in range(n)
    ]
    c_const, variance_matrix0 = variance_from_cu(cu0, corr)

    # 二次系数：a = c_k^2（r_kk ≡ 1）；
    # b = 2 c_k Σ_{j≠k} c_j u_j r_kj，完整保留相关项。
    c_k = c_mags[k]
    a_coef = c_k * c_k
    b_coef = 2.0 * c_k * sum(cu0[j] * float(corr[k, j]) for j in range(n) if j != k)

    # a 在数值上应为正的有限数（灵敏度非零时 c_k^2 > 0）。
    if not math.isfinite(a_coef) or not math.isfinite(b_coef) or not math.isfinite(c_const):
        raise UnprocessableError(
            "方差二次系数不是有限实数，相关结构数值不自洽",
            ["correlation_matrix"],
            code="not_positive_semidefinite",
        )

    if c_k == 0.0:
        # 灵敏度为零：未知量对总方差毫无影响；若目标已满足则不存在有限
        # 上界（任意大都可行），否则目标不可达。两种情形都无法给出答案。
        if c_const <= target_variance:
            raise UnprocessableError(
                f"待分配输入量 {inputs[k]['name']!r} 的灵敏度系数为零，"
                "其不确定度对合成结果无影响，满足目标的取值不存在有限上界",
                ["allocate_input"],
                code="no_finite_bound",
            )
        raise UnprocessableError(
            f"待分配输入量 {inputs[k]['name']!r} 的灵敏度系数为零，"
            "无法通过调整其不确定度达到目标",
            ["allocate_input"],
            code="zero_sensitivity",
        )

    # 无约束（全体实数）最小方差与最小点，用于判定目标可达性。
    vertex_x = -b_coef / (2.0 * a_coef)
    min_variance = c_const - b_coef * b_coef / (4.0 * a_coef)
    if min_variance < 0.0 and min_variance > -_TARGET_TOL * max(1.0, c_const):
        min_variance = 0.0

    target_loc = ["target_combined_uncertainty"]
    if min_variance > target_variance + _TARGET_TOL * max(1.0, target_variance):
        raise UnprocessableError(
            f"目标合成标准不确定度不可达：即使把 {inputs[k]['name']!r} 的标准不确定度"
            f"取为 {max(0.0, vertex_x):.6g} {unit_str(inputs[k]['unit'])}（最小总方差处），"
            f"合成标准不确定度仍约为 {math.sqrt(max(0.0, min_variance)):.6g} "
            f"{unit_str(display_unit)}，高于目标 {target:.6g} {unit_str(display_unit)}",
            target_loc,
            code="target_unreachable",
        )

    # 解 a x^2 + b x + c = target_variance 的最大非负根。
    discriminant = b_coef * b_coef - 4.0 * a_coef * (c_const - target_variance)
    if discriminant < 0.0:
        if discriminant > -_TARGET_TOL * max(1.0, b_coef * b_coef, 4.0 * a_coef * c_const):
            discriminant = 0.0
        else:
            raise UnprocessableError(
                "目标合成标准不确定度不可达：方差二次方程无实根",
                target_loc,
                code="target_unreachable",
            )
    x_max = (-b_coef + math.sqrt(max(0.0, discriminant))) / (2.0 * a_coef)
    if x_max < 0.0:
        # 两根均为负：非负半轴上可行集为空（x=0 处已超目标）。
        if x_max > -_TARGET_TOL:
            x_max = 0.0
        else:
            raise UnprocessableError(
                "目标合成标准不确定度不可达：待分配输入量不确定度取 0 时"
                "合成标准不确定度已高于目标，且增大该不确定度只会进一步偏离",
                target_loc,
                code="target_unreachable",
            )

    # 代回求得的最大不确定度，按与传播完全相同的路径重算合成值与贡献分解。
    u_known = [inputs[i]["u_magnitude"] if i != k else x_max for i in range(n)]
    cu = [c_mags[i] * u_known[i] for i in range(n)]
    total_variance, variance_matrix = variance_from_cu(cu, corr)

    # 数值上把根的微小残差抹平：略微回收 x_max，使方差不越过目标。
    for _ in range(6):
        if total_variance <= target_variance:
            break
        slope = 2.0 * a_coef * x_max + b_coef
        if slope <= 0.0:
            break
        x_max = max(0.0, x_max - (total_variance - target_variance) / slope)
        u_known[k] = x_max
        cu[k] = c_k * x_max
        total_variance, variance_matrix = variance_from_cu(cu, corr)

    uc = math.sqrt(total_variance)
    variance_display_unit = unit_str(display_unit ** 2)
    contributions = _build_contributions(
        inputs, cu, corr, variance_matrix, total_variance, variance_display_unit
    )

    unknown_unit = inputs[k]["unit"]
    # 系数单位：a: (y/u)^2，b: (y/u)^2·u = y^2/u，c: y^2（均以结果单位 y、
    # 未知量单位 u 表示）。
    a_unit = unit_str(display_unit ** 2 / unknown_unit ** 2)
    b_unit = unit_str(display_unit ** 2 / unknown_unit)
    c_unit = variance_display_unit

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
        "allocate_input": inputs[k]["name"],
        "allocated_standard_uncertainty": x_max,
        "allocated_standard_uncertainty_unit": unit_str(unknown_unit),
        "target_combined_uncertainty": {
            "value": float(req.target_combined_uncertainty.value),
            "unit": req.target_combined_uncertainty.unit,
        },
        "variance_quadratic": {
            "a": a_coef,
            "a_unit": a_unit,
            "b": b_coef,
            "b_unit": b_unit,
            "c": c_const,
            "c_unit": c_unit,
            "unknown_uncertainty_unit": unit_str(unknown_unit),
        },
    }
