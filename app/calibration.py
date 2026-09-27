"""校准修正：相邻节点线性插值与节点协方差的不确定度合成。

给定按示值严格递增的 n 个校准节点（n ≥ 2）：

    (x_i, c_i, u_i)，i = 0, ..., n-1

（x_i 示值、c_i 修正值、u_i 修正值标准不确定度）以及节点修正值相关系数
矩阵 R，对待修正示值 x（标准不确定度 u(x)）做分段线性插值修正：

    区间定位：x ∈ [x_k, x_{k+1})；节点处（x = x_k，k < n-1）选右侧区间，
    末节点（x = x_{n-1}）选左侧区间 [x_{n-2}, x_{n-1}]。

    插值权重：t = (x − x_k)/(x_{k+1} − x_k)，w_k = 1−t、w_{k+1} = t，
    其余为 0；修正值 c(x) = Σ_i w_i c_i，修正后结果 y = x + c(x)。

不确定度合成（示值误差与节点误差独立，两类方差相加）：

    修正值方差：u²(c) = wᵀ Σ w，其中 Σ = D R D 为节点修正值协方差矩阵
    （D = diag(u_i)）；

    示值方差贡献：修正曲线斜率 s = (c_{k+1} − c_k)/(x_{k+1} − x_k)，
    由 y = x + c(x) 得 dy/dx = 1 + s，故 u²_x = (1 + s)² u²(x)；

    合成标准不确定度 u_c = √(u²(c) + u²_x)。

全部量须同量纲：以待修正示值单位为显示单位；示值按点量（仿射）换算，
修正值与标准不确定度按差值量（仅取换算比例、不应用零点偏移，如 1 °C
的差值等于 1 K 而非 274.15 K）换算。单位解析与相关系数矩阵校验复用
units / correlation 模块。
"""
from __future__ import annotations

import bisect
import math

import numpy as np
from pint import DimensionalityError, OffsetUnitCalculusError

from .correlation import validate_correlation_matrix
from .errors import UnprocessableError
from .units import Quantity, convert_magnitude, parse_unit, same_dimensionality, unit_str


def _convert_difference(
    value: float, unit, display_unit, loc: list
) -> float:
    """把差值量（修正值、标准不确定度）换算到显示单位。

    只取单位换算比例、不应用零点偏移（用两点差求比例）；比例单位下与
    ``convert_magnitude`` 一致。
    """
    try:
        high = float(Quantity(1.0, unit).to(display_unit).magnitude)
        low = float(Quantity(0.0, unit).to(display_unit).magnitude)
    except (DimensionalityError, OffsetUnitCalculusError) as exc:
        raise UnprocessableError(
            f"单位不兼容：{unit:~} 无法换算为 {display_unit:~}：{exc}",
            loc,
            code="incompatible_unit",
        ) from exc
    return value * (high - low)


def _prepare_quantity(
    value: float,
    unit_text: str,
    display_unit,
    loc: list,
    *,
    difference: bool,
) -> float:
    """解析单位、校验与显示单位同量纲，并换算到显示单位。

    ``difference=True`` 按差值量（仅比例）换算，否则按点量（仿射）换算；
    loc 定位到该量对象（如 ``["nodes", 0, "correction"]``）。
    """
    unit = parse_unit(unit_text, loc + ["unit"])
    if not same_dimensionality(unit, display_unit):
        raise UnprocessableError(
            f"单位 {unit:~} 与待修正示值单位 {display_unit:~} 量纲不一致",
            loc + ["unit"],
            code="incompatible_unit",
        )
    if difference:
        return _convert_difference(float(value), unit, display_unit, loc + ["value"])
    return convert_magnitude(Quantity(float(value), unit), display_unit, loc + ["value"])


def apply_calibration(req) -> dict:
    """执行校准修正计算，返回可直接序列化的结果字典。"""
    n = len(req.nodes)

    # 显示单位 = 待修正示值单位；全部量须与其同量纲。
    display_unit = parse_unit(req.indication.unit, ["indication", "unit"])
    x = float(req.indication.value)
    u_x = _prepare_quantity(
        req.indication_standard_uncertainty.value,
        req.indication_standard_uncertainty.unit,
        display_unit,
        ["indication_standard_uncertainty"],
        difference=True,
    )

    xs: list[float] = []
    cs: list[float] = []
    us: list[float] = []
    for index, node in enumerate(req.nodes):
        loc = ["nodes", index]
        xs.append(_prepare_quantity(
            node.indication.value, node.indication.unit, display_unit,
            loc + ["indication"], difference=False,
        ))
        cs.append(_prepare_quantity(
            node.correction.value, node.correction.unit, display_unit,
            loc + ["correction"], difference=True,
        ))
        us.append(_prepare_quantity(
            node.correction_standard_uncertainty.value,
            node.correction_standard_uncertainty.unit, display_unit,
            loc + ["correction_standard_uncertainty"], difference=True,
        ))

    display = unit_str(display_unit)

    # 节点示值必须严格递增（换算到同一显示单位后判定）。
    for index in range(1, n):
        if not xs[index] > xs[index - 1]:
            raise UnprocessableError(
                f"校准节点示值必须严格递增：第 {index - 1} 个节点示值 "
                f"{xs[index - 1]}，第 {index} 个为 {xs[index]}（单位 {display}）",
                ["nodes", index, "indication", "value"],
                code="indication_not_strictly_increasing",
            )

    # 节点修正值相关系数矩阵（缺省为单位阵）。
    corr = validate_correlation_matrix(req.correlation_matrix, n)

    # 待修正示值须落在节点表范围内（闭区间）。
    if x < xs[0] or x > xs[-1]:
        raise UnprocessableError(
            f"待修正示值 {x} 超出校准节点范围 [{xs[0]}, {xs[-1]}]（单位 {display}）",
            ["indication", "value"],
            code="indication_out_of_range",
            details={
                "indication": x,
                "min_indication": xs[0],
                "max_indication": xs[-1],
                "unit": display,
            },
        )

    # 区间定位：bisect_right 使节点处（x = x_k）选右侧区间；
    # 末节点处钳到最后一段（左侧区间）。
    k = min(bisect.bisect_right(xs, x) - 1, n - 2)
    x0, x1 = xs[k], xs[k + 1]

    # 线性插值权重（仅区间两端节点非零）。
    t = (x - x0) / (x1 - x0)
    weights = [0.0] * n
    weights[k] = 1.0 - t
    weights[k + 1] = t

    correction = weights[k] * cs[k] + weights[k + 1] * cs[k + 1]
    slope = (cs[k + 1] - cs[k]) / (x1 - x0)

    # 修正值方差：wᵀΣw，Σ = D R D（节点修正值协方差矩阵）。
    w_arr = np.asarray(weights, dtype=np.float64)
    covariance = corr * np.outer(us, us)
    correction_variance = float(w_arr @ covariance @ w_arr)
    # 相关矩阵已校验半正定，数学上非负；钳制浮点残差，显著为负报 422。
    if correction_variance < 0.0:
        tolerance = 1e-12 * max(us) ** 2 if max(us) > 0.0 else 0.0
        if correction_variance < -tolerance:
            raise UnprocessableError(
                f"修正值方差为负（{correction_variance:.6g}），相关结构数值不自洽",
                ["correlation_matrix"],
                code="not_positive_semidefinite",
            )
        correction_variance = 0.0

    # 示值方差贡献：(1 + 修正曲线斜率)² u²（与节点误差独立，直接相加）。
    indication_variance = (1.0 + slope) ** 2 * u_x ** 2

    combined = math.sqrt(correction_variance + indication_variance)
    y = x + correction

    # 方差单位 = 显示单位的平方；degC 等偏移单位的平方仅用于展示。
    try:
        variance_unit = unit_str(display_unit ** 2)
    except OffsetUnitCalculusError:
        variance_unit = f"{display} ** 2"

    return {
        "result": {
            "value": y,
            "unit": display,
            "standard_uncertainty": combined,
        },
        "correction": {"value": correction, "unit": display},
        "number_of_nodes": n,
        "interval": {
            "index": k,
            "lower": {"value": x0, "unit": display},
            "upper": {"value": x1, "unit": display},
        },
        "weights": weights,
        "slope": slope,
        "correction_variance_contribution": {
            "value": correction_variance,
            "unit": variance_unit,
        },
        "indication_variance_contribution": {
            "value": indication_variance,
            "unit": variance_unit,
        },
        "correlation_matrix": [[float(v) for v in row] for row in corr],
    }
