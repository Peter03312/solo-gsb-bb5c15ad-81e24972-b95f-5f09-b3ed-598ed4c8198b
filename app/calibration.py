"""校准修正：相邻节点线性插值 + 两类方差贡献的合成标准不确定度。

给定 n ≥ 2 个校准节点 (x_i, c_i, u_i)——示值 x_i 严格递增，c_i 为该示值处
的修正值，u_i 为修正值的标准不确定度——以及节点修正值相关系数矩阵 R，
对待修正示值 x（标准不确定度 u_x）：

1. 区间选择：取相邻节点区间 [x_i, x_{i+1}] 使 x_i ≤ x ≤ x_{i+1}；示值恰好
   落在节点上时取**右侧**区间，末节点取**左侧**区间（i = n−2）；示值超出
   节点范围返回 422（indication_out_of_range）；
2. 线性插值：t = (x − x_i)/(x_{i+1} − x_i)，修正值 c(x) = (1−t)·c_i + t·c_{i+1}，
   修正后示值 y = x + c(x)；
3. 修正值方差：插值权重向量 w（仅第 i、i+1 两个分量非零）与节点协方差
   Σ = D R D（D = diag(u_i)）给出 u(c)² = wᵀΣw；
4. 示值方差贡献：修正曲线在该区间的斜率 s = (c_{i+1} − c_i)/(x_{i+1} − x_i)
   （同量纲，故无量纲），y 对 x 的灵敏度为 dy/dx = 1 + s，贡献
   (1 + s)²·u_x²；示值误差与节点误差相互独立，两类方差相加后开方即
   合成标准不确定度。

所有量（节点示值/修正值/修正值不确定度、待修正示值及其不确定度）须同量纲，
统一换算到待修正示值的声明单位后计算，结果以该单位表示。修正值与不确定度
是**差值量**，按单位仿射斜率换算（1 单位差 = scale 目标单位差），degC/degF
等偏移单位的差值因此也能正确处理。单位解析/换算与相关矩阵校验复用
units / correlation 模块。
"""
from __future__ import annotations

import bisect
import math

from pint import DimensionalityError, OffsetUnitCalculusError

from .correlation import validate_correlation_matrix
from .errors import UnprocessableError
from .units import (
    Quantity,
    convert_magnitude,
    parse_unit,
    same_dimensionality,
    unit_str,
)


def _convert_difference(
    value: float, unit, target, loc: list[str | int]
) -> float:
    """把差值量（修正值/不确定度）按单位比例换算到 target 单位。

    差值不带偏移：1 unit 的差 = scale target 的差，scale 由两点仿射斜率
    求得，偏移单位（如 degC）的差值因此换算为对应的目标单位差（而非按
    绝对量加上偏移）。
    """
    try:
        offset = float(Quantity(0.0, unit).to(target).magnitude)
        scale = float(Quantity(1.0, unit).to(target).magnitude) - offset
    except (DimensionalityError, OffsetUnitCalculusError) as exc:
        raise UnprocessableError(
            f"单位不兼容：{unit:~} 无法换算为 {target:~}：{exc}",
            loc,
            code="incompatible_unit",
        ) from exc
    return value * scale


def _parse_same_dimension(unit_text: str, ref_unit, loc: list[str | int]):
    """解析单位并要求与参考单位同量纲；否则抛定位明确的 422。"""
    unit = parse_unit(unit_text, loc)
    if not same_dimensionality(unit, ref_unit):
        raise UnprocessableError(
            f"单位 {unit:~} 与待修正示值单位 {ref_unit:~} 量纲不一致",
            loc,
            code="incompatible_unit",
        )
    return unit


def _prepare_nodes(req, ref_unit) -> tuple[list[float], list[float], list[float]]:
    """解析全部节点并换算到参考单位，返回 (示值, 修正值, 修正值不确定度)。"""
    xs: list[float] = []
    cs: list[float] = []
    us: list[float] = []
    for index, node in enumerate(req.nodes):
        loc = ["nodes", index]

        ind_unit = _parse_same_dimension(
            node.indication.unit, ref_unit, loc + ["indication", "unit"]
        )
        # 示值是绝对量，按仿射换算（含偏移单位的绝对换算）。
        xs.append(
            convert_magnitude(
                Quantity(float(node.indication.value), ind_unit),
                ref_unit,
                loc + ["indication", "value"],
            )
        )

        cor_unit = _parse_same_dimension(
            node.correction.unit, ref_unit, loc + ["correction", "unit"]
        )
        cs.append(
            _convert_difference(
                float(node.correction.value),
                cor_unit,
                ref_unit,
                loc + ["correction", "value"],
            )
        )

        unc = node.correction_standard_uncertainty
        unc_unit = _parse_same_dimension(
            unc.unit, ref_unit, loc + ["correction_standard_uncertainty", "unit"]
        )
        us.append(
            _convert_difference(
                float(unc.value),
                unc_unit,
                ref_unit,
                loc + ["correction_standard_uncertainty", "value"],
            )
        )
    return xs, cs, us


def apply_calibration(req) -> dict:
    """执行校准修正（线性插值），返回可直接序列化的结果字典。"""
    # 工作/显示单位 = 待修正示值的声明单位。
    ref_unit = parse_unit(req.indication.unit, ["indication", "unit"])
    display = unit_str(ref_unit)
    x = float(req.indication.value)  # schema 已保证有限

    unc_unit = _parse_same_dimension(
        req.indication_standard_uncertainty.unit,
        ref_unit,
        ["indication_standard_uncertainty", "unit"],
    )
    u_x = _convert_difference(
        float(req.indication_standard_uncertainty.value),
        unc_unit,
        ref_unit,
        ["indication_standard_uncertainty", "value"],
    )

    xs, cs, us = _prepare_nodes(req, ref_unit)
    n = len(xs)

    # 节点示值严格递增（换算到统一单位后判定）。
    for i in range(n - 1):
        if not xs[i] < xs[i + 1]:
            raise UnprocessableError(
                f"节点示值必须严格递增：第 {i} 个节点示值 {xs[i]} 不小于"
                f"第 {i + 1} 个节点示值 {xs[i + 1]}（单位 {display}）",
                ["nodes", i + 1, "indication", "value"],
                code="nodes_not_strictly_increasing",
            )

    corr = validate_correlation_matrix(req.correlation_matrix, n)

    # 区间选择：bisect_right 使示值落在节点上时取右侧区间；
    # 末节点（bisect_right = n）钳到 i = n−2，即左侧区间。
    if x < xs[0] or x > xs[-1]:
        raise UnprocessableError(
            f"待修正示值 {x} 超出校准节点范围 [{xs[0]}, {xs[-1]}]（单位 {display}）",
            ["indication", "value"],
            code="indication_out_of_range",
            details={
                "indication": x,
                "lower": xs[0],
                "upper": xs[-1],
                "unit": display,
            },
        )
    i = min(bisect.bisect_right(xs, x), n - 1) - 1

    x_lo, x_hi = xs[i], xs[i + 1]
    span = x_hi - x_lo  # 严格递增已校验，必为正
    t = (x - x_lo) / span
    w_lo, w_hi = 1.0 - t, t

    correction = w_lo * cs[i] + w_hi * cs[i + 1]
    corrected = x + correction
    slope = (cs[i + 1] - cs[i]) / span  # 同量纲相除，无量纲

    # 修正值方差：wᵀΣw，Σ = D R D（仅 i、i+1 两个权重非零）。
    weights = [0.0] * n
    weights[i] = w_lo
    weights[i + 1] = w_hi
    wu = [weights[k] * us[k] for k in range(n)]
    correction_variance = sum(
        wu[k] * float(corr[k, l]) * wu[l] for k in range(n) for l in range(n)
    )
    # 数学上 PSD 保证非负；仅允许浮点量级的负残差并钳制为 0。
    if correction_variance < 0.0:
        if correction_variance > -1e-12:
            correction_variance = 0.0
        else:
            raise UnprocessableError(
                f"插值修正值方差为负（{correction_variance:.6g}），相关结构数值不自洽",
                ["correlation_matrix"],
                code="not_positive_semidefinite",
            )

    # 示值方差贡献：dy/dx = 1 + s，示值误差与节点误差独立，方差直接相加。
    amplification = 1.0 + slope
    indication_variance = amplification * amplification * u_x * u_x
    combined_variance = correction_variance + indication_variance
    u_combined = math.sqrt(combined_variance) if combined_variance > 0.0 else 0.0

    # 方差单位 = 显示单位的平方；偏移单位（如 degC）的平方在 pint 中有歧义，
    # 按显示字符串拼出平方单位（仅用于展示），与 repeatability 模块一致。
    try:
        variance_unit = unit_str(ref_unit ** 2)
    except OffsetUnitCalculusError:
        variance_unit = f"{display} ** 2"

    return {
        "result": {
            "value": corrected,
            "unit": display,
            "standard_uncertainty": u_combined,
        },
        "correction": {
            "value": correction,
            "unit": display,
            "standard_uncertainty": math.sqrt(correction_variance),
        },
        "interval": {
            "lower_node_index": i,
            "upper_node_index": i + 1,
            "lower": x_lo,
            "upper": x_hi,
            "unit": display,
        },
        "weights": [
            {"node_index": i, "weight": w_lo},
            {"node_index": i + 1, "weight": w_hi},
        ],
        "correction_curve_slope": slope,
        "correction_variance_contribution": {
            "value": correction_variance,
            "unit": variance_unit,
        },
        "indication_variance_contribution": {
            "value": indication_variance,
            "unit": variance_unit,
        },
        "combined_variance": {"value": combined_variance, "unit": variance_unit},
        "number_of_nodes": n,
        "correlation_matrix": [[float(v) for v in row] for row in corr],
    }
