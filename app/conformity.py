"""规格符合性判定：一阶传播 + 正态假设下落入规格区间的概率。

在 /api/v1/propagate 的一阶传播结果（y, u_c）之上，假定被测量真值服从
正态分布 Y ~ N(y, u_c²)，计算其落入闭规格区间的概率

    p = P(L ≤ Y ≤ U) = Φ((U−y)/u_c) − Φ((L−y)/u_c)

并与置信阈值 t（0.5 < t < 1）比较给出判定：

    p ≥ t     -> 合格（pass）
    p ≤ 1 − t -> 不合格（fail）
    其余      -> 待定（indeterminate）

数值稳定性：正态 CDF 与上尾一律经 erfc 计算（Φ(z)=½·erfc(−z/√2)，
Q(z)=½·erfc(z/√2)）；双侧概率在区间整体位于均值上方时改用上尾差
Q(z_L)−Q(z_U)，避免 1−Φ 在远尾饱和为 1 而把非零尾部概率截成 0
（双精度下可表达到 ~1e-308 量级）。

接受区间：固定 u_c，求使 p(y) ≥ t 的全体 y。单侧界限直接由标准正态
分位数给出半无限区间；双侧界限下 p(y) 在中点 m=(L+U)/2 处取最大值
p_max = 2Φ((U−L)/(2u_c)) − 1，p_max < t 时接受区间为空，否则在
(−∞, m] 与 [m, +∞) 上分别二分求解 p(y)=t 的两个边界（p 在两侧各自
单调，二分收敛）。u_c = 0 时按确定值处理：p∈{0,1}，接受区间即闭规格
区间本身。

输入量解析、公式白名单、量纲/单位校验、相关矩阵校验全部复用传播模块
（propagation / correlation）。
"""
from __future__ import annotations

import math
from statistics import NormalDist

from .correlation import validate_correlation_matrix
from .errors import UnprocessableError
from .propagation import _prepare_inputs, evaluate_model, variance_from_cu
from .units import (
    Quantity,
    convert_magnitude,
    parse_unit,
    same_dimensionality,
    unit_str,
)

_SQRT2 = math.sqrt(2.0)
# 标准正态分布（仅用于逆 CDF 求分位数）。
_STD_NORMAL = NormalDist()


def _norm_cdf(z: float) -> float:
    """标准正态 CDF；erfc 实现使远尾（z → −∞）仍给出可表示的非零值。"""
    return 0.5 * math.erfc(-z / _SQRT2)


def _norm_sf(z: float) -> float:
    """标准正态上尾概率 Q(z) = 1−Φ(z)；远尾不饱和为 0/1 的精度损失最小。"""
    return 0.5 * math.erfc(z / _SQRT2)


def _interval_probability(
    y: float,
    uc: float,
    lower: float | None,
    upper: float | None,
) -> float:
    """P(lower ≤ Y ≤ upper)，Y ~ N(y, uc²)；None 表示该侧无界。"""
    if uc == 0.0:
        # 确定值：闭区间包含判定。
        if lower is not None and y < lower:
            return 0.0
        if upper is not None and y > upper:
            return 0.0
        return 1.0
    if lower is not None and upper is not None:
        z_l = (lower - y) / uc
        z_u = (upper - y) / uc
        if z_l >= 0.0:
            # 区间整体位于均值上方：上尾差避免 1−1 抵消，保留非零尾概率。
            return _norm_sf(z_l) - _norm_sf(z_u)
        return _norm_cdf(z_u) - _norm_cdf(z_l)
    if upper is not None:
        return _norm_cdf((upper - y) / uc)
    # 仅下限（schema 保证至少一个界限）。
    return _norm_sf((lower - y) / uc)


def _convert_limit(spec, display_unit, loc: list) -> float:
    """把带单位的规格界限换算到结果显示单位；量纲不一致抛 422。"""
    unit = parse_unit(spec.unit, loc + ["unit"])
    if not same_dimensionality(unit, display_unit):
        raise UnprocessableError(
            f"规格界限单位 {unit:~} 与公式输出单位 {display_unit:~} 量纲不一致",
            loc + ["unit"],
            code="incompatible_unit",
        )
    return convert_magnitude(
        Quantity(float(spec.value), unit), display_unit, loc + ["value"]
    )


def _bisect_boundary(prob, a: float, b: float, t: float) -> float:
    """在 [a, b] 上二分求解 prob(y)=t，返回满足 p≥t 一侧的边界值。

    调用前须保证 prob(a)、prob(b) 分居 t 两侧（含相等）；迭代中保持
    a、b 各自初始的“是否满足 p≥t”属性不变。
    """
    a_ok = prob(a) >= t
    for _ in range(200):
        mid = 0.5 * (a + b)
        if mid == a or mid == b:
            break
        if (prob(mid) >= t) == a_ok:
            a = mid
        else:
            b = mid
    return a if a_ok else b


def _acceptance_interval(
    uc: float,
    lower: float | None,
    upper: float | None,
    t: float,
) -> tuple[float | None, float | None, bool]:
    """固定 u_c，求 {y : p(y) ≥ t}。

    返回 (下边界, 上边界, 是否为空)；边界为 None 表示该侧无穷。
    """
    if lower is None:
        # 仅上限：Φ((U−y)/u_c) ≥ t ⟺ y ≤ U − u_c·z_t（u_c=0 时即 y ≤ U）。
        z_t = _STD_NORMAL.inv_cdf(t)
        return None, upper - uc * z_t, False
    if upper is None:
        # 仅下限：Φ((y−L)/u_c) ≥ t ⟺ y ≥ L + u_c·z_t。
        z_t = _STD_NORMAL.inv_cdf(t)
        return lower + uc * z_t, None, False
    if uc == 0.0:
        # 确定值：p∈{0,1}，接受区间即闭规格区间本身。
        return lower, upper, False

    def prob(y: float) -> float:
        return _interval_probability(y, uc, lower, upper)

    # p(y) 在中点处取最大值；最大概率仍低于阈值则接受区间为空。
    midpoint = 0.5 * (lower + upper)
    if prob(midpoint) < t:
        return None, None, True

    # 下边界：p 在 (−∞, m] 单调递增且 y→−∞ 时 p→0，向下扩边必可夹逼。
    step = max(uc, upper - lower)
    lo = midpoint - step
    while prob(lo) >= t:
        step *= 2.0
        lo = midpoint - step
    y_lo = _bisect_boundary(prob, lo, midpoint, t)

    # 上边界：p 在 [m, +∞) 单调递减且 y→+∞ 时 p→0。
    step = max(uc, upper - lower)
    hi = midpoint + step
    while prob(hi) >= t:
        step *= 2.0
        hi = midpoint + step
    y_hi = _bisect_boundary(prob, midpoint, hi, t)

    return y_lo, y_hi, False


def assess_conformity(req) -> dict:
    """执行规格符合性判定，返回可直接序列化的结果字典。"""
    inputs = _prepare_inputs(req)
    n = len(inputs)

    corr = validate_correlation_matrix(req.correlation_matrix, n)
    model = evaluate_model(inputs, req.formula)
    result_q = model["result_q"]
    display_unit = model["display_unit"]

    cu = [model["c_mags"][i] * inputs[i]["u_magnitude"] for i in range(n)]
    total_variance, _ = variance_from_cu(cu, corr)
    uc = math.sqrt(total_variance)
    y = float(result_q.magnitude)

    # 规格界限换算到结果显示单位；双侧须满足 下限 < 上限。
    lower = (
        _convert_limit(req.lower_limit, display_unit, ["lower_limit"])
        if req.lower_limit is not None
        else None
    )
    upper = (
        _convert_limit(req.upper_limit, display_unit, ["upper_limit"])
        if req.upper_limit is not None
        else None
    )
    if lower is not None and upper is not None and not lower < upper:
        raise UnprocessableError(
            f"规格下限必须小于上限（换算后下限 {lower}、上限 {upper}，"
            f"单位 {unit_str(display_unit)}）",
            ["lower_limit", "value"],
            code="invalid_spec_limits",
        )

    t = float(req.confidence_threshold)
    p = _interval_probability(y, uc, lower, upper)
    # 数值误差防御性钳制到 [0, 1]。
    p = min(1.0, max(0.0, p))

    if p >= t:
        decision = "pass"
    elif p <= 1.0 - t:
        decision = "fail"
    else:
        decision = "indeterminate"

    acc_lo, acc_hi, acc_empty = _acceptance_interval(uc, lower, upper, t)

    display = unit_str(display_unit)
    return {
        "decision": decision,
        "probability": p,
        "confidence_threshold": t,
        "result": {
            "value": y,
            "unit": display,
            "standard_uncertainty": uc,
        },
        "lower_limit": (
            {"value": lower, "unit": display} if lower is not None else None
        ),
        "upper_limit": (
            {"value": upper, "unit": display} if upper is not None else None
        ),
        "acceptance_interval": {
            "lower": acc_lo,
            "upper": acc_hi,
            "unit": display,
            "is_empty": acc_empty,
        },
        "formula": req.formula,
    }
