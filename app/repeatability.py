"""等间隔重复读数的 A 类标准不确定度：Bartlett 加窗长程方差估计。

对同一被测量按等时间间隔取得的 n 个同单位读数 x_1, ..., x_n，先去均值，
按**有偏**（1/n 归一化）样本自协方差

    γ_k = (1/n) Σ_{i=1}^{n-k} (x_i − x̄)(x_{i+k} − x̄),  k = 0, ..., L

估计序列自协方差，再用 Bartlett（三角）权重 w_k = 1 − k/(L+1) 估计均值的
方差：

    u_A² = [ γ_0 + 2 Σ_{k=1}^{L} w_k γ_k ] / n

正相关的平稳序列相邻读数携带重复信息，朴素 s/√n 会高估均值不确定度；
加窗长程方差修正这一点。有效样本量

    n_eff = n γ_0 / [ γ_0 + 2 Σ_{k=1}^{L} w_k γ_k ]

为长程方差分母为零时（如全部读数相同）无定义，返回 null。

加窗括号 S = Σ_{k=-L}^{L} w_{|k|} γ_k 非负：令 A 为 n×(n+L) 带状阵
（A[i, s]=1 当 i ≤ s ≤ i+L），则 (AAᵀ)[i,j] = L+1−|i−j|（|i−j|≤L），
即 (L+1) 倍 Bartlett 权矩阵，故

    (L+1) n S = dᵀAAᵀd = Σ_{s=0}^{n+L−1} ( Σ_{i=max(0,s−L)}^{min(n−1,s)} d_i )² ≥ 0

（Bartlett/Fejér 核半正定）。合法输入下 S 非负；浮点残差导致的
微小负值钳制为 0，显著为负则按数值不自洽报 422。

仅依赖单位注册表校验单位字符串；结果单位与读数单位保持一致，u_A 可
直接作为 /api/v1/propagate 输入量的 standard_uncertainty。
"""
from __future__ import annotations

import math

from pint import OffsetUnitCalculusError

from .errors import UnprocessableError
from .units import parse_unit, unit_str


def _autocovariances(deviations: list[float], n: int, lag_max: int) -> list[float]:
    """γ_k = (1/n) Σ_{i=0}^{n-1-k} d_i d_{i+k}，k = 0..L。"""
    gammas: list[float] = []
    for k in range(lag_max + 1):
        gammas.append(math.fsum(
            deviations[i] * deviations[i + k] for i in range(n - k)
        ) / n)
    return gammas


def assess_repeatability(req) -> dict:
    """执行 A 类标准不确定度估计，返回可直接序列化的结果字典。"""
    n = len(req.readings)
    lag_max = int(req.max_lag)

    # 单位仅校验可识别性；不做任何换算，结果保持输入单位。
    unit = parse_unit(req.unit, ["unit"])
    display = unit_str(unit)

    # 0 ≤ L 已由 schema 的 ge=0 保证；上界 L < n 与具体读数数量有关，在此校验。
    if lag_max >= n:
        raise UnprocessableError(
            f"最大滞后 L 必须小于读数数量 n（得到 L={lag_max}、n={n}，"
            f"要求 0 ≤ L ≤ {n - 1}）",
            ["max_lag"],
            code="lag_too_large",
        )

    readings = [float(v) for v in req.readings]  # schema 已保证全部有限
    mean = math.fsum(readings) / n
    deviations = [x - mean for x in readings]

    gammas = _autocovariances(deviations, n, lag_max)
    gamma_0 = gammas[0]

    # Bartlett 加窗长程方差（公式中的方括号项）。
    long_run_variance = gamma_0 + 2.0 * math.fsum(
        (1.0 - k / (lag_max + 1)) * gammas[k] for k in range(1, lag_max + 1)
    )

    # 半正定二次型保证 S ≥ 0；仅允许浮点量级的负残差并钳制为 0。
    if long_run_variance < 0.0:
        tolerance = 1e-12 * gamma_0 if gamma_0 > 0.0 else 0.0
        if long_run_variance < -tolerance:
            raise UnprocessableError(
                "Bartlett 加窗后的长程方差估计为负"
                f"（{long_run_variance:.6g}），自协方差估计数值不自洽；"
                "请减小最大滞后 L 或检查读数序列",
                ["max_lag"],
                code="negative_variance_estimate",
            )
        long_run_variance = 0.0

    variance = long_run_variance / n
    u_a = math.sqrt(variance) if variance > 0.0 else 0.0
    effective_n = (
        n * gamma_0 / long_run_variance if long_run_variance != 0.0 else None
    )

    # 自协方差单位 = 读数单位的平方。degC/degF 等偏移单位的平方在量纲上有
    # 歧义（其平方量纲是温度差的平方），pint 直接 u**2 会抛
    # OffsetUnitCalculusError；按显示字符串拼出平方单位即可（仅用于展示）。
    try:
        gamma_unit = unit_str(unit ** 2)
    except OffsetUnitCalculusError:
        gamma_unit = f"{display} ** 2"

    return {
        "result": {
            "value": mean,
            "unit": display,
            "standard_uncertainty": u_a,
        },
        "number_of_readings": n,
        "max_lag": lag_max,
        "autocovariances": [
            {"lag": k, "value": gammas[k], "unit": gamma_unit}
            for k in range(lag_max + 1)
        ],
        "variance": variance,
        "variance_unit": gamma_unit,
        "effective_sample_size": effective_n,
    }
