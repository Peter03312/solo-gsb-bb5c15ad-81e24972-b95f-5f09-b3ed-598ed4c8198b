"""非线性测量模型的蒙特卡洛传播校核（相关正态输入）。

对测量模型 y = f(x_1, ..., x_n)，在输入量联合正态假设下用蒙特卡洛方法
（GUM Supplement 1 思路）抽样传播：

1. 对（半正定、允许奇异的）相关系数矩阵 R 做特征分解 R = V diag(λ) Vᵀ，
   令标准正态样本 Z（samples × n）经
       X_std = Z · diag(√λ) · Vᵀ
   得到相关标准正态样本——零特征值方向不产生随机扰动，故奇异相关矩阵
   （完全相关）同样可运行；
2. 每个输入量按其**声明单位**解释标准不确定度 u_i，样本
   x_i = m_i + u_i · X_std[:,i]，随后经（仿射）单位映射换算到 SI 基准
   单位，保证 degC/degF 等偏移单位也正确；
3. 用 SymPy ``lambdify`` 生成的向量化白名单函数逐样本求值，结果再换算回
   一阶传播所采用的显示单位；
4. 返回样本均值、样本标准不确定度（ddof=1）、2.5%/50%/97.5% 分位数，
   以及相对同一请求 GUM 一阶传播结果的偏差。

任一样本落入公式无效域（sqrt/log 取非正、反三角函数越界、除零、溢出等
导致结果非有限）时返回 422（code=invalid_sample_domain），并给出失败
样本数量与首个失败样本索引。

输入量解析、公式白名单、量纲与单位校验、相关矩阵校验全部复用传播模块
（propagation / correlation）。
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import sympy as sp

from .correlation import validate_correlation_matrix
from .errors import UnprocessableError
from .propagation import WHITELIST, _prepare_inputs, evaluate_model, variance_from_cu
from .units import Quantity, unit_str

# lambdify 命名空间：白名单内的自定义 SymPy 函数需显式映射到 numpy 原语，
# 其余名称（sqrt/exp/log/sin/abs/pi/E 等）由 numpy 模块提供。
_NUMERIC_NAMESPACE = {
    "Log10": np.log10,
    "Log2": np.log2,
}


def _affine_to_base(unit) -> tuple[float, float]:
    """返回线性/偏移单位 -> SI 基准单位的仿射映射 v_base = a*v + b。

    对比例单位 b=0；对 degC/degF 等偏移单位 b≠0。用两个参考点求斜率，
    避免直接依赖单位是否带偏移的内部判断。
    """
    v0 = float(Quantity(20.0, unit).to_base_units().magnitude)
    v1 = float(Quantity(21.0, unit).to_base_units().magnitude)
    a = v1 - v0
    return a, v0 - 20.0 * a


def _correlated_standard_normal(corr: np.ndarray, samples: int, rng) -> np.ndarray:
    """由相关系数矩阵的特征分解生成相关标准正态样本（samples × n）。

    R = V diag(λ) Vᵀ，则 (diag(√λ) Vᵀ)ᵀ(diag(√λ) Vᵀ) = R；
    零（及数值误差造成的微小负）特征值开方取 0，奇异相关矩阵可直接运行。
    """
    eigenvalues, eigvecs = np.linalg.eigh(corr)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    n = corr.shape[0]
    z = rng.standard_normal((samples, n))
    return z @ (eigvecs * np.sqrt(eigenvalues)) @ eigvecs.T


def propagate_monte_carlo(req) -> dict:
    """执行蒙特卡洛传播校核，返回可直接序列化的结果字典。"""
    inputs = _prepare_inputs(req)
    n = len(inputs)

    corr = validate_correlation_matrix(req.correlation_matrix, n)
    model = evaluate_model(inputs, req.formula)
    expr: sp.Expr = model["expr"]
    display_unit = model["display_unit"]

    # 一阶（GUM 线性化）结果，作为对照基准。
    cu = [model["c_mags"][i] * inputs[i]["u_magnitude"] for i in range(n)]
    first_variance, _ = variance_from_cu(cu, corr)
    first_uc = math.sqrt(first_variance)
    first_value = float(model["result_q"].magnitude)

    # 各输入量 声明单位 -> SI 基准单位 的仿射映射（a, b），含偏移单位。
    input_affine = [_affine_to_base(p["unit"]) for p in inputs]

    # 抽样：相关标准正态 -> 声明单位样本 -> SI 基准单位样本。
    rng = np.random.default_rng(req.seed)
    std_samples = _correlated_standard_normal(corr, req.samples, rng)
    base_samples = np.empty((req.samples, n), dtype=np.float64)
    for i, p in enumerate(inputs):
        a, b = input_affine[i]
        declared = p["magnitude"] + p["u_magnitude"] * std_samples[:, i]
        base_samples[:, i] = a * declared + b

    # 向量化白名单求值（基准单位下量纲自洽）。
    numeric_fn = sp.lambdify(
        [p["symbol"] for p in inputs],
        expr,
        modules=[_NUMERIC_NAMESPACE, "numpy"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            values_base = np.asarray(
                numeric_fn(*[base_samples[:, i] for i in range(n)]),
                dtype=np.float64,
            )
    if values_base.ndim == 0:
        values_base = np.full(req.samples, float(values_base))

    # 无效域：sqrt/log 取非正、反三角越界、除零、溢出等均表现为非有限值。
    invalid = ~np.isfinite(values_base)
    invalid_count = int(invalid.sum())
    if invalid_count:
        first_index = int(np.argmax(invalid))
        raise UnprocessableError(
            f"{invalid_count} 个蒙特卡洛样本落入公式无效域"
            f"（首个为第 {first_index} 个样本）；请检查公式定义域与输入不确定度",
            ["formula"],
            code="invalid_sample_domain",
            details={
                "failed_samples": invalid_count,
                "first_invalid_sample_index": first_index,
                "samples": req.samples,
                "seed": req.seed,
            },
        )

    # 结果从 SI 基准单位换算回一阶传播的显示单位（同样用仿射映射，兼容偏移单位）。
    a_out, b_out = _affine_to_base(display_unit)
    values = (values_base - b_out) / a_out

    # 健全性检查：向量化数值模型在输入估计值处应与带单位求值一致。
    nominal_base = np.array(
        [input_affine[i][0] * inputs[i]["magnitude"] + input_affine[i][1] for i in range(n)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            nominal_check = float(np.asarray(numeric_fn(*nominal_base), dtype=np.float64))
    nominal_display = (nominal_check - b_out) / a_out
    if abs(nominal_display - first_value) > 1e-9 * max(1.0, abs(first_value)):
        raise UnprocessableError(
            "蒙特卡洛数值模型在输入估计值处的求值与带单位求值不一致，"
            "请检查公式与单位",
            ["formula"],
            code="evaluation_error",
        )

    mean = float(values.mean())
    std_u = float(values.std(ddof=1))
    # 常量公式（样本理论上完全相同）经向量化求值可能残留 ~1e-15 的浮点抖动，钳为 0。
    if std_u <= 1e-12 * max(1.0, abs(mean)):
        std_u = 0.0
    q025, q50, q975 = (float(v) for v in np.quantile(values, [0.025, 0.5, 0.975]))

    mean_abs = mean - first_value
    mean_rel = mean_abs / abs(first_value) if first_value != 0.0 else None
    uc_abs = std_u - first_uc
    uc_rel = uc_abs / first_uc if first_uc > 0.0 else None

    display_unit_str = unit_str(display_unit)
    return {
        "result": {
            "value": mean,
            "unit": display_unit_str,
            "standard_uncertainty": std_u,
        },
        "quantiles": {"q025": q025, "q50": q50, "q975": q975},
        "samples": req.samples,
        "seed": req.seed,
        "number_of_inputs": n,
        "correlation_matrix": [[float(x) for x in row] for row in corr],
        "first_order": {
            "value": first_value,
            "unit": display_unit_str,
            "standard_uncertainty": first_uc,
        },
        "deviation": {
            "mean_absolute": mean_abs,
            "mean_relative": mean_rel,
            "standard_uncertainty_absolute": uc_abs,
            "standard_uncertainty_relative": uc_rel,
        },
        "formula": req.formula,
        "whitelist": list(WHITELIST),
    }
