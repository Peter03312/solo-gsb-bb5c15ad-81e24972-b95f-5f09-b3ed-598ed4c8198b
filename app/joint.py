"""多输出联合不确定度传播：共享输入协方差的 JΣJᵀ（GUM 一阶）。

对 m 个输出量 y_k = f_k(x_1, ..., x_n)，一阶雅可比 J（J[k,i] = ∂f_k/∂x_i
在输入估计值处取值）与输入协方差 Σ = D R D（D = diag(u(x_i))，R 为输入
相关系数矩阵）给出输出协方差矩阵：

    C = J Σ Jᵀ，  C[k,l] = Σ_i Σ_j c_i^(k) u(x_i) r_ij u(x_j) c_j^(l)

对角元即各输出量的合成方差 u_c(y_k)²；非对角元为输出间协方差，归一化后
得到输出相关系数矩阵。输入量解析、公式白名单、量纲与单位校验、相关矩阵
校验全部复用传播模块（propagation / correlation）。
"""
from __future__ import annotations

import math

from .correlation import validate_correlation_matrix
from .errors import UnprocessableError
from .propagation import WHITELIST, _prepare_inputs, evaluate_model
from .units import unit_str


def _check_duplicate_names(req) -> None:
    """输出量名称必须互不重复；重复时 loc 定位到后出现的那个。"""
    seen: set[str] = set()
    for index, output in enumerate(req.outputs):
        if output.name in seen:
            raise UnprocessableError(
                f"输出量名称 {output.name!r} 重复",
                ["outputs", index, "name"],
                code="duplicate_name",
            )
        seen.add(output.name)


def _evaluate_output(inputs: list[dict], index: int, formula: str) -> dict:
    """求解单个输出公式；公式相关错误的 loc 重定位到 outputs.<index>.formula。"""
    try:
        return evaluate_model(inputs, formula)
    except UnprocessableError as exc:
        if exc.loc and exc.loc[0] == "formula":
            raise UnprocessableError(
                exc.message,
                ["outputs", index, "formula", *exc.loc[1:]],
                code=exc.code,
            ) from exc
        raise


def propagate_joint(req) -> dict:
    """执行多输出联合传播，返回可直接序列化的结果字典。"""
    inputs = _prepare_inputs(req)
    n = len(inputs)

    corr = validate_correlation_matrix(req.correlation_matrix, n)
    _check_duplicate_names(req)

    models = [
        _evaluate_output(inputs, k, output.formula)
        for k, output in enumerate(req.outputs)
    ]
    m = len(models)

    # 每个输出的 c_i·u(x_i) 向量（以该输出的显示单位计），即 J·D 的行。
    cus = [
        [model["c_mags"][i] * inputs[i]["u_magnitude"] for i in range(n)]
        for model in models
    ]

    # 输出协方差矩阵 C = (J D) R (J D)ᵀ。
    cov = [[0.0] * m for _ in range(m)]
    for k in range(m):
        for l in range(k, m):
            value = sum(
                cus[k][i] * float(corr[i, j]) * cus[l][j]
                for i in range(n)
                for j in range(n)
            )
            cov[k][l] = cov[l][k] = value

    # 对角元即各输出的合成方差；数学上必为非负，钳制数值误差。
    for k in range(m):
        if cov[k][k] < 0.0:
            if cov[k][k] > -1e-12:
                cov[k][k] = 0.0
            else:
                raise UnprocessableError(
                    f"输出 {req.outputs[k].name!r} 传播后方差为负"
                    f"（{cov[k][k]:.6g}），相关结构数值不自洽",
                    ["correlation_matrix"],
                    code="not_positive_semidefinite",
                )

    ucs = [math.sqrt(cov[k][k]) for k in range(m)]

    # 输出相关系数矩阵；零不确定度输出的非对角相关系数无定义，置 0。
    out_corr = [[0.0] * m for _ in range(m)]
    for k in range(m):
        out_corr[k][k] = 1.0
        for l in range(k + 1, m):
            if ucs[k] > 0.0 and ucs[l] > 0.0:
                # 数值误差可能使 |r| 略大于 1，钳制到 [-1, 1]。
                r = max(-1.0, min(1.0, cov[k][l] / (ucs[k] * ucs[l])))
            else:
                r = 0.0
            out_corr[k][l] = out_corr[l][k] = r

    display_units = [model["display_unit"] for model in models]
    covariance_matrix = [
        [
            {
                "value": cov[k][l],
                "unit": unit_str(display_units[k] * display_units[l]),
            }
            for l in range(m)
        ]
        for k in range(m)
    ]

    return {
        "outputs": [
            {
                "name": req.outputs[k].name,
                "value": float(models[k]["result_q"].magnitude),
                "unit": unit_str(models[k]["result_q"].units),
                "standard_uncertainty": ucs[k],
            }
            for k in range(m)
        ],
        "number_of_inputs": n,
        "number_of_outputs": m,
        "correlation_matrix": [[float(x) for x in row] for row in corr],
        "sensitivity_coefficients": [
            {
                "output": req.outputs[k].name,
                "coefficients": models[k]["sensitivities"],
            }
            for k in range(m)
        ],
        "output_covariance_matrix": covariance_matrix,
        "output_correlation_matrix": out_corr,
        "whitelist": list(WHITELIST),
    }
