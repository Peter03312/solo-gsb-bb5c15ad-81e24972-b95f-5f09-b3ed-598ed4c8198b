"""相关系数矩阵校验：形状、取值、对称性、半正定性。"""
from __future__ import annotations

import math

import numpy as np

from .errors import UnprocessableError

_VALUE_TOL = 1e-9
_SYM_TOL = 1e-9


def validate_correlation_matrix(
    matrix: list[list[float]] | None, n: int
) -> np.ndarray:
    """校验并返回 n×n 的 float64 相关系数矩阵；缺省返回单位阵。

    任一失败抛 :class:`UnprocessableError`，loc 精确定位到出错元素。
    """
    if matrix is None:
        return np.eye(n, dtype=np.float64)

    loc = ["correlation_matrix"]

    if len(matrix) != n:
        raise UnprocessableError(
            f"相关系数矩阵行数 {len(matrix)} 与输入量数量 {n} 不一致",
            loc,
            code="bad_matrix_shape",
        )
    for i, row in enumerate(matrix):
        if len(row) != n:
            raise UnprocessableError(
                f"相关系数矩阵第 {i} 行长度 {len(row)} 应为 {n}（必须为方阵）",
                loc + [i],
                code="bad_matrix_shape",
            )
        for j, value in enumerate(row):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise UnprocessableError(
                    f"矩阵元素 ({i},{j}) 必须是数值",
                    loc + [i, j],
                    code="bad_matrix_value",
                )
            if not math.isfinite(float(value)):
                raise UnprocessableError(
                    f"矩阵元素 ({i},{j}) 不是有限实数",
                    loc + [i, j],
                    code="bad_matrix_value",
                )
            if value < -1.0 - _VALUE_TOL or value > 1.0 + _VALUE_TOL:
                raise UnprocessableError(
                    f"相关系数 ({i},{j}) = {value} 超出 [-1, 1] 范围",
                    loc + [i, j],
                    code="correlation_out_of_range",
                )

    corr = np.asarray(matrix, dtype=np.float64)

    # 对角线必须为 1。
    for i in range(n):
        if abs(corr[i, i] - 1.0) > _VALUE_TOL:
            raise UnprocessableError(
                f"相关系数矩阵对角线元素 ({i},{i}) = {corr[i, i]}，必须为 1",
                loc + [i, i],
                code="diagonal_not_one",
            )

    # 对称性。
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j] - corr[j, i]) > _SYM_TOL:
                raise UnprocessableError(
                    f"相关系数矩阵不对称：r[{i},{j}] = {corr[i, j]}，"
                    f"r[{j},{i}] = {corr[j, i]}",
                    loc + [i, j],
                    code="asymmetric_matrix",
                )

    corr = (corr + corr.T) / 2.0

    # 半正定性（对称化后做特征值检查）。
    eigenvalues = np.linalg.eigvalsh(corr)
    tol = _VALUE_TOL * max(n, 1)
    min_eig = float(np.min(eigenvalues))
    if min_eig < -tol:
        worst = int(np.argmin(eigenvalues))
        raise UnprocessableError(
            f"相关系数矩阵非半正定：最小特征值为 {min_eig:.6g}（容差 {-tol:.2e}），"
            f"请检查相关系数取值是否自洽（对应第 {worst} 个主方向）",
            loc,
            code="not_positive_semidefinite",
        )
    # 微小负特征值钳制到 0，保证后续数值非负。
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return _reconstruct(corr, eigenvalues)


def _reconstruct(symmetric: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
    """由修正后的特征值重建矩阵（保留到 12 位有效数字的对称形式）。"""
    eigvecs = np.linalg.eigh(symmetric)[1]
    rebuilt = (eigvecs * eigenvalues) @ eigvecs.T
    return np.round((rebuilt + rebuilt.T) / 2.0, 12)
