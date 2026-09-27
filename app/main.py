"""FastAPI 应用入口。"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from .allocation import allocate
from .conformity import assess_conformity
from .errors import UnprocessableError, request_validation_handler, unprocessable_handler
from .formula import WHITELIST_DISPLAY
from .joint import propagate_joint
from .monte_carlo import propagate_monte_carlo
from .propagation import propagate
from .repeatability import assess_repeatability
from .schemas import (
    AllocationRequest,
    AllocationResponse,
    ConformityRequest,
    ConformityResponse,
    JointPropagationRequest,
    JointPropagationResponse,
    MonteCarloPropagationRequest,
    MonteCarloPropagationResponse,
    PropagationRequest,
    PropagationResponse,
    RepeatabilityRequest,
    RepeatabilityResponse,
)

app = FastAPI(
    title="校准计算不确定度传播 API",
    description=(
        "提交带数值、单位、标准不确定度的输入量、相关系数矩阵和白名单测量公式，"
        "服务端进行量纲校验，并用符号偏导与协方差矩阵计算合成标准不确定度（GUM 一阶）。"
    ),
    version="1.0.0",
)

app.add_exception_handler(UnprocessableError, unprocessable_handler)
app.add_exception_handler(RequestValidationError, request_validation_handler)


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/v1/whitelist", tags=["uncertainty"])
def whitelist() -> dict:
    """返回公式允许使用的运算与函数。"""
    return {"whitelist": WHITELIST_DISPLAY}


_EXAMPLE = {
    "inputs": [
        {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
        {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.02},
    ],
    "formula": "U*I",
    "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
}

_ALLOC_EXAMPLE = {
    "inputs": [
        {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
        # I 的不确定度是待求未知量，可省略其 standard_uncertainty。
        {"name": "I", "value": 2.0, "unit": "A"},
    ],
    "formula": "U*I",
    "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
    "allocate_input": "I",
    "target_combined_uncertainty": {"value": 0.5, "unit": "W"},
}

_JOINT_EXAMPLE = {
    "inputs": [
        {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
        {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.05},
    ],
    "outputs": [
        {"name": "P", "formula": "U*I"},
        {"name": "R", "formula": "U/I"},
    ],
    "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
}

_MONTE_CARLO_EXAMPLE = {
    "inputs": [
        {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
        {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.05},
    ],
    "formula": "U*I",
    "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
    "samples": 50000,
    "seed": 20260920,
}

_CONFORMITY_EXAMPLE = {
    "inputs": [
        {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
        {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.05},
    ],
    "formula": "U*I",
    "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
    "lower_limit": {"value": 18.5, "unit": "W"},
    "upper_limit": {"value": 21.5, "unit": "W"},
    "confidence_threshold": 0.95,
}

_REPEATABILITY_EXAMPLE = {
    "readings": [10.00, 10.01, 10.03, 10.02, 10.00, 9.98, 9.99, 10.01],
    "unit": "V",
    "max_lag": 2,
}


@app.post(
    "/api/v1/propagate",
    response_model=PropagationResponse,
    tags=["uncertainty"],
    responses={422: {"description": "相关矩阵非法或单位/量纲不兼容等"}},
    openapi_extra={
        "examples": {"欧姆定律 P=U*I": {"value": _EXAMPLE}},
    },
)
def calculate(req: PropagationRequest) -> dict:
    """执行一次不确定度传播计算。"""
    return propagate(req)


@app.post(
    "/api/v1/propagate-joint",
    response_model=JointPropagationResponse,
    tags=["uncertainty"],
    responses={
        422: {
            "description": (
                "输出量名称重复、输出公式非法、相关矩阵非法或单位/量纲不兼容等"
            )
        }
    },
    openapi_extra={
        "examples": {"欧姆定律 P=U*I 与 R=U/I 联合传播": {"value": _JOINT_EXAMPLE}},
    },
)
def calculate_joint(req: JointPropagationRequest) -> dict:
    """多输出联合传播：对同一输入协方差按一阶雅可比计算 JΣJᵀ，给出输出协方差与相关系数矩阵。"""
    return propagate_joint(req)


@app.post(
    "/api/v1/propagate-monte-carlo",
    response_model=MonteCarloPropagationResponse,
    tags=["uncertainty"],
    responses={
        422: {
            "description": (
                "样本落入公式无效域 invalid_sample_domain（含失败数量与首个样本索引），"
                "或相关矩阵非法、单位/量纲不兼容、samples 越界等"
            )
        }
    },
    openapi_extra={
        "examples": {"P=U*I：相关输入的蒙特卡洛校核（固定 seed 可复现）": {"value": _MONTE_CARLO_EXAMPLE}},
    },
)
def calculate_monte_carlo(req: MonteCarloPropagationRequest) -> dict:
    """对非线性测量模型做相关正态输入的蒙特卡洛传播，校核 GUM 一阶结果。"""
    return propagate_monte_carlo(req)


@app.post(
    "/api/v1/allocate-uncertainty",
    response_model=AllocationResponse,
    tags=["uncertainty"],
    responses={
        422: {
            "description": (
                "目标不可达 target_unreachable / 灵敏度为零 zero_sensitivity / "
                "无有限上界 no_finite_bound，或相关矩阵非法、单位量纲不兼容等"
            )
        }
    },
    openapi_extra={
        "examples": {"P=U*I：反求 I 的最大允许不确定度": {"value": _ALLOC_EXAMPLE}},
    },
)
def allocate_uncertainty(req: AllocationRequest) -> dict:
    """把指定输入量的标准不确定度视为未知量，反求满足目标合成标准不确定度的最大非负值。"""
    return allocate(req)


@app.post(
    "/api/v1/conformity",
    response_model=ConformityResponse,
    tags=["uncertainty"],
    responses={
        422: {
            "description": (
                "规格界限缺失 missing_spec_limits / 量纲不兼容 incompatible_unit / "
                "下限不小于上限 invalid_spec_limits，或相关矩阵非法、单位/量纲不兼容等"
            )
        }
    },
    openapi_extra={
        "examples": {"P=U*I：双侧规格限 18.5~21.5 W，t=0.95": {"value": _CONFORMITY_EXAMPLE}},
    },
)
def check_conformity(req: ConformityRequest) -> dict:
    """规格符合性判定：一阶传播 + N(y, u_c²) 假设下落入闭规格区间的概率、判定与接受区间。"""
    return assess_conformity(req)


@app.post(
    "/api/v1/repeatability",
    response_model=RepeatabilityResponse,
    tags=["uncertainty"],
    responses={
        422: {
            "description": (
                "读数少于 3 个 / 含非有限值、单位无法识别，或最大滞后 L 越界"
                "（lag_too_large / negative_variance_estimate）等"
            )
        }
    },
    openapi_extra={
        "examples": {"等间隔电压重复读数，L=2 的 A 类标准不确定度": {"value": _REPEATABILITY_EXAMPLE}},
    },
)
def estimate_repeatability(req: RepeatabilityRequest) -> dict:
    """由等间隔重复读数估计均值的 A 类标准不确定度：1/n 自协方差 + Bartlett 加窗长程方差。"""
    return assess_repeatability(req)
