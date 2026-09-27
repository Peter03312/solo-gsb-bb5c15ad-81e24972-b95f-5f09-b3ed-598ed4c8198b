"""请求/响应数据模型。"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

# 有限实数（拒绝 NaN/Inf），不确定度非负。
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegFiniteFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class UncertaintySpec(BaseModel):
    """带单位的标准不确定度；dimensionless 时可省略单位。"""

    value: NonNegFiniteFloat
    unit: str | None = Field(default=None, description="不确定度单位；缺省视为无量纲")


def _coerce_uncertainty(value: Any) -> Any:
    """校验并规整 standard_uncertainty 的原始 JSON 取值。"""
    if _is_real_number(value):
        if value < 0:
            raise PydanticCustomError(
                "non_negative",
                "标准不确定度必须为非负数，得到 {value}",
                {"value": value},
            )
        return float(value)
    if isinstance(value, dict):
        # 在此校验数值，避免落入联合分支产生 'float' 之类的定位尾巴。
        num = value.get("value")
        if not _is_real_number(num) or num < 0:
            raise PydanticCustomError(
                "non_negative",
                "不确定度 value 必须为非负有限实数",
            )
        value = {**value, "value": float(num)}
        unit = value.get("unit")
        if unit is not None and not isinstance(unit, str):
            raise PydanticCustomError("invalid_unit", "不确定度 unit 必须为字符串")
        return value
    raise PydanticCustomError(
        "invalid_uncertainty",
        "standard_uncertainty 必须是非负数值或 {{value, unit}} 对象",
    )


class InputQuantity(BaseModel):
    """一个输入量：数值、单位、标准不确定度。"""

    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="输入量名称（Python 标识符）",
    )
    value: FiniteFloat
    unit: str = Field(..., min_length=1, description="数值单位，如 m / kg / mV / rad")
    standard_uncertainty: float | UncertaintySpec = Field(
        ...,
        description="标量时与 value 同单位；对象形式 {value, unit} 可显式给出不确定度单位",
    )

    @field_validator("standard_uncertainty", mode="before")
    @classmethod
    def _validate_uncertainty_shape(cls, value: Any) -> Any:
        """按原始 JSON 类型分流，避免智能联合把标量胁迫为对象。"""
        return _coerce_uncertainty(value)


class AllocationInputQuantity(InputQuantity):
    """不确定度分配请求中的输入量；待分配输入量可省略自身不确定度。"""

    standard_uncertainty: float | UncertaintySpec | None = Field(
        default=None,
        description="待分配输入量可省略（其不确定度是待求未知量）；其余输入量必须给出",
    )

    @field_validator("standard_uncertainty", mode="before")
    @classmethod
    def _validate_uncertainty_shape(cls, value: Any) -> Any:
        if value is None:
            return None
        return _coerce_uncertainty(value)


class PropagationRequest(BaseModel):
    inputs: list[InputQuantity] = Field(..., min_length=1)
    formula: str = Field(..., min_length=1, description="仅含白名单运算的测量公式")
    correlation_matrix: list[list[FiniteFloat]] | None = Field(
        default=None,
        description="输入量相关系数矩阵，顺序与 inputs 一致；缺省为单位阵（互不相关）",
    )


class NamedFormula(BaseModel):
    """一个具名输出公式（联合传播的输出量定义）。"""

    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="输出量名称（Python 标识符，须互不重复）",
    )
    formula: str = Field(..., min_length=1, description="仅含白名单运算的测量公式")


class JointPropagationRequest(BaseModel):
    """多输出联合传播请求：同一组输入量、同一相关矩阵，2～8 条输出公式。"""

    inputs: list[InputQuantity] = Field(..., min_length=1)
    outputs: list[NamedFormula] = Field(
        ..., min_length=2, max_length=8, description="2～8 个具名输出公式"
    )
    correlation_matrix: list[list[FiniteFloat]] | None = Field(
        default=None,
        description="输入量相关系数矩阵，顺序与 inputs 一致；缺省为单位阵（互不相关）",
    )


class TargetUncertainty(BaseModel):
    """带单位的目标合成标准不确定度。"""

    value: NonNegFiniteFloat
    unit: str = Field(..., min_length=1, description="目标合成标准不确定度的单位，须与模型输出同量纲")


class TargetUncertaintyEcho(BaseModel):
    """回显请求中指定的目标合成标准不确定度。"""

    value: float
    unit: str


class AllocationRequest(BaseModel):
    """不确定度逆分配请求：把 allocate_input 的标准不确定度视为未知量。"""

    inputs: list[AllocationInputQuantity] = Field(..., min_length=1)
    formula: str = Field(..., min_length=1, description="仅含白名单运算的测量公式")
    correlation_matrix: list[list[FiniteFloat]] | None = Field(
        default=None,
        description="输入量相关系数矩阵，顺序与 inputs 一致；缺省为单位阵（互不相关）",
    )
    allocate_input: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="待分配标准不确定度的输入量名称（须在 inputs 中声明）",
    )
    target_combined_uncertainty: TargetUncertainty = Field(
        ...,
        description="目标合成标准不确定度（带单位，须与公式输出同量纲）",
    )


class CorrelationTerm(BaseModel):
    """与另一输入量 j 的协方差对 i 的不确定度贡献项。"""

    input: str
    correlation: float
    covariance_contribution: float
    unit: str


class VarianceContribution(BaseModel):
    """按输入量分解的方差贡献：u_i^2 项 + 全部相关项。"""

    input: str
    variance_contribution: float = Field(..., description="u_i^2 项，(c_i u_i)^2")
    unit: str
    share: float = Field(..., description="占总方差的比例，相关项分摊后各项之和为 1")
    correlation_terms: list[CorrelationTerm]


class QuantityResult(BaseModel):
    value: float
    unit: str
    standard_uncertainty: float


class SensitivityCoefficient(BaseModel):
    input: str
    value: float
    unit: str


class PropagationResponse(BaseModel):
    result: QuantityResult
    number_of_inputs: int = Field(..., description="参与传播的输入量数量")
    correlation_matrix: list[list[float]]
    sensitivity_coefficients: list[SensitivityCoefficient]
    variance_contributions: list[VarianceContribution]
    total_variance: float
    total_variance_unit: str
    formula: str
    whitelist: list[str]


class JointOutputResult(BaseModel):
    """单个输出量的结果：名称、值、单位、合成标准不确定度。"""

    name: str
    value: float
    unit: str
    standard_uncertainty: float


class CovarianceCell(BaseModel):
    """输出协方差矩阵单元：数值 + 单位（输出 i 单位 × 输出 j 单位）。"""

    value: float
    unit: str


class OutputSensitivities(BaseModel):
    """单个输出量对各输入量的灵敏度系数（雅可比矩阵的一行）。"""

    output: str
    coefficients: list[SensitivityCoefficient]


class JointPropagationResponse(BaseModel):
    outputs: list[JointOutputResult]
    number_of_inputs: int = Field(..., description="参与传播的输入量数量")
    number_of_outputs: int = Field(..., description="参与联合传播的输出量数量")
    correlation_matrix: list[list[float]] = Field(
        ..., description="输入量相关系数矩阵（回显，顺序与 inputs 一致）"
    )
    sensitivity_coefficients: list[OutputSensitivities]
    output_covariance_matrix: list[list[CovarianceCell]] = Field(
        ...,
        description="输出协方差矩阵 JΣJᵀ；单元 (i,j) 的单位 = 输出 i 单位 × 输出 j 单位",
    )
    output_correlation_matrix: list[list[float]] = Field(
        ...,
        description="输出相关系数矩阵；某输出标准不确定度为 0 时其非对角相关系数无定义，置 0",
    )
    whitelist: list[str]


class VarianceQuadraticCoefficients(BaseModel):
    """总方差关于未知标准不确定度 u 的二次式 Var(u) = a u^2 + b u + c。"""

    a: float = Field(..., description="二次项系数（=c_k^2，矩阵对角元恒为 1，故 a>0）")
    a_unit: str
    b: float = Field(..., description="一次项系数（含全部相关项）")
    b_unit: str
    c: float = Field(..., description="常数项（其余输入量在未知量取 0 时的总方差）")
    c_unit: str
    unknown_uncertainty_unit: str = Field(
        ..., description="未知标准不确定度 u 的单位（即待分配输入量自身单位）"
    )


class AllocationResponse(PropagationResponse):
    """不确定度逆分配结果：代回求得的最大不确定度后，结构与传播响应一致并附分配信息。"""

    allocate_input: str = Field(..., description="被分配标准不确定度的输入量名称")
    allocated_standard_uncertainty: float = Field(
        ..., description="满足目标的最大非负标准不确定度，单位与该输入量声明单位一致"
    )
    allocated_standard_uncertainty_unit: str
    target_combined_uncertainty: TargetUncertaintyEcho = Field(
        ..., description="回显目标合成标准不确定度（值/单位）"
    )
    variance_quadratic: VarianceQuadraticCoefficients


# -- 蒙特卡洛传播 ---------------------------------------------------------


class MonteCarloPropagationRequest(BaseModel):
    """蒙特卡洛传播校核请求：inputs/formula/correlation_matrix 与传播接口一致。"""

    inputs: list[InputQuantity] = Field(..., min_length=1)
    formula: str = Field(..., min_length=1, description="仅含白名单运算的测量公式")
    correlation_matrix: list[list[FiniteFloat]] | None = Field(
        default=None,
        description="输入量相关系数矩阵，顺序与 inputs 一致；须半正定（允许奇异），缺省为单位阵",
    )
    samples: int = Field(
        ...,
        ge=1000,
        le=100_000,
        description="蒙特卡洛抽样次数（1000～100000）",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=2**63 - 1,
        description="可选随机数种子（非负整数）；给定后结果可复现，缺省则每次不同",
    )


class Quantiles(BaseModel):
    """输出量样本的 2.5% / 50% / 97.5% 分位数（线性插值）。"""

    q025: float = Field(..., serialization_alias="2.5%", description="2.5% 分位数")
    q50: float = Field(..., serialization_alias="50%", description="中位数（50% 分位数）")
    q975: float = Field(..., serialization_alias="97.5%", description="97.5% 分位数")

    model_config = {"populate_by_name": True}


class FirstOrderSummary(BaseModel):
    """同一请求的 GUM 一阶传播结果摘要。"""

    value: float = Field(..., description="一阶传播在输入估计值处的模型输出值")
    unit: str
    standard_uncertainty: float = Field(
        ..., description="一阶合成标准不确定度 u_c（线性化近似）"
    )


class DeviationSummary(BaseModel):
    """蒙特卡洛结果相对一阶传播结果的偏差（绝对差与相对差）。"""

    mean_absolute: float = Field(
        ..., description="样本均值与一阶结果值之差（蒙特卡洛 − 一阶）"
    )
    mean_relative: float | None = Field(
        ..., description="均值相对偏差；一阶结果值为 0 时无定义（null）"
    )
    standard_uncertainty_absolute: float = Field(
        ..., description="样本标准差与一阶 u_c 之差（蒙特卡洛 − 一阶）"
    )
    standard_uncertainty_relative: float | None = Field(
        ..., description="标准不确定度相对偏差；一阶 u_c 为 0 时无定义（null）"
    )


class MonteCarloPropagationResponse(BaseModel):
    """蒙特卡洛传播响应：样本统计量、分位数及相对一阶传播结果的偏差。"""

    result: QuantityResult = Field(
        ...,
        description="样本均值（value）、单位与样本标准不确定度（标准差，ddof=1）",
    )
    quantiles: Quantiles
    samples: int = Field(..., description="实际参与统计的抽样次数")
    seed: int | None = Field(..., description="回显随机数种子；未指定时为 null")
    number_of_inputs: int
    correlation_matrix: list[list[float]] = Field(
        ..., description="输入量相关系数矩阵（回显，顺序与 inputs 一致）"
    )
    first_order: FirstOrderSummary = Field(
        ..., description="同一模型的 GUM 一阶传播结果，供对照校核"
    )
    deviation: DeviationSummary = Field(
        ..., description="蒙特卡洛统计量相对一阶传播结果的偏差"
    )
    formula: str
    whitelist: list[str]


# -- 规格符合性判定 ---------------------------------------------------------


class SpecLimit(BaseModel):
    """带单位的规格界限（下/上限共用）；单位须与公式输出同量纲。"""

    value: FiniteFloat
    unit: str = Field(..., min_length=1, description="界限单位，须与公式输出同量纲")


class ConformityRequest(BaseModel):
    """规格符合性判定请求：传播输入 + 至少一个带单位规格界限 + 置信阈值 t。"""

    inputs: list[InputQuantity] = Field(..., min_length=1)
    formula: str = Field(..., min_length=1, description="仅含白名单运算的测量公式")
    correlation_matrix: list[list[FiniteFloat]] | None = Field(
        default=None,
        description="输入量相关系数矩阵，顺序与 inputs 一致；缺省为单位阵（互不相关）",
    )
    lower_limit: SpecLimit | None = Field(
        default=None, description="规格下限（带单位）；与上限至少给出一个"
    )
    upper_limit: SpecLimit | None = Field(
        default=None, description="规格上限（带单位）；与下限至少给出一个"
    )
    confidence_threshold: float = Field(
        ...,
        gt=0.5,
        lt=1.0,
        allow_inf_nan=False,
        description="置信阈值 t（0.5 < t < 1）",
    )

    @model_validator(mode="after")
    def _require_at_least_one_limit(self) -> "ConformityRequest":
        if self.lower_limit is None and self.upper_limit is None:
            raise PydanticCustomError(
                "missing_spec_limits",
                "至少给出一个带单位的规格界限（lower_limit 或 upper_limit）",
            )
        return self


class SpecLimitEcho(BaseModel):
    """换算到结果显示单位后的规格界限。"""

    value: float
    unit: str


class AcceptanceInterval(BaseModel):
    """固定 u_c 时使 p ≥ t 的全部 y 构成的闭接受区间（以结果显示单位计）。"""

    lower: float | None = Field(..., description="下边界；null 表示 −∞")
    upper: float | None = Field(..., description="上边界；null 表示 +∞")
    unit: str
    is_empty: bool = Field(
        ...,
        description="双侧界限下中点最大概率仍低于阈值时为空（lower/upper 均为 null）",
    )


class ConformityResponse(BaseModel):
    """规格符合性判定响应：概率 p、判定、一阶结果、换算界限与接受区间。"""

    decision: Literal["pass", "fail", "indeterminate"] = Field(
        ...,
        description="p≥t 合格 pass；p≤1−t 不合格 fail；其余待定 indeterminate",
    )
    probability: float = Field(..., description="真值落入闭规格区间的概率 p")
    confidence_threshold: float = Field(..., description="回显置信阈值 t")
    result: QuantityResult = Field(
        ..., description="一阶传播结果：y、单位与合成标准不确定度 u_c"
    )
    lower_limit: SpecLimitEcho | None = Field(
        ..., description="换算到结果显示单位的规格下限；未给出时为 null"
    )
    upper_limit: SpecLimitEcho | None = Field(
        ..., description="换算到结果显示单位的规格上限；未给出时为 null"
    )
    acceptance_interval: AcceptanceInterval
    formula: str


# -- A 类标准不确定度（重复读数）-------------------------------------------


class RepeatabilityRequest(BaseModel):
    """等间隔重复读数的 A 类标准不确定度估计请求。"""

    readings: list[FiniteFloat] = Field(
        ...,
        min_length=3,
        description="同一被测量的等间隔重复读数（至少 3 个，同一单位；拒绝 NaN/Inf）",
    )
    unit: str = Field(..., min_length=1, description="全部读数的共同单位，如 V / m / mV")
    max_lag: int = Field(
        ...,
        ge=0,
        description="自协方差最大滞后 L，须满足 0 ≤ L < 读数数量 n",
    )


class Autocovariance(BaseModel):
    """单个滞后 k 的有偏自协方差估计 γ_k（除以 n）。"""

    lag: int = Field(..., description="滞后阶数 k（0 ≤ k ≤ L）")
    value: float = Field(..., description="γ_k，1/n 归一化的样本自协方差")
    unit: str = Field(..., description="自协方差单位 = 读数单位的平方，如 V ** 2")


class RepeatabilityResponse(BaseModel):
    """A 类标准不确定度估计响应：均值、u_A、各滞后自协方差与有效样本量。"""

    result: QuantityResult = Field(
        ...,
        description="读数均值（value）、单位与 A 类标准不确定度 u_A；"
        "standard_uncertainty 可直接填入 /api/v1/propagate 的 standard_uncertainty",
    )
    number_of_readings: int = Field(..., description="读数数量 n")
    max_lag: int = Field(..., description="回显最大滞后 L")
    autocovariances: list[Autocovariance] = Field(
        ..., description="各滞后 γ_k（k=0..L），1/n 归一化"
    )
    variance: float = Field(
        ...,
        description="均值方差 u_A² = [γ_0 + 2 Σ_{k=1}^L (1−k/(L+1)) γ_k] / n",
    )
    variance_unit: str = Field(..., description="u_A² 的单位 = 读数单位的平方")
    effective_sample_size: float | None = Field(
        ...,
        description="有效样本量 n_eff = n γ_0 / [γ_0 + 2 Σ…]；分母为零时为 null",
    )
