"""Pint 单位注册表与单位辅助函数。"""
from __future__ import annotations

import pint
from pint import DimensionalityError, OffsetUnitCalculusError, UndefinedUnitError

from .errors import UnprocessableError

# 独立注册表，避免依赖全局 default_registry。
ureg = pint.UnitRegistry()
Quantity = ureg.Quantity


def parse_unit(text: str, loc: list[str | int]) -> pint.Unit:
    """解析单位字符串；失败时抛出带定位的 422。"""
    text = text.strip()
    if not text:
        raise UnprocessableError("单位不能为空", loc, code="empty_unit")
    try:
        return ureg.parse_expression(text).units
    except (UndefinedUnitError, AttributeError, TypeError, ValueError, SyntaxError) as exc:
        raise UnprocessableError(
            f"无法识别的单位 {text!r}：{exc}", loc, code="unknown_unit"
        ) from exc


def make_quantity(magnitude: float, unit: pint.Unit) -> Quantity:
    """构造带单位量；量纲约束导致的错误转为 422。"""
    try:
        return Quantity(magnitude, unit)
    except (DimensionalityError, OffsetUnitCalculusError) as exc:
        raise UnprocessableError(f"单位 {unit:~} 与数值不兼容：{exc}", ["unit"], code="incompatible_unit") from exc


def same_dimensionality(a: pint.Unit, b: pint.Unit) -> bool:
    return a.dimensionality == b.dimensionality


def convert_magnitude(q: Quantity, target: pint.Unit, loc: list[str | int]) -> float:
    """把量 q 换算到 target 单位并返回数值；量纲不兼容时抛 422。"""
    try:
        return float(q.to(target).magnitude)
    except (DimensionalityError, OffsetUnitCalculusError) as exc:
        raise UnprocessableError(
            f"单位不兼容：{q.units:~} 无法换算为 {target:~}：{exc}",
            loc,
            code="incompatible_unit",
        ) from exc


def unit_str(unit: pint.Unit) -> str:
    """单位显示字符串；无量纲返回 dimensionless。"""
    rendered = f"{unit:~}".strip()
    return rendered if rendered else "dimensionless"


# 结果单位显示时优先匹配的常用导出单位（按校准领域常见度排列）。
_PREFERRED_UNITS = [
    "W", "J", "N", "Pa", "Hz", "C", "V", "ohm", "S", "F", "T", "Wb", "H",
    "lx",
]


def preferred_unit(unit: pint.Unit) -> pint.Unit:
    """把任意单位乘积规约为同量纲的常用导出单位（如 A·V -> W）。

    无匹配时原样返回。
    """
    target_dim = unit.dimensionality
    best = unit
    best_len = len(unit_str(unit))
    for name in _PREFERRED_UNITS:
        try:
            candidate = ureg.parse_units(name)
        except UndefinedUnitError:
            continue
        if candidate.dimensionality == target_dim:
            rendered_len = len(name)
            if rendered_len < best_len:
                best, best_len = candidate, rendered_len
    return best
