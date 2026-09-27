"""领域错误类型与 FastAPI 异常处理器。

所有输入/语义层面的校验失败统一抛出 :class:`UnprocessableError`，
由 FastAPI 处理器转换为 422 响应，响应体带可定位的 loc 字段。
"""
from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class UnprocessableError(Exception):
    """业务规则校验失败（HTTP 422）。

    loc 指向出错位置，例如 ``["inputs", 0, "unit"]``、
    ``["correlation_matrix", 1, 2]`` 或 ``["formula"]``。
    """

    def __init__(
        self,
        message: str,
        loc: list[str | int],
        code: str = "invalid",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.loc = loc
        self.code = code
        self.details = details


def _error_response(
    code: str,
    message: str,
    loc: list[Any] | None = None,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "loc": loc or [],
    }
    if details is not None:
        error["details"] = details
    return JSONResponse(status_code=422, content={"error": error})


async def unprocessable_handler(_: Request, exc: UnprocessableError) -> JSONResponse:
    return _error_response(exc.code, exc.message, exc.loc, exc.details)


async def request_validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """把 Pydantic 的字段校验错误规整为与业务错误一致的 422 结构。"""
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = [x for x in first.get("loc", []) if x != "body"]
    message = first.get("msg", "请求体校验失败")
    return _error_response(first.get("type", "validation_error"), message, loc)
