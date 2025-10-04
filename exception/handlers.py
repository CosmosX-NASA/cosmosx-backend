# error/handlers.py
from log import logger
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_404_NOT_FOUND,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from exception.exceptions import (
    AppError,
    ImageNotFoundError,
)

# HTTP 상태코드 매핑
STATUS_BY_TYPE: dict[type[AppError], int] = {
    ImageNotFoundError: HTTP_404_NOT_FOUND,
}


def _status_for_app_error(exc: AppError) -> int:
    for cls, status in STATUS_BY_TYPE.items():
        if isinstance(exc, cls):
            return status
    return HTTP_400_BAD_REQUEST


def _problem(detail: str, *, code: str, status: int, request: Request, context: dict | None = None):
    payload = {
        "detail": detail,
        "code": code,
        "path": request.url.path,
    }
    if context:
        payload["context"] = context  # 외부 노출 OK한 정보만
    return JSONResponse(status_code=status, content=payload)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        status = _status_for_app_error(exc)
        # 상태코드에 따라 로깅 레벨 차등
        if 400 <= status < 500:
            logger.info("app_error", extra={"path": request.url.path,
                                            "context": exc.context, "code": exc.code})
        else:
            logger.warning("app_error", extra={"path": request.url.path,
                                               "context": exc.context, "code": exc.code})
        return _problem(str(exc), code=exc.code, status=status, request=request, context=exc.context)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.error(
            "unhandled_exception",
            extra={
                "path": request.url.path,
                "code": "INTERNAL_SERVER_ERROR",
                "context": {"error": str(exc)},
            },
        )
        # 내부 오류는 상세를 숨기고 일반화된 메시지
        return _problem("Internal server error", code="INTERNAL_SERVER_ERROR",
                        status=HTTP_500_INTERNAL_SERVER_ERROR, request=request)
