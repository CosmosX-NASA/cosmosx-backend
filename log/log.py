import logging
import logging.config
import os


def _ensure_log_fields(record: logging.LogRecord) -> logging.LogRecord:
    if not hasattr(record, "code"):
        record.code = "-"
    if not hasattr(record, "path"):
        record.path = "-"
    if not hasattr(record, "context"):
        record.context = "-"
    return record


class CosmosLogger(logging.getLoggerClass()):
    def makeRecord(
        self,
        name,
        level,
        fn,
        lno,
        msg,
        args,
        exc_info,
        func=None,
        extra=None,
        sinfo=None,
    ):
        override = {}
        forwarded_extra = extra
        if extra:
            forwarded_extra = extra.copy()
            for key in ("code", "path", "context"):
                if key in forwarded_extra:
                    override[key] = forwarded_extra.pop(key)

        record = super().makeRecord(
            name,
            level,
            fn,
            lno,
            msg,
            args,
            exc_info,
            func=func,
            extra=forwarded_extra,
            sinfo=sinfo,
        )

        for key, value in override.items():
            setattr(record, key, value)

        return _ensure_log_fields(record)


logging.setLoggerClass(CosmosLogger)
logging.root.__class__ = CosmosLogger


class ContextFilter(logging.Filter):
    def filter(self, record):
        _ensure_log_fields(record)
        return True


if not os.path.exists("logs"):
    os.makedirs("logs", exist_ok=True)

logging_config_path = os.path.join(os.path.dirname(__file__), "logging.conf")
logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)

_root_logger = logging.getLogger()
_has_context_filter = any(isinstance(f, ContextFilter)
                          for f in _root_logger.filters)
if not _has_context_filter:
    _root_logger.addFilter(ContextFilter())

logger = logging.getLogger(__name__)
