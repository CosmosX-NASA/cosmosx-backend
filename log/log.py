import logging.config
import os


class ContextFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "code"):
            record.code = "-"
        if not hasattr(record, "path"):
            record.path = "-"
        if not hasattr(record, "context"):
            record.context = "-"
        return True


if not os.path.exists('logs'):
    os.makedirs('logs', exist_ok=True)

# 로깅 설정 파일 경로
logging_config_path = os.path.join(os.path.dirname(__file__), 'logging.conf')
logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)

logger = logging.getLogger(__name__)
