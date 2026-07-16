import logging

from app.core.config import LogFormat, Settings
from app.core.request_context import get_request_id

APPLICATION_LOGGER_NAME = "pcb_aoi"

_APPLICATION_HANDLER_MARKER = "_pcb_aoi_application_handler"
_PLAIN_LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "[service=%(service_name)s request_id=%(request_id)s] %(message)s"
)


class RequestContextFilter(logging.Filter):
    def __init__(self, service_name: str) -> None:
        super().__init__()
        self.service_name = service_name

    def filter(self, record: logging.LogRecord) -> bool:
        record.service_name = self.service_name
        record.request_id = get_request_id() or "-"
        return True


def _build_formatter(log_format: LogFormat) -> logging.Formatter:
    """Build the selected formatter; future formats can be added here."""
    if log_format is LogFormat.PLAIN:
        return logging.Formatter(
            fmt=_PLAIN_LOG_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    raise ValueError(f"Unsupported log format: {log_format}")


def is_application_handler(handler: logging.Handler) -> bool:
    """Return whether a handler is managed by this application."""
    return bool(getattr(handler, _APPLICATION_HANDLER_MARKER, False))


def configure_logging(settings: Settings) -> logging.Logger:
    """Configure and return the application logger without duplicate handlers."""
    logger = logging.getLogger(APPLICATION_LOGGER_NAME)
    logger.setLevel(settings.log_level.value)
    logger.propagate = False

    managed_handlers = [
        handler for handler in logger.handlers if is_application_handler(handler)
    ]
    if managed_handlers:
        handler = managed_handlers[0]
        for duplicate in managed_handlers[1:]:
            logger.removeHandler(duplicate)
            duplicate.close()
    else:
        handler = logging.StreamHandler()
        setattr(handler, _APPLICATION_HANDLER_MARKER, True)
        logger.addHandler(handler)

    handler.setLevel(settings.log_level.value)
    handler.setFormatter(_build_formatter(settings.log_format))
    for existing_filter in list(handler.filters):
        if isinstance(existing_filter, RequestContextFilter):
            handler.removeFilter(existing_filter)
    handler.addFilter(RequestContextFilter(settings.application_name))

    return logger
