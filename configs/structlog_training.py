import structlog

structlog.configure()  # can even skip config for now, has sane defaults
logger = structlog.get_logger()

logger.error("something happened", key1="value1", key2='value2', count=5)
log = logger.bind(request_id="abc-123", key1="value1", key2='value2', count=5)
log.info("step one")     # includes request_id=abc-123
log.info("step two")     # also includes request_id=abc-123, automatically