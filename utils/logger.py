import logging

try:
    from backend.config import LOG_DIR, LOG_FILE
except ImportError:
    from config import LOG_DIR, LOG_FILE


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(
            logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        )
        logger.addHandler(ch)

        # File handler (best-effort; keep console logging even if file setup fails).
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
            )
            logger.addHandler(fh)
        except OSError:
            logger.warning("File logging disabled: unable to create log file at %s", LOG_FILE)

    return logger