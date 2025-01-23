import numpy
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(filename="log.txt", level=logging.INFO)


def test_fn():
    var = 2

    logger.info(f"AMAN! {var} executed")
