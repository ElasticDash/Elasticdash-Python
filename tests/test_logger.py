import os

from elasticdash import ElasticDash
from elasticdash.logger import elasticdash_logger

"""
Level	Numeric value
logging.DEBUG	10
logging.INFO	20
logging.WARNING	30
logging.ERROR	40
"""


def test_default_elasticdash():
    ElasticDash()

    assert elasticdash_logger.level == 30


def test_via_env():
    os.environ["ELASTICDASH_DEBUG"] = "True"

    ElasticDash()

    assert elasticdash_logger.level == 10

    os.environ.pop("ELASTICDASH_DEBUG")


def test_debug_elasticdash():
    ElasticDash(debug=True)
    assert elasticdash_logger.level == 10

    # Reset
    elasticdash_logger.setLevel("WARNING")
