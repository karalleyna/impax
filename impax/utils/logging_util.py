"""
Utilities for logging to console and/or additional files.
References:
https://github.com/google/ldif/blob/master/ldif/util/logging_util.py
"""

import logging
from logging import Logger

log = Logger("impax_logger")
log.setLevel(logging.INFO)
