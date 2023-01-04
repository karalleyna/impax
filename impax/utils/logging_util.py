"""
Utilities for logging to console and/or additional files.
References:
https://github.com/google/ldif/blob/master/ldif/util/logging_util.py
"""

from impax.utils.base.log import LOG


def log(msg, level="info"):
    log.info(f"ldif {level.upper()}: {msg}")
    LOG.log(msg, level)
