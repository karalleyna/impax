"""
Utilities for logging to console and/or additional files.
References:
https://github.com/google/ldif/blob/master/ldif/util/logging_util.py
"""

from impax.utils.base.log import Log


def log(msg, level="info"):
    Log.log(Log, f"ldif {level.upper()}: {msg}")
    Log.log(Log, msg, level)
