"""
General python utility functions.
References:
https://github.com/google/ldif/blob/master/ldif/util/py_util.py
"""

import contextlib
import shutil
import tempfile


@contextlib.contextmanager
def py2_temporary_directory():
    d = tempfile.mkdtemp()
    try:
        yield d
    finally:
        shutil.rmtree(d)


def merge(x, y):
    z = x.copy()
    z.update(y)
    return z


def merge_into(x, ys):
    out = []
    for y in ys:
        out.append(merge(x, y))
    return out
