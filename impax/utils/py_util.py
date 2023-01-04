"""
General python utility functions.
References:
https://github.com/google/ldif/blob/master/ldif/util/py_util.py
"""

import contextlib
import functools
import os
import shutil
import subprocess as sp
import tempfile

import numpy as np


def compose(*fs):
    composition = lambda f, g: lambda x: f(g(x))
    identity = lambda x: x
    return functools.reduce(composition, fs, identity)


def maybe(x, f):
    """Returns [f(x)], unless f(x) raises an exception. In that case, []."""
    try:
        result = f(x)
        output = [result]
    # pylint:disable=broad-except
    except Exception:
        # pylint:enable=broad-except
        output = []
    return output


@contextlib.contextmanager
def py2_temporary_directory():
    d = tempfile.mkdtemp()
    try:
        yield d
    finally:
        shutil.rmtree(d)


@contextlib.contextmanager
def x11_server():
    """Generates a headless x11 target to use."""
    idx = np.random.randint(10, 10000)
    prev_display_name = os.environ["DISPLAY"]
    x11 = sp.Popen("Xvfb :%i" % idx, shell=True)
    os.environ["DISPLAY"] = ":%i" % idx
    try:
        yield idx
    finally:
        x11.kill()
        os.environ["DISPLAY"] = prev_display_name


def merge(x, y):
    z = x.copy()
    z.update(y)
    return z


def merge_into(x, ys):
    out = []
    for y in ys:
        out.append(merge(x, y))
    return
