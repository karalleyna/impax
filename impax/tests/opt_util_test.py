import jax
import jax.numpy as jnp
import os
import sys
from opt_util import *

ldif_root = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.append(ldif_root)
print(ldif_root)
import ldif.ldif.util.opt_util as opt_util_tf
import unittest

grads_and_vars = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
print(clip_by_global_norm(grads_and_vars))