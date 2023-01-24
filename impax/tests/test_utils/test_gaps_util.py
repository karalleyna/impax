import jax
from jax import random

from impax.utils import gaps_util


def test_gaps():

    pts = jax.random.uniform(key=random.PRNGKey(0), shape=(32, 3))

    rt = gaps_util.ptsview(pts, mesh=None, camera="fixed")
