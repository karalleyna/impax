import jax
import jax.numpy as jnp
import pytest
from ml_collections import ConfigDict

import impax.utils.sdf_util as sdf_util
import ldif.ldif.util.sdf_util as original


@pytest.mark.parametrize("seed", [2, 4, 8, 16])
def test_apply_class_transfer(seed):

    sdf = jax.random.normal(key=jax.random.PRNGKey(seed), shape=(16, 32, 32, 32))

    offset = 0.3
    dtype = None
    soft_transfer = True

    cfg = {"hparams": {"lhdn": False, "hdn": 100}}

    model_config = ConfigDict(cfg)
    org = original.apply_class_transfer(sdf, model_config, soft_transfer, offset, dtype)

    ret = sdf_util.apply_class_transfer(
        sdf, soft_transfer, offset, model_config.hparams.lhdn, model_config.hparams.hdn, dtype
    )

    assert jnp.allclose(org.numpy(), ret)


@pytest.mark.parametrize("seed", [2, 4, 8, 16])
def test_apply_density_transfer(seed):
    sdf = jax.random.normal(key=jax.random.PRNGKey(seed), shape=(16, 32, 32, 32))
    gt = original.apply_density_transfer(sdf)
    ret = sdf_util.apply_density_transfer(sdf)

    assert jnp.allclose(gt.numpy(), ret)
