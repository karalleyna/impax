import jax.numpy as jnp

# local
from impax.configs import autoencoder
from impax.configs import tensorflow
from impax.representations import structured_implicit_functions
from impax.training import metrics

from ldif.ldif.representation import structured_implicit_function as orig_sdf
from ldif.ldif.training import metrics as orig


def test_point_iou():

    net = lambda *args, **kwargs: None

    constants = jnp.ones((quadric_count, 1))
    centers = jnp.ones((quadric_count, 3))
    radii = jnp.ones((quadric_count, 3))
    iparams = jnp.ones((3, quadric_count, 3))

    sample_locations = jnp.ones((1, 16, 3))
    model_config = tensorflow.get_config()
    tf_structure_imp = orig_sdf.StructuredImplicit(
        model_config, constants, centers, radii, iparams, net
    )
    gt = orig.point_iou(
        tf_structure_imp, sample_locations, sample_locations, model_config
    )

    model_config = autoencoder.get_config()
    structured_implicit = structured_implicit_functions.StructuredImplicit(
        constants, centers, radii, iparams, net, model_config
    )
    ret = metrics.point_iou(
        structured_implicit, sample_locations, sample_locations, model_config
    )

    assert jnp.allclose(gt.numpy(), ret)
