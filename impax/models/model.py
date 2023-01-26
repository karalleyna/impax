"""Example models."""

import jax.numpy as jnp
from jax import random, vmap

import flax.linen as nn

from typing import Any
from logging import log

import ml_collections

# local
from impax.networks.cnn import EarlyFusionCNN, MidFusionCNN
from impax.networks.occnet import Decoder
from impax.networks.pointnet import PointNet

from impax.representations import structured_implicit_function

from impax.models.prediction import Prediction

from impax.utils import geom_util
from impax.utils.model_utils import TwinInference


class PointEncoder(nn.Module):
    output_dim: int
    maxpool_feature: int
    model_config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, points) -> Any:

        """Encodes a point cloud (either with or without normals) to an embedding."""
        assert len(points.shape) == 3
        # TODO(kgenova) This could reshape to the batch dimension to support
        batch_size = points.shape[0]
        embedding = PointNet(
            self.output_dim,
            self.model_config.maxpool_feature,
            self.model_config.to_64d,
        )(points)

        embedding = jnp.reshape(embedding, [batch_size, self.output_dim])
        if len(embedding.shape) != 2 or embedding.shape[-1] != self.output_dim:
            raise ValueError(f"Unexpected output shape: {embedding.shape}")
        return embedding


class PointNetSIFPredictor(nn.Compact):
    element_count: int
    element_length: int
    model_config: ml_collections.ConfigDict
    max_encodable: int = 1024

    @nn.compact
    def __call__(self, observation, key: random.PRNGKey = random.PRNGKey(0)):

        """A cnn that maps 1+ images with 1+ chanels to a feature vector."""
        inputs = jnp.concatenate(
            [observation.surface_points, observation.normals], axis=-1
        )
        batch_size, sample_count = inputs.shape[:2]
        if sample_count > self.max_encodable:
            sample_indices = random.randint(
                key,
                [batch_size, sample_count],
                minval=0,
                maxval=sample_count - 1,
                dtype=jnp.int32,
            )
            inputs = jnp.take_along_axis(inputs, sample_indices, axis=0)
        embedding = PointEncoder(
            self.element_count * self.element_length, self.model_config
        )(inputs)
        batch_size = inputs.shape[0]
        prediction = jnp.reshape(
            embedding, [batch_size, self.element_count, self.element_length]
        )
        return prediction, embedding


class StructuredImplicitModel(nn.Module):
    """An instance of a network that predicts a structured implicit function."""

    model_config: ml_collections.ConfigDict
    _forward_call_count: int = 0
    _eval_implicit_parameters_call_count: int = 0
    _enable_deprecated: bool = False

    @nn.compact
    def _global_local_forward(self, observation):
        """A forward pass that include both template and element inference."""

        explicit_element_length = structured_implicit_function.element_explicit_dof(
            self.model_config
        )
        implicit_embedding_length = structured_implicit_function.element_implicit_dof(
            self.model_config
        )
        num_elements = self.model_config.num_shape_elements

        if explicit_element_length <= 0:
            raise ValueError(
                "Invalid element length. Embedding has length "
                "%i, but total length is only %i."
                % (implicit_embedding_length, explicit_element_length)
            )

        if self.model_config.model_architecture == "efcnn":
            self.inference_model = EarlyFusionCNN(
                num_elements, explicit_element_length
            )(jnp.array(observation.tensor))
        elif self.model_config.model_architecture == "mfcnn":
            self.inference_model = MidFusionCNN(num_elements, explicit_element_length)(
                jnp.array(observation.tensor),
                jnp.array(observation.cam_to_worlds),
            )

        elif self.model_config.model_architecture == "pn":
            self.inference_model = PointNetSIFPredictor(
                num_elements, explicit_element_length, self._model_config
            )
        else:
            raise ValueError(
                "Invalid StructuredImplicitModel architecture hparam: %s"
                % self.model_config.model_architecture
            )
        if self.model_config.enable_implicit_parameters:
            self.single_element_implicit_eval_fun = Decoder
        else:
            self.single_element_implicit_eval_fun = None

        batch_size = self.model_config.batch_size
        num_shape_elements = self.model_config.num_shape_elements
        sampling_scheme = self.model_config.sampling_scheme
        implicit_parameter_length = self.model_config.implicit_parameter_length

        explicit_parameters, explicit_embedding = self.inference_model(observation)
        sif = structured_implicit_function.StructuredImplicit.from_activation(
            self._model_config, explicit_parameters, self
        )
        # Now we can compute world2local
        world2local = sif.world2local

        if implicit_parameter_length > 0:

            (local_points, local_normals, _, _,) = geom_util.local_views_of_shape(
                observation.surface_points,
                world2local,
                local_point_count=self.model_config.num_local_points,
                global_normals=observation.normals,
            )
            # Output shapes are both [B, EC, LPC, 3].
            if "n" not in sampling_scheme:
                flat_point_features = jnp.reshape(
                    local_points,
                    [
                        batch_size * num_shape_elements,
                        self.model_config.num_local_points,
                        3,
                    ],
                )
            else:
                flat_point_features = jnp.reshape(
                    jnp.concatenate([local_points, local_normals], axis=-1),
                    [
                        batch_size * num_shape_elements,
                        self.model_config.num_local_points,
                        6,
                    ],
                )
            encoded_iparams = PointEncoder(
                flat_point_features,
                implicit_parameter_length,
                self.model_config,
            )
            iparams = jnp.reshape(
                encoded_iparams,
                [
                    batch_size,
                    num_shape_elements,
                    implicit_parameter_length,
                ],
            )
            sif.set_iparams(iparams)
            embedding = jnp.concatenate(
                [
                    explicit_embedding,
                    jnp.reshape(iparams, [batch_size, -1]),
                ],
                axis=-1,
            )
        else:
            embedding = explicit_embedding
        self._forward_call_count += 1
        return Prediction(self.model_config, observation, sif, embedding)

    @nn.compact
    def __call__(self, observation):
        """Evaluates the explicit and implicit parameter vectors as a Prediction."""
        implicit_architecture = self.model_config.implicit_architecture
        if implicit_architecture == "p":
            return self._global_local_forward(observation)
        element_count = self.model_config.num_shape_elements
        flat_element_length = structured_implicit_function.element_dof(
            self.model_config
        )
        if implicit_architecture == "1":
            structured_implicit_activations, embedding = self.inference_model(
                observation,
                element_count,
                flat_element_length,
                self.model_config,
            )
        elif implicit_architecture == "2":
            (structured_implicit_activations, embedding,) = TwinInference(
                self.inference_model,
                observation,
                element_count,
                flat_element_length,
                self.model_config,
            )
        else:
            raise ValueError(f"Invalid value for {implicit_architecture}")

        self._forward_call_count += 1
        structured_implicit = (
            structured_implicit_function.StructuredImplicit.from_activation(
                self.model_config, structured_implicit_activations, self
            )
        )
        return Prediction(
            self.model_config, observation, structured_implicit, embedding
        )

    def eval_implicit_parameters(self, implicit_parameters, samples):
        """Decodes each implicit parameter vector at each of its sample points.

        Args:
          implicit_parameters: Tensor with shape [batch_size, element_count,
            element_embedding_length]. The embedding associated with each element.
          samples: Tensor with shape [batch_size, element_count, sample_count, 3].
            The sample locations. Each embedding vector will be decoded at each of
            its sample locations.

        Returns:
          Tensor with shape [batch_size, element_count, sample_count, 1]. The
            decoded value of each element's embedding at each of the samples for
            that embedding.
        """
        # Each element has its own network:
        if self.single_element_implicit_eval_fun is None:
            raise ValueError("The implicit decoder function is None.")
        implicit_param_shape_in = implicit_parameters.shape
        log.info(f"BID0: Input implicit param shape: {implicit_param_shape_in}")
        log.info(f"BID0: Input samples shape: {samples.shape}")
        # TODO(kgenova) Now that batching OccNet is supported, batch this call.
        if self._enable_deprecated:
            log.info("Deprecated eval.")
            vals = self._deprecated_multielement_eval(implicit_parameters, samples)
        else:
            (
                batch_size,
                element_count,
                element_embedding_length,
            ) = implicit_parameters.shape
            sample_count = samples.shape[-2]
            batched_parameters = jnp.reshape(
                implicit_parameters,
                [batch_size * element_count, element_embedding_length],
            )
            batched_samples = jnp.reshape(
                samples, [batch_size * element_count, sample_count, 3]
            )
            if self.model_config.seperate_network:
                raise ValueError(
                    "Incompatible hparams. Must use _deprecated_multielement_eval"
                    "if requesting separate network weights per shape element."
                )

            batched_vals = self.single_element_implicit_eval_fun(
                batched_parameters,
                batched_samples,
                apply_sigmoid=False,
                model_config=self.model_config,
            )
            vals = jnp.reshape(
                batched_vals, [batch_size, element_count, sample_count, 1]
            )
        self._eval_implicit_parameters_call_count += 1
        return vals

    def _deprecated_multielement_eval(self, implicit_parameters, samples):
        """An eval provided for backwards compatibility."""
        evals = vmap(self.single_element_implicit_eval_fun, in_axes=(1, 1, None, None))(
            implicit_parameters, samples
        )
        return evals
