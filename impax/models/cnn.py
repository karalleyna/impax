"""A simple feed forward CNN."""
import functools
from typing import Any

from flax import linen as nn
import jax.numpy as jnp

# local
from impax.utils.logging_util import log
from impax.models.resnet_v2 import ResNet

class EarlyFusionCNN(nn.Module):
    """A CNN that maps 1+ images with 1+ chanels to a feature vector."""
    element_count: int
    element_length: int
    architecture: str = 'r18'
    use_running_average: bool = True
    @nn.compact
    def __call__(self, x) -> Any:
        batch_size =x.shape[0]
        if self.architecture == 'cnn':
            x = net_util.inputs_to_feature_vector(x, 1024, model_config)
        elif self.architecture in ['r18', 'r50', 'h50', 'k50', 's50']:
            batch_size, image_count, height, width, channel_count = x.shape
            log.verbose(f"Input shape to early-fusion cnn: {x.shape}")
            x = ResNet()(x)
    
            log.verbose('Embedding shape: %s' % repr(x.shape))
            current_total_dimensionality = functools.reduce(
                lambda x, y: x * y,
                x.shape[1:])
            x = jnp.reshape(
                x, [model_config.hparams.bs, current_total_dimensionality])

        net = x
        for _ in range(2):
            net = nn.Dense(num_outputs=2048)(net)
            net = nn.BatchNorm(use_running_average=self.use_running_average)(net)
            net = nn.leaky_relu(net)
            
        prediction = nn.Dense(
            inputs=net,
            num_outputs=self.element_count * self.element_length,
            activation_fn=None,
            normalizer_fn=None)
        prediction = jnp.reshape(prediction,
                                [batch_size, self.element_count, self.element_length])
        return prediction, x
            




class MidFusionCNN(nn.Module):
    element_count: int
    element_length: int
    embedding_length: int = 1023
    use_running_average: bool = True
    
    """A CNN architecture that fuses individual image channels in the middle."""
    @nn.compact
    def __call__(self, x) -> Any:
        batch_size = x.shape[0]
        individual_images = jnp.split(
            x, num_or_size_splits=x.shape[1], axis=1)
        image_count = len(individual_images)
        assert image_count == model_config.hparams.rc  # just debugging, can remove.
        embeddings = []
        embedding_length = 1023
        for i, image in enumerate(individual_images):
            with tf.variable_scope('mfcnn', reuse=i != 0):
            embedding = net_util.inputs_to_feature_vector(image, embedding_length,
                                                            model_config)
            if model_config.hparams.fua == 't':
                embedding = jnp.reshape(
                    embedding, [model_config.hparams.bs, 3, embedding_length // 3])
                embedding = jnp.pad(embedding, jnp.array([[0, 0], [0, 1], [0, 0]]))
                cam2world_i = observation.cam_to_worlds[:, i, :, :]  # [bs, rc, 4, 4]
                embedding = jnp.matmul(cam2world_i, embedding)
                # embedding shape [bs, 4, embedding_length // 3]
                embedding = jnp.reshape(embedding[:, :3, :],
                                        [model_config.hparams.bs, embedding_length])
            embeddings.append(embedding)
        if image_count > 1:
            embeddings = jnp.ensure_shape(
                tf.stack(embeddings, axis=1),
                [model_config.hparams.bs, image_count, embedding_length])
            embedding = jnp.reduce_max(embeddings, axis=1)
        else:
            embedding = embeddings[0]
        # TODO(kgenova) Should be a helper:
        net = embedding
        normalizer, normalizer_params = net_util.get_normalizer_and_mode(
            model_config)
        for _ in range(2):
            net = nn.Dense(2048)(net)
            if self.use_running_average:
                net = nn.BatchNorm(use_running_average=self.use_running_average)(net)
            net = nn.leaky_relu(net)
        prediction = nn.Dense(self.element_count * self.element_length)(net)
        prediction = jnp.reshape(prediction,
                                [batch_size, self.element_count, self.element_length])
        return prediction, embedding