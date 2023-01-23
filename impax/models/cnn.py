"""A simple feed forward CNN."""
from typing import Any

from flax import linen as nn
import jax.numpy as jnp

# local
from impax.utils.logging_util import log
from impax.models.resnet_v2 import ResNet
from impax.utils.model_utils import Encoder


class EarlyFusionCNN(nn.Module):
    """A CNN that maps 1+ images with 1+ chanels to a feature vector."""

    element_count: int
    element_length: int
    architecture: str = "r18"
    batch_size: int = 32
    use_running_average: bool = True

    @nn.compact
    def __call__(self, x) -> Any:
        batch_size, num_images, height, width, num_channels = x.shape
        log.warning(f"Input shape to early-fusion cnn: {x.shape}")

        if num_images == 1:
            x = jnp.reshape(x, [batch_size, height, width, num_channels])
        else:
            x = jnp.reshape(
                jnp.transpose(x, perm=[0, 2, 3, 1, 4]),
                [batch_size, height, width, num_images * num_channels],
            )

        if self.architecture == "cnn":
            embedding, _ = Encoder(
                convolution_layers=[16, 32, 64, 128, 128],
                dense_layers=[
                    1024,
                ],
                use_running_average=self.use_running_average,
            )(x)
        elif self.architecture in ["r18", "r50", "h50", "k50", "s50"]:
            embedding = ResNet(
                [64, 128, 256, 512],
            )(x)
            embedding = jnp.reshape(embedding, [batch_size, -1])

        prediction = embedding
        for _ in range(2):
            prediction = nn.Dense(2048)(prediction)
            prediction = nn.BatchNorm(use_running_average=self.use_running_average)(
                prediction
            )
            prediction = nn.leaky_relu(prediction)

        prediction = nn.Dense(self.element_count * self.element_length)(prediction)
        prediction = jnp.reshape(
            prediction, [batch_size, self.element_count, self.element_length]
        )
        return prediction, embedding


class MidFusionCNN(nn.Module):
    observation: Any
    num_elements: int
    element_length: int
    batch_size: int = 32
    rotate: bool = True
    use_running_average: bool = True

    """A CNN architecture that fuses individual image channels in the middle."""

    @nn.compact
    def __call__(self, x) -> Any:

        batch_size = x.shape[0]
        individual_images = jnp.split(x, indices_or_sections=x.shape[1], axis=1)
        num_images = len(individual_images)
        embeddings = []
        embedding_length = 1023
        for i, image in enumerate(individual_images):
            embedding, _ = Encoder(
                convolution_layers=[16, 32, 64, 128, 128],
                dense_layers=[
                    embedding_length,
                ],
                use_running_average=self.use_running_average,
            )(image)

            if self.rotate:
                embedding = jnp.reshape(
                    embedding, [self.batch_size, 3, embedding_length // 3]
                )
                embedding = jnp.pad(embedding, jnp.array([[0, 0], [0, 1], [0, 0]]))
                cam2world_i = self.observation.cam_to_worlds[
                    :, i, :, :
                ]  # [bs, rc, 4, 4]
                embedding = jnp.matmul(cam2world_i, embedding)
                # embedding shape [bs, 4, embedding_length // 3]
                embedding = jnp.reshape(
                    embedding[:, :3, :], [self.batch_size, embedding_length]
                )
            embeddings.append(embedding)
        if num_images > 1:

            embedding = jnp.max(jnp.stack(embeddings, axis=0), axis=1)
        else:
            embedding = embeddings[0]

        prediction = embedding

        for _ in range(2):
            prediction = nn.Dense(2048)(prediction)
            prediction = nn.BatchNorm(use_running_average=self.use_running_average)(
                prediction
            )
            prediction = nn.leaky_relu(prediction)
        prediction = nn.Dense(self.num_elements * self.element_length)(prediction)
        prediction = jnp.reshape(
            prediction, [batch_size, self.num_elements, self.element_length]
        )
        return prediction, embedding
