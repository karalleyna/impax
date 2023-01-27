from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp

from impax.networks.resnet_v2 import BottleneckResNetBlock, ResNet50


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or [batch,
        height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
        Should be a positive integer.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == "channels_first":
        padded_inputs = jnp.pad(
            inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]
        )
    else:
        padded_inputs = jnp.pad(
            inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
        )
    return padded_inputs


def resize(inputs, height, width, data_format):
    if data_format == "channels_first":
        inputs = jnp.transpose(inputs, axes=[0, 2, 3, 1])
    else:
        assert data_format == "channels_last"
    output_res = (len(inputs), height, width, inputs.shape[-1])
    output = jax.image.resize(inputs, output_res, method="nearest", antialias=True)
    if data_format == "channels_first":
        output = jnp.transpose(output, axes=[0, 3, 1, 2])
    return output


class ResnetUnet(nn.Module):
    """The main interface to the resnet_unet architecture."""

    flat_predict: bool = True  # model_config.hparams.fp
    sc: int = 1  # model_config.hparams.sc
    param_count: int = 1024

    @nn.compact
    def __call__(self, x, train: bool = True):
        assert len(x.shape) == 4
        batch_size, input_height, input_width, feature_count = x.shape

        if input_height != 224 or input_width != 224:
            inputs = jax.image.resize(
                x,
                (batch_size, 224, 224, feature_count),
                method="nearest",
                antialias=True,
            )

        data_format = "channels_last"
        model = ResNet50(return_intermediates=True)

        if data_format == "channels_first":
            fullres_in = jnp.transpose(inputs, axes=[0, 3, 1, 2])
        else:
            fullres_in = inputs

        feature_axis = 1 if data_format == "channels_first" else 3
        post_fc, intermediate_outputs = model(inputs)
        assert post_fc.shape == (batch_size, 1024)

        if self.flat_predict:
            post_fc = nn.relu(post_fc)
            prediction = nn.Dense(self.param_count * self.sc)(post_fc)
            prediction = jnp.reshape(
                prediction, [batch_size, self.sc, self.param_count]
            )
            return prediction

        _ = resize(fullres_in, 4, 4, data_format)  # pylint: disable=unused-variable
        input_at_7x7 = resize(fullres_in, 7, 7, data_format)
        input_at_14x14 = resize(fullres_in, 14, 14, data_format)
        input_at_28x28 = resize(fullres_in, 28, 28, data_format)
        input_at_56x56 = resize(fullres_in, 56, 56, data_format)

        skip_from_7x7 = intermediate_outputs[3]
        skip_from_14x14 = intermediate_outputs[2]
        skip_from_28x28 = intermediate_outputs[1]
        skip_from_56x56 = intermediate_outputs[0]

        if data_format == "channels_first":
            assert skip_from_7x7.shape == (batch_size, 2048, 7, 7)
            assert skip_from_14x14.shape == (batch_size, 1024, 14, 14)
            assert skip_from_28x28.shape == (batch_size, 512, 28, 28)
            assert skip_from_56x56.shape == (batch_size, 256, 56, 56)
        else:
            assert skip_from_7x7.shape == (batch_size, 7, 7, 2048)
            assert skip_from_14x14.shape == (batch_size, 14, 14, 1024)
            assert skip_from_28x28.shape == (batch_size, 28, 28, 512)
            assert skip_from_56x56.shape == (batch_size, 56, 56, 256)

        # The 1024 feature vector
        features_4x4 = jnp.reshape(post_fc, [batch_size, 4, 4, 64])
        if data_format == "channels_first":
            features_4x4 = jnp.transpose(features_4x4, axes=[0, 3, 1, 2])

        conv = partial(nn.Conv, use_bias=False)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
        )

        for i in range(3):
            features_4x4 = BottleneckResNetBlock(
                1024, conv, norm, name=f"up_block_layer_4x4_{i}"
            )(features_4x4)

        output_4x4 = nn.Conv(self.param_count, kernel_size=(1, 1), strides=1)(
            fixed_padding(features_4x4, kernel_size=1, data_format=data_format)
        )

        if data_format == "channels_first":
            assert output_4x4.shape == (batch_size, self.param_count, 4, 4)
        else:
            assert output_4x4.shape == (batch_size, 4, 4, self.param_count)

        # Move up to 7x7:
        output_4x4_at_7x7 = resize(output_4x4, 7, 7, data_format)
        features_4x4_at_7x7 = resize(features_4x4, 7, 7, data_format)
        features_7x7 = jnp.concatenate(
            [output_4x4_at_7x7, input_at_7x7, features_4x4_at_7x7, skip_from_7x7],
            axis=feature_axis,
        )

        for i in range(3):
            features_7x7 = BottleneckResNetBlock(
                512, conv, norm, name=f"up_block_layer_7x7_{i}"
            )(features_7x7)

        output_7x7 = nn.Conv(self.param_count, kernel_size=(1, 1), strides=1)(
            fixed_padding(features_7x7, kernel_size=1, data_format=data_format)
        )

        # Move up to 14x14:
        output_7x7_at_14x14 = resize(output_7x7, 14, 14, data_format)
        features_7x7_at_14x14 = resize(features_7x7, 14, 14, data_format)
        features_14x14 = jnp.concatenate(
            [
                output_7x7_at_14x14,
                input_at_14x14,
                features_7x7_at_14x14,
                skip_from_14x14,
            ],
            axis=feature_axis,
        )

        for i in range(3):
            features_14x14 = BottleneckResNetBlock(
                256, conv, norm, name=f"up_block_layer_14x14_{i}"
            )(features_14x14)

        output_14x14 = nn.Conv(self.param_count, kernel_size=(1, 1), strides=1)(
            fixed_padding(features_14x14, kernel_size=1, data_format=data_format)
        )

        # Move up to 28x28
        output_14x14_at_28x28 = resize(output_14x14, 28, 28, data_format)
        features_14x14_at_28x28 = resize(features_14x14, 28, 28, data_format)
        features_28x28 = jnp.concatenate(
            [
                output_14x14_at_28x28,
                input_at_28x28,
                features_14x14_at_28x28,
                skip_from_28x28,
            ],
            axis=feature_axis,
        )

        for i in range(3):
            features_28x28 = BottleneckResNetBlock(
                256, conv, norm, name=f"up_block_layer_28x28_{i}"
            )(features_28x28)

        output_28x28 = nn.Conv(self.param_count, kernel_size=(1, 1), strides=1)(
            fixed_padding(features_28x28, kernel_size=1, data_format=data_format)
        )

        # Move up to 56x56 (finally):
        output_28x28_at_56x56 = resize(output_28x28, 56, 56, data_format)
        features_28x28_at_56x56 = resize(features_28x28, 56, 56, data_format)
        features_56x56 = jnp.concatenate(
            [
                output_28x28_at_56x56,
                input_at_56x56,
                features_28x28_at_56x56,
                skip_from_56x56,
            ],
            axis=feature_axis,
        )

        for i in range(3):
            features_56x56 = BottleneckResNetBlock(
                128, conv, norm, name=f"up_block_layer_56x56_{i}"
            )(features_56x56)

        output_56x56 = nn.Conv(self.param_count, kernel_size=(1, 1), strides=1)(
            fixed_padding(features_56x56, kernel_size=1, data_format=data_format)
        )

        if data_format == "channels_first":
            flat_out_4x4 = jnp.reshape(
                jnp.transpose(output_4x4, perm=[0, 2, 3, 1]),
                [batch_size, 4 * 4, self.param_count],
            )
            flat_out_7x7 = jnp.reshape(
                jnp.transpose(output_7x7, perm=[0, 2, 3, 1]),
                [batch_size, 7 * 7, self.param_count],
            )
            flat_out_14x14 = jnp.reshape(
                jnp.transpose(output_14x14, perm=[0, 2, 3, 1]),
                [batch_size, 14 * 14, self.param_count],
            )
            flat_out_28x28 = jnp.reshape(
                jnp.transpose(output_28x28, perm=[0, 2, 3, 1]),
                [batch_size, 28 * 28, self.param_count],
            )
            flat_out_56x56 = jnp.reshape(
                jnp.transpose(output_56x56, perm=[0, 2, 3, 1]),
                [batch_size, 56 * 56, self.param_count],
            )
        else:
            flat_out_4x4 = jnp.reshape(
                output_4x4, [batch_size, 4 * 4, self.param_count]
            )
            flat_out_7x7 = jnp.reshape(
                output_7x7, [batch_size, 7 * 7, self.param_count]
            )
            flat_out_14x14 = jnp.reshape(
                output_14x14, [batch_size, 14 * 14, self.param_count]
            )
            flat_out_28x28 = jnp.reshape(
                output_28x28, [batch_size, 28 * 28, self.param_count]
            )
            flat_out_56x56 = jnp.reshape(
                output_56x56, [batch_size, 56 * 56, self.param_count]
            )

        return [
            flat_out_4x4,
            flat_out_7x7,
            flat_out_14x14,
            flat_out_28x28,
            flat_out_56x56,
        ]


if __name__ == "__main__":
    renset = ResnetUnet(flat_predict=False)

    vars = renset.init(jax.random.PRNGKey(0), jnp.ones((32, 64, 64, 3)))

    output, vars = renset.apply(
        vars, jnp.ones((32, 64, 64, 3)), mutable=["batch_stats"]
    )
    for elem in output:
        print(elem.shape)
