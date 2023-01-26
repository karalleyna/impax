from impax.configs import autoencoder


def get_config():
    """Sets hyperparameters according to the published LDIF results."""
    config = autoencoder.get_config()
    config.model_architecture = "efcnn"
    config.sampling_scheme = "imdpn"
    config.cnn_architecture = "s50"
    config.hyo = False
    config.hyp = False
    config.num_shape_elements = 32
    config.num_blobs = 16
    config.loss = "unsbbgi"
    config.maxpool_feature = 512
    config.implicit_parameter_length = 32
    config.enable_implicit_parameters = True
    return config
