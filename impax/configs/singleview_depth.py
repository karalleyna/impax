from impax.configs import autoencoder


def get_config():
    """A singleview-depth architecture."""
    config = autoencoder.get_config()
    config["input_height"] = 224
    config["input_width"] = 224
    config["cnn_architecture"] = "s50"
    config["sampling_scheme"] = "im1xyzpn"
    config["index_of_dodecahedron"] = 0
    config["implicit_architecture"] = "p"
    config["num_shape_elements"] = 32
    config["num_blobs"] = 16
    return config
