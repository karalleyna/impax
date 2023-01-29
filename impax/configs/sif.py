import ml_collections

from impax.configs import autoencoder


def get_backward_compatible_config():
    config = ml_collections.ConfigDict(
        {
            "relative_encoder_learning_rate": 1.0,
            "cache": False,
            "use_occnet_decoder": False,
            "fix_occnet": True,
            "num_threads": 10,
            "implicit_architecture": "2",
            "num_local_points": 1024,
            "to_64d": True,
            "implicit_embedding_version": "v1",
            "loss_receptive_field": "g",
            "num_blobs": 0,
            "background": "b",
            "balanced": False,
            "waymo_scaling": False,
            "rescaling": 1.0,
            "reconstruction_equation": "r",
            "debug_mode": False,
            "prediction_mode": "a",
            "transformations": "i",
            "depth_map_noise": 0.0,
            "point_cloud_noise": 0.0,
            "xyz_noise": 0.0,
            "aggregation_method": "s",
            "element_center_variance_threshold": 0.0,
            "nearest_neighbors_threshold": 0.0,
            "center_variance_loss_weight": 0.0,
            "nearest_neighbor_loss_weight": 0.0,
            "overlap_loss_weight": 0.0,
            "point_encoder": "pn",
            "feature_count": 64,
            "fix_pointnet": False,
            "maxpool_feature": 1024,
            "use_pointnet": True,
            "data_augmentation": "f",
            "crop_input": False,
            "num_input_crops": 1024,
            "crop_loss": False,
            "num_supervision_crops": 1024,
            "hyo": False,
            "hyp": False,
        }
    )
    return config


def get_improved_sif_config():
    config = autoencoder.get_config()
    config["model_architecture"] = "efcnn"
    config["sampling_scheme"] = "imdpn"
    config["cnn_architecture"] = "s50"
    config["hyo"] = False
    config["hyp"] = False
    config["num_shape_elements"] = 32
    config["num_blobs"] = 16
    config["loss"] = "unsbbgi"
    config["maxpool_feature"] = 512
    config["implicit_parameter_length"] = 0
    config["enable_implicit_parameters"] = False
    return config


def get_config():
    """The SIF representation trained according to the original SIF paper."""
    config = autoencoder.get_config()
    config.input_height = 137
    config.input_width = 137
    config.proto = "ShapeNetNSSDodecaSparseLRGMediumSlimPC"
    config.sampling_scheme = "im1dpn"
    config.implicit_architecture = "1"
    config.cnn_architecture = "cnn"
    config.num_shape_elements = 100
    config.num_blobs = 0
    config.class_category = "all"
    config.cache = False
    config.coordinates = "aa"
    config.num_sample_points = 3000
    config.balanced = False
    config.batch_size = 2
    config.implicit_parameter_length = 0
    config.enable_implicit_parameters = False
    config.upweighting_factor = 10.0
    config.upweight = True
    config.loss = "unsec"
    config.l2_norm_weight = 0.1
    config.grid_threshold = 0.0
    config.inside_loss_weight = 0.0
    config.squared_loss_weight = 0.0
    config.direct_loss_weight = 0.0
    config.element_center_weight = 0.01

    return config
