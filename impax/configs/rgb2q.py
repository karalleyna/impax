import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    # The task identifier, for looking up backwards compatible hparams (below):
    # It is called identity in the original repo.
    config.id = "rgb2q"
    # The batch size.
    # It is called bs in the original repo.
    config.batch_size = 32
    # The learning rate.
    # It is called lr in the original repo.
    config.learning_rate = 5e-05
    # The number of primitives (must match the dataset).
    # It is called sc in the original repo.
    config.num_primitives = 25
    # The length of each primitive.
    # It is called sd in the original repo.
    config.primitive_length = 42
    # The length of the explicit component.
    # It is called ed in the original repo.
    config.explicit_component_length = 10
    # The number of latent fully connected layers. >= 0.
    # It is called fcc in the original repo.
    config.num_latent_fully_connected_layers = 2
    # The width of the latent fully connected layers.
    # It is called fcl in the original repo.
    config.latent_fully_connected_layer_width = 2
    # ??
    config.ft = "t"
    # How to apply input augmentation: 'f', 't', 'io'
    config.input_augmentation = "f"
    # The encoder architecture. 'rn', 'sr50', 'vl', 'vlp':
    config.encoder = "rn"
    # The levelset of the reconstruction:
    config.level_set = -0.07
    # The distillation loss type: 'l1', 'l2', or'hu'
    config.distillation_loss = "l1"
    # [w]: The distillation loss weighting:
    # 'relative': Relative based on [ilw].
    # 'uniform': Uniform 1.0
    # 'inverse': inverse weighting (basically bad)
    config.distillation_loss_weighting = "relative"
    # Distillation supervision:
    # 'all': All
    # 'implicit': Just implicits
    # 'e': Just explicits
    return config
