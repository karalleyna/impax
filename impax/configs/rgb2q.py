"""
Consists of the definition of the hyperparameters of RGB2Q.
"""
import ml_collections


def get_backward_compatible_config():
    config = ml_collections.ConfigDict(
        {
            "l": "l2",
            "input_width": "u",
            "distillation_supervision": "a",
            "background": "b",
            "distillation_implicit_weight": 1.0,
            "distillation_loss_weight": 1.0,
            "whitening": False,
            "reconstruction_loss_weight": 1.0,
            "segmentation_loss_weight": 0.0,
            "depth_loss_weight": 0.0,
            "xyz_loss_weight": 0.0,
            "extra_tasks": "",
            "resolution": 137,
            "decoder_architecture": "dl",
            "tiling": False,
            "image_type": "rgb",
            "fix_object_detection": False,
            "extra_prediction": False,
            "secondary_encoder_enabled": True,
            "to_xyz": False,
            "use_prediction": True,
            "depth_prediction_loss": False,
            "to_decoder": False,
            "segmentation": "n",
            "depth_segmentation": "d",
            "secondary_encoder_architecture": "epr50",
            "depth_regression_loss_type": "l2",
            "local_pointnet_encoder": False,
            "explicit_component_length": 10,
            "threshold": 4.0,
            "point_validity": False,
            "finetune_decoder": False,
            "num_blobs": 0,
            "add_normals": False,
            "grf": False,
            "add_secondary_input": False,
            "is_normal_input": False,
        }
    )
    return config


def get_config():
    config = ml_collections.ConfigDict()
    # [identity]: The task identifier, for looking up backwards compatible hparams (below):
    # It is called identity in the original repo.
    config.id = "rgb2q"
    # [bs]: The batch size.
    # It is called bs in the original repo.
    config.batch_size = 32
    # [lr]: The learning rate.
    # It is called lr in the original repo.
    config.learning_rate = 5e-05
    # [sc]: The number of primitives (must match the dataset).
    # It is called sc in the original repo.
    config.num_primitives = 25
    # [sd]: The length of each primitive.
    # It is called sd in the original repo.
    config.primitive_length = 42
    # [ed]: The length of the explicit component.
    # It is called ed in the original repo.
    config.explicit_component_length = 10
    # [fcc]: The number of latent fully connected layers. >= 0.
    # It is called fcc in the original repo.
    config.num_latent_fully_connected_layers = 2
    # [fcl]: The width of the latent fully connected layers.
    # It is called fcl in the original repo.
    config.latent_fully_connected_layer_width = 2
    # [ft]: ??
    config.ft = True
    # [aug]: How to apply input augmentation: 'f', 't', 'io'
    config.input_augmentation = False
    # [enc]: The encoder architecture. 'rn', 'sr50', 'vl', 'vlp':
    config.encoder = "rn"
    # [lset]: The levelset of the reconstruction:
    config.level_set = -0.07
    # [l]: The distillation loss type: 'l1', 'l2', or'hu'
    config.distillation_loss = "l1"
    # [w]: The distillation loss weighting:
    # 'relative': Relative based on [ilw].
    # 'uniform': Uniform 1.0
    # 'inverse': inverse weighting (basically bad)
    config.distillation_loss_weighting = "relative"
    # [sv]: Distillation supervision:
    # 'all': All
    # 'implicit': Just implicits
    # 'explicit': Just explicits
    config.distillation_supervision = "all"

    # [cnna]: The network architecture of the CNN, if applicable.
    #  'cnn': A simple cnn with 5 conv layers and 5 FC layers.
    #  'r18': ResNet18.
    #  'r50': ResNet50.
    #  'h50': Tf-hub ImageNet pretrained ResNet50
    config.cnn_architecture = "r50"
    # [rc]: The random image count, if random image samples are used.
    config.num_random_images = 1

    # [bg]: The background
    # 'b' or 'w' (black/white)
    # with or without 's' for smooth:
    config.background = "ws"

    # [dlw]: The distillation loss weight:
    config.distillation_loss_weight = 1.0
    # [ilw]: The relative weight of implicits within the distillation loss:
    config.distillation_implicit_weight = 1.0
    # [elw]: The relative weight of explicits within the distillation loss:
    config.distillation_explicit_weight = 1.0
    # [slw]: The argmax segmentation task loss weight:
    config.segmentation_loss_weight = 1.0
    # [tlw]: The depth task loss weight:
    config.depth_loss_weight = 1.0
    # [xlw]: The xyz task loss weight:
    config.xyz_loss_weight = 1.0
    # [xt]: The set of extra tasks:
    # 'd': Predict depth
    # 'x': Predict XYZ
    # 'a': Predict the argmax image.
    # 's': Predict a segmentation image.
    # 'n': Predict a normals image.
    config.extra_tasks = "a"
    # [wh]: Whether to apply whitening:
    config.whitening = True
    # [rlw]: The reconstruction loss weight:
    config.reconstruction_loss_weight = 1.0
    # [res]: The operating resolution (excluding pretrained networks).
    config.resolution = 137
    # [dec]: The decoder architecture:
    config.decoder_architecture = "dl"
    # [et]: Whether to enable tiling to three channels for pretrained input support:
    config.tiling = False
    # [im]: The type of image to use for the input:
    config.image_type = "rgb"
    # [eil: Whether to predict the blobs from the extra prediction.
    config.extra_prediction = False
    # [spt]: Whether the secondary encoder is pretrained (if there is one)
    config.secondary_encoder_enabled = True
    # [sec]: The secondary encoder architecture.
    config.secondary_encoder_architecture = "epr50"
    # [txd]: Whether the depth prediction should be transformed to XYZ before being
    # given to the secondary encoder:
    config.to_xyz = True
    # [up]: Whether to use the predicted image 't' or the GT image 'f' in the inline-
    # encoder:
    config.use_prediction = True
    # [rdx]: Whether to apply the depth prediction loss in xyz space:
    config.depth_prediction_loss = False
    # [st]: Whether each task gets its own decoder:
    config.to_decoder = False
    # [lsg]: What kind of segmentation to do in the loss function:
    # 'n': None- the loss is applied to the entire image.
    # 'p': Predicted: The loss is applied where the prediction segments.
    # 'g': Ground truth: The loss is applied based on the GT segmentation.
    # 'o': The loss is based on GT == 1.0
    config.segmentation = "n"
    # [lsn]: What kind of loss to apply for the normals.
    config.normal_loss_type = "a"
    # [pnf] The predict normal's frame.
    # False for world('w'  in the original repo)
    # True for cam('c'  in the original repo)
    config.cam = True
    # [nlw]: The normal prediction loss weight.
    config.normal_loss_weight = 1.0
    # [ni] Whether to add the normals as a secondary input.
    config.add_secondary_input = False
    # [nmo]: Whether to *only* input the normals:
    config.is_normal_input = False
    # [dxw]: locals xyz loss weight
    config.xyz_loss_weight = 1.0
    # [dnw]: locals normal loss weight
    config.normal_loss_weight = 1.0
    # [drl]: The type of depth regression loss:
    config.depth_regression_loss_type = "l2"
    # [msg]: What kind of segmentation to do in the depth -> XYZ mapping:
    # 'd': Do the mapping based on where the predicted depth is nontrivial.
    # 'p': Do the mapping based on the predicted segmentation mask.
    # 'g': Do the mapping based on the ground truth segmentation mask.
    config.depth_segmentation = "p"
    # [lpe]: Whether to have a local pointnet encoder. Only works if an xyz
    # image is available at test time:
    config.local_pointnet_encoder = False
    # [lpn]: Whether to add normals as a feature for the local pointnets.
    config.add_normals = False
    # [lpt]: The threshold (in radii) of the local pointclouds.
    config.threshold = 4.0
    # [vf]: Whether to add a point validity one-hot feature to the pointnet.
    config.point_validity = False
    # [fod]: Whether to fix the object_detection encoder defaults.
    config.fix_object_detection = True
    # [ftd]: Whether to finetune the decoder
    config.finetune_decoder = False
    # # [iz]: Whether to ignore 0s in the local pointnet encoders.
    # iz='f',
    config.ignore_zeros = False
    # Whether to include global scope in the final feature vector.
    config.include_global_scope = False
    # The task identifier, for looking up backwards compatible hparams (below):
    config.task = "rgb2q"
    return config
