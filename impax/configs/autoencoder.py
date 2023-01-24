"""
Consists of the definition of the hyperparameters of a model.
"""
import ml_collections
import optax


def get_config():
    config = ml_collections.ConfigDict()
    # [ob]: Whether to optimize the blobs.
    config.optimize_blobs = "t"
    # [cp]: The constant prediction mode.
    # 'a': abs
    # 's': sigmoid.
    config.prediction_mode = "a"
    # [dbg]: Whether to run in debug mode (checks for NaNs/Infs).
    config.debug_mode = False
    # [bs]: The batch size.
    config.batch_size = 24  # 8,
    # [h]: The height of the input observation.
    config.input_height = 137
    # [w]: The width of the input observation.
    config.input_width = 137
    # [lr]: The learning rate.
    config.learning_rate = 5e-5
    # Whether to 'nerfify' the occnet inputs:
    config.nerfify_occnet = False
    # Whether to 'nerfify' the pointnet inputs:
    config.nerify_pointnet = False
    # [loss]: A string encoding the loss to apply. Concatenate to apply several.
    #   'u': The uniform sample loss from the paper.
    #   'ns': The near surface sample loss from the paper.
    #   'ec': The element center loss from the paper.
    config.loss = "unsbbgi"
    # [lrf]: The loss receptive field.
    #   'g': Global: the loss is based on the global reconstruction.
    #   'l': Local: The loss is based on the local reconstruction.
    #   'x': Local Points: The loss is local points locally reconstructed.
    config.loss_receptive_field = "g"
    # [arch]: A string encoding the model architecture to use:
    #  'efcnn': A simple early-fusion feed-forward CNN.
    #  'ttcnn': A two-tower early-fusion feed-forward CNN. One tower for
    #           explicit parameters, and one tower for implicit parameters.
    config.model_architecture = "efcnn"
    # [cnna]: The network architecture of the CNN, if applicable.
    #  'cnn': A simple cnn with 5 conv layers and 5 FC layers.
    #  'r18': ResNet18.
    #  'r50': ResNet50.
    #  'h50': Tf-hub ImageNet pretrained ResNet50
    config.cnn_architecture = "r50"
    # [da]: The type of data augmentation to apply. Not compatible with
    # arch=efcnn, because depth maps can't be augmented.
    # 'f': No data augmentation.
    # 'p': Panning rotations.
    # 'r': Random SO(3) rotations.
    # 't': Random transformations with centering, rotations, and scaling.
    config.data_augmentation = "f"
    # [cri]: Whether to crop the region in the input
    config.crop_input = False
    # [crl]: Whether to crop the region in the loss
    config.crop_loss = False
    # [cic]: The input crop count.
    config.num_input_crops = 1024
    # [clc]: The supervision crop count.
    config.num_supervision_crops = 1024
    # [ia]: The architecture for implicit parameter prediction.
    #  '1': Predict both implicit and explicit parameters with the same network.
    #  '2': Predict implicit and explicit parameters with two separate networks.
    #  'p': Predict implicit parameters with local pointnets.
    config.implicit_architecture = "p"
    # [p64]: Whether to enable the pointnet 64-d transformation.
    config.to_64d = False
    # [fua]: Whether to enable 3D rotation of the model latents.
    config.apply_3d_rotation = True
    # [ipe]: Whether to enable implicit parameters. 't' or 'f'.
    config.enable_implicit_parameters = False
    # [ip]: The type of implicit parameters to use.
    #  'sb': Predict a minivector associated with each shape element that is
    #        residual to the prediction ('strictly better').
    config.implicit_parameters = "sb"
    # [ipc]: Whether the implicit parameters actually contribute (for debugging)
    # 't': Yes
    # 'f': No.
    config.debug_implicit_parameters = True
    # [ips]: The size of the implicit parameter vector.
    config.implicit_parameter_length = 32
    # [npe]: Whether there is a separate network per shape element.
    #  't': Each shape element has its own neural network.
    #  'f': The shape elements share a single network.
    config.seperate_network = False
    # [pe]: The point encoder to use.
    # 'pn': PointNet
    # 'dg': DG-CNN
    config.point_encoder = "pn"
    # [nf]: The DG-CNN feature count.
    config.feature_count = 64
    # [fbp]: Whether to fix the very slow pointnet reduce.
    config.fix_pointnet = True
    # [mfc]: The maxpool feature count for PointNet.
    config.maxpool_feature = 1024
    # [udp]: Whether to use the deprecated PointNet with Conv 2Ds.
    config.use_pointnet = True
    # [orc]: The number of ResNet layers in OccNet
    config.num_resnet_layers = 1
    # [sc]: The number of shape elements.
    config.num_shape_elements = 100
    # [lyr]: The number of blobs with left-right symmetry.
    config.num_blobs = 10
    # [opt]: The optimizer to use.
    #   'adm': optax.Adam.
    #   'sgd': SGD.
    config.optimizer = optax.adam
    # [tx]: How to transform to element coordinate frames when generating
    # OccNet sample points.
    #   'crs': Center, rotate, scale. Any subset of these is fine, but they
    #     will be applied as SRC so 'cs' probably doesn't make much sense.
    #   'i': Identity (i.e. global coordinates).
    config.transformations = "crs"
    # [lhdn]: Whether to learn the sigmoid normalization parameter.
    config.learn_sigmoid_normalization = False
    # [nrm]: The type of batch normalization to apply:
    #   'none': Do not apply any batch normalization.
    #   'bn': Enables batch norm, sets trainable to true, and sets is_training
    #     to true only for the train task.
    config.batch_normalization = None
    # [samp]: The input sampling scheme for the model:
    #   'imd': A dodecahedron of depth images.
    #   'imdpn': A dodecahedron of depth images and a point cloud with normals.
    #   'im1d': A single depth image from the dodecahedron.
    #   'rgb': A single rgb image from a random position.
    #   'imrd': One or more depth images from the dodecahedron.
    config.sampling_scheme = "imdpn"
    # [bg]: The background color for rgb images:
    # 'b': Black.
    # 'ws': White-smooth.
    config.background = "ws"
    # [lpc]: The number of local points per frame.
    config.num_local_points = 1024
    # [rc]: The random image count, if random image samples are used.
    config.num_random_images = 1
    # [spc]: The sample point count. If a sparse loss sampling strategy is
    #   selected, then this is the number of random points to sample.
    config.num_sample_points = 1024
    # [xsc]: The lrf='x' pre-sample point count. The number of samples taken
    # from the larger set before again subsampling to [spc] samples.
    config.num_subsampled_points = 10000
    # [sync]: 't' or 'f'. Whether to synchronize the GPUs during training, which
    #   increases the effective batch size and avoid stale gradients at the
    #   expense of decreased performance.
    config.synchronize_gpus = False
    # [gpu]: If 'sync' is true, this should be set to the number of GPUs used in
    #   training; otherwise it is ignored.
    config.num_gpus = 0
    # [vbs]: The virtual batch size; only used if 'sync' is true. The number of
    #   training examples to pool before applying a gradient.
    config.virtual_batch_size = 64
    # [r]: 'iso' for isotropic, 'aa' for anisotropic and axis-aligned to the
    #   normalized mesh coordinates, 'cov' for general Gaussian RBFs.
    config.coordinates = "cov"
    # [ir]: When refining a prediction with gradient descent, which points to
    #   use. Still needs to be refactored with the rest of the eval code.
    config.refine_prediction = "zero-set"
    # [res]: The rescaling between input and training SDFs.
    config.rescaling = 1.0
    # [pou]: Whether the representation is a Partition of Unity. Either 't' or
    #   'f'. If 't', the sum is normalized by the sum of the weights. If 'f',
    #   it is not.
    config.is_partition = False
    # [didx]: When only one of the dodecahedron images is shown to the network,
    #   the index of the image to show.
    config.index_of_dodecahedron = 1
    # [gh]: The height of the gaps (depth and luminance) images.
    config.gap_height = 224
    # [gw]: The width of the gaps (depth and luminance) images.
    config.gap_width = 224
    # [ucf]: The upweighting factor for interior points relative to exterior
    #   points.
    config.upweighting_factor = 1.0
    # [lset]: The isolevel of F defined to be the surface.
    config.isolevel = -0.07
    # [igt]: The inside-the-grid threshold in the element center lowres grid
    #   inside loss.
    # Based on a 32^3 voxel grid with an extent of [-0.7, 0.7]
    config.grid_threshold = 0.044
    # [wm]: Waymo scaling:
    config.waymo_scaling = False
    # [rsl]: The rescaling factor for the dataset.
    config.rescaling = 1.0
    # [ig]: The weight on the element center lowers grid inside loss.
    config.inside_loss_weight = 1.0
    # [gs]: The weight on the element center lowers grid squared loss.
    config.squared_loss_weight = 0.0
    # [gd]: The weight on the element center lowres grid direct loss.
    config.direct_loss_weight = 0.0
    # [cm]: The weight on the loss that says that element centers should have
    #   a small l2 norm.
    config.l2_norm_weight = 0.01
    # [cc]: The weight on the component of the center loss that says that
    #   element centers must be inside the predicted surface.
    config.element_center_weight = 0.0
    # [vt]: The threshold for variance in the element center variance loss.
    config.element_center_variance_threshold = 0.0
    # [nnt]: The threshold for nearest neighbors in the element center nn loss.
    config.nearest_neighbors_threshold = 0.0
    # [vw]: The weight on the center variance loss
    config.center_variance_loss_weight = 0.0
    # [nw]: The weight on the center nearest neighbor loss.
    config.nearest_neighbor_loss_weight = 0.0
    # [ow]: The weight on the overlap loss:
    config.overlap_loss_weight = 0.0
    # [dd]: Whether to turn on the deprecated single-shape occnet decoder.
    #   't' or 'f'.
    config.use_occnet_decoder = False
    # [fon]: Whether to fix the occnet pre-cbn residual bug. 't' or 'f'.
    config.fix_occnet = True
    # [itc]: The input pipeline thread count.
    config.num_threads = 12
    # [dmn]: The amount of noise in the depth map(s).
    config.depth_map_noise = 0.0
    # [pcn]: The amount of noise in the point cloud (along the normals).
    config.point_cloud_noise = 0.0
    # [xin]: The amount of noise in the xyz image (in any direction).
    config.xyz_noise = 0.0
    # [hdn]: The 'hardness' of the soft classification transfer function. The
    #   larger the value, the more like a true 0/1 class label the transferred
    #   F values will be, but the shorter the distance until the gradient is
    #   within roundoff error.
    config.hardness = 100.0
    # [ag]: The blob aggregation method. 's' for sum, 'm' for max.
    config.aggregation_method = "s"
    # [l2w]: The weight on the 'Uniform Sample Loss' from the paper.
    config.uniform_sample_loss_weight = 1.0
    # [a2w]: The weight on the 'Near Surface Sample Loss' from the paper.
    config.near_surface_sample_loss_weight = 0.1
    # [fcc]: The number of fully connected layers after the first embedding
    #   vector, including the final linear layer.
    config.num_layers_before_final_layer = 3
    # [fcs]: The width of the fully connected layers that are immediately before
    #   the linear layer.
    config.num_layers_before_linear = 2048
    # [ibblw]: The weight on the component of the center loss that says that
    #   element centers must be inside the ground truth shape bounding box.
    config.center_loss_weight = 10.0
    # [aucf]: Whether to apply the ucf hyperparameter to the near-surface sample
    #  loss. 't' or 'f'.
    config.upweight = False
    # [cd]: Whether to cache the input data once read. 't' or 'f'.
    config.cache = False
    # [elr]: The relative encoder learning rate (relative to the main LR)
    config.relative_encoder_learning_rate = 1.0
    # [cat]: The class to train on.
    config.class_category = "all"
    # [ident] : A unique identifying string for this hparam dictionary
    config.hyperparams = "sif"
    # [blc]: Whether to balance the categories. Requires a batch size multiplier of 13
    config.balanced = True

    # [rec]: The reconstruction equations
    config.reconstruction_equation = "r"

    # [iec]: The version of the implicit embedding CNN architecture.
    config.implicit_embedding_version = "v2"

    return config
