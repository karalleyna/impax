import ml_collections


def get_config():
    config = ml_collections.ConfigDict(  # [ob]: Whether to optimize the blobs.
        dict(
            ob="t",
            # [cp]: The constant prediction mode.
            # 'a': abs
            # 's': sigmoid.
            cp="a",
            # [dbg]: Whether to run in debug mode (checks for NaNs/Infs).
            dbg="f",
            # [bs]: The batch size.
            bs=24,  # 8,
            # [h]: The height of the input observation.
            h=137,
            # [w]: The width of the input observation.
            w=137,
            # [lr]: The learning rate.
            lr=5e-5,
            # Whether to 'nerfify' the occnet inputs:
            hyo="f",
            # Whether to 'nerfify' the pointnet inputs:
            hyp="f",
            # [loss]: A string encoding the loss to apply. Concatenate to apply several.
            #   'u': The uniform sample loss from the paper.
            #   'ns': The near surface sample loss from the paper.
            #   'ec': The element center loss from the paper.
            loss="unsbbgi",
            # [lrf]: The loss receptive field.
            #   'g': Global: the loss is based on the global reconstruction.
            #   'l': Local: The loss is based on the local reconstruction.
            #   'x': Local Points: The loss is local points locally reconstructed.
            lrf="g",
            # [arch]: A string encoding the model architecture to use:
            #  'efcnn': A simple early-fusion feed-forward CNN.
            #  'ttcnn': A two-tower early-fusion feed-forward CNN. One tower for
            #           explicit parameters, and one tower for implicit parameters.
            arch="efcnn",
            # [cnna]: The network architecture of the CNN, if applicable.
            #  'cnn': A simple cnn with 5 conv layers and 5 FC layers.
            #  'r18': ResNet18.
            #  'r50': ResNet50.
            #  'h50': Tf-hub ImageNet pretrained ResNet50
            cnna="r50",
            # [da]: The type of data augmentation to apply. Not compatible with
            # arch=efcnn, because depth maps can't be augmented.
            # 'f': No data augmentation.
            # 'p': Panning rotations.
            # 'r': Random SO(3) rotations.
            # 't': Random transformations with centering, rotations, and scaling.
            da="f",
            # [cri]: Whether to crop the region in the input
            cri="f",
            # [crl]: Whether to crop the region in the loss
            crl="f",
            # [cic]: The input crop count.
            cic=1024,
            # [clc]: The supervision crop count.
            clc=1024,
            # [ia]: The architecture for implicit parameter prediction.
            #  '1': Predict both implicit and explicit parameters with the same network.
            #  '2': Predict implicit and explicit parameters with two separate networks.
            #  'p': Predict implicit parameters with local pointnets.
            ia="p",
            # [p64]: Whether to enable the pointnet 64-d transformation.
            p64="f",
            # [fua]: Whether to enable 3D rotation of the model latents.
            fua="t",
            # [ipe]: Whether to enable implicit parameters. 't' or 'f'.
            ipe="f",
            # [ip]: The type of implicit parameters to use.
            #  'sb': Predict a minivector associated with each shape element that is
            #        residual to the prediction ('strictly better').
            ip="sb",
            # [ipc]: Whether the implicit parameters actually contribute (for debugging)
            # 't': Yes
            # 'f': No.
            ipc="t",
            # [ips]: The size of the implicit parameter vector.
            ips=32,
            # [npe]: Whether there is a separate network per shape element.
            #  't': Each shape element has its own neural network.
            #  'f': The shape elements share a single network.
            npe="f",
            # [pe]: The point encoder to use.
            # 'pn': PointNet
            # 'dg': DG-CNN
            pe="pn",
            # [nf]: The DG-CNN feature count.
            nf=64,
            # [fbp]: Whether to fix the very slow pointnet reduce.
            fbp="t",
            # [mfc]: The maxpool feature count for PointNet.
            mfc=1024,
            # [udp]: Whether to use the deprecated PointNet with Conv 2Ds.
            udp="t",
            # [orc]: The number of ResNet layers in OccNet
            orc=1,
            # [sc]: The number of shape elements.
            sc=100,
            # [lyr]: The number of blobs with left-right symmetry.
            lyr=10,
            # [opt]: The optimizer to use.
            #   'adm': Adam.
            #   'sgd': SGD.
            opt="adm",
            # [tx]: How to transform to element coordinate frames when generating
            # OccNet sample points.
            #   'crs': Center, rotate, scale. Any subset of these is fine, but they
            #     will be applied as SRC so 'cs' probably doesn't make much sense.
            #   'i': Identity (i.e. global coordinates).
            tx="crs",
            # [lhdn]: Whether to learn the sigmoid normalization parameter.
            lhdn="f",
            # [nrm]: The type of batch normalization to apply:
            #   'none': Do not apply any batch normalization.
            #   'bn': Enables batch norm, sets trainable to true, and sets is_training
            #     to true only for the train task.
            nrm="none",
            # [samp]: The input sampling scheme for the model:
            #   'imd': A dodecahedron of depth images.
            #   'imdpn': A dodecahedron of depth images and a point cloud with normals.
            #   'im1d': A single depth image from the dodecahedron.
            #   'rgb': A single rgb image from a random position.
            #   'imrd': One or more depth images from the dodecahedron.
            samp="imdpn",
            # [bg]: The background color for rgb images:
            # 'b': Black.
            # 'ws': White-smooth.
            bg="ws",
            # [lpc]: The number of local points per frame.
            lpc=1024,
            # [rc]: The random image count, if random image samples are used.
            rc=1,
            # [spc]: The sample point count. If a sparse loss sampling strategy is
            #   selected, then this is the number of random points to sample.
            spc=1024,
            # [xsc]: The lrf='x' pre-sample point count. The number of samples taken
            # from the larger set before again subsampling to [spc] samples.
            xsc=10000,
            # [sync]: 't' or 'f'. Whether to synchronize the GPUs during training, which
            #   increases the effective batch size and avoid stale gradients at the
            #   expense of decreased performance.
            sync="f",
            # [gpu]: If 'sync' is true, this should be set to the number of GPUs used in
            #   training; otherwise it is ignored.
            gpuc=0,
            # [vbs]: The virtual batch size; only used if 'sync' is true. The number of
            #   training examples to pool before applying a gradient.
            vbs=64,
            # [r]: 'iso' for isotropic, 'aa' for anisotropic and axis-aligned to the
            #   normalized mesh coordinates, 'cov' for general Gaussian RBFs.
            r="cov",
            # [ir]: When refining a prediction with gradient descent, which points to
            #   use. Still needs to be refactored with the rest of the eval code.
            ir="zero-set",
            # [res]: The rescaling between input and training SDFs.
            res=1.0,
            # [pou]: Whether the representation is a Partition of Unity. Either 't' or
            #   'f'. If 't', the sum is normalized by the sum of the weights. If 'f',
            #   it is not.
            pou="f",
            # [didx]: When only one of the dodecahedron images is shown to the network,
            #   the index of the image to show.
            didx=1,
            # [gh]: The height of the gaps (depth and luminance) images.
            gh=224,
            # [gw]: The width of the gaps (depth and luminance) images.
            gw=224,
            # [ucf]: The upweighting factor for interior points relative to exterior
            #   points.
            ucf=1.0,
            # [lset]: The isolevel of F defined to be the surface.
            lset=-0.07,
            # [igt]: The inside-the-grid threshold in the element center lowres grid
            #   inside loss.
            igt=0.044,  # Based on a 32^3 voxel grid with an extent of [-0.7, 0.7]
            # [wm]: Waymo scaling:
            wm="f",
            # [rsl]: The rescaling factor for the dataset.
            rsl=1.0,
            # [ig]: The weight on the element center lowres grid inside loss.
            ig=1.0,
            # [gs]: The weight on the element center lowres grid squared loss.
            gs=0.0,
            # [gd]: The weight on the element center lowres grid direct loss.
            gd=0.0,
            # [cm]: The weight on the loss that says that element centers should have
            #   a small l2 norm.
            cm=0.01,
            # [cc]: The weight on the component of the center loss that says that
            #   element centers must be inside the predicted surface.
            cc=0.0,
            # [vt]: The threshold for variance in the element center variance loss.
            vt=0.0,
            # [nnt]: The threshold for nearest neighbors in the element center nn loss.
            nnt=0.0,
            # [vw]: The weight on the center variance loss
            vw=0.0,
            # [nw]: The weight on the center nearest neighbor loss.
            nw=0.0,
            # [ow]: The weight on the overlap loss:
            ow=0.0,
            # [dd]: Whether to turn on the deprecated single-shape occnet decoder.
            #   't' or 'f'.
            dd="f",
            # [fon]: Whether to fix the occnet pre-cbn residual bug. 't' or 'f'.
            fon="t",
            # [itc]: The input pipeline thread count.
            itc=12,
            # [dmn]: The amount of noise in the depth map(s).
            dmn=0.0,
            # [pcn]: The amount of noise in the point cloud (along the normals).
            pcn=0.0,
            # [xin]: The amount of noise in the xyz image (in any direction).
            xin=0.0,
            # [hdn]: The 'hardness' of the soft classification transfer function. The
            #   larger the value, the more like a true 0/1 class label the transferred
            #   F values will be, but the shorter the distance until the gradient is
            #   within roundoff error.
            hdn=100.0,
            # [ag]: The blob aggregation method. 's' for sum, 'm' for max.
            ag="s",
            # [l2w]: The weight on the 'Uniform Sample Loss' from the paper.
            l2w=1.0,
            # [a2w]: The weight on the 'Near Surface Sample Loss' from the paper.
            a2w=0.1,
            # [fcc]: The number of fully connected layers after the first embedding
            #   vector, including the final linear layer.
            fcc=3,
            # [fcs]: The width of the fully connected layers that are immediately before
            #   the linear layer.
            fcs=2048,
            # [ibblw]: The weight on the component of the center loss that says that
            #   element centers must be inside the ground truth shape bounding box.
            ibblw=10.0,
            # [aucf]: Whether to apply the ucf hyperparameter to the near-surface sample
            #  loss. 't' or 'f'.
            aucf="f",
            # [cd]: Whether to cache the input data once read. 't' or 'f'.
            cd="f",
            # [elr]: The relative encoder learning rate (relative to the main LR)
            elr=1.0,
            # [cat]: The class to train on.
            cat="all",
            # A unique identifying string for this hparam dictionary
            ident="sif",
            # Whether to balance the categories. Requires a batch size multiplier of 13
            blc="t",
            # [rec]: The reconstruction equations
            rec="r",
            # The version of the implicit embedding CNN architecture.
            iec="v2",
        )
    )
    return config
