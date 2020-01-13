function [opts] = vl_simplenn_para_test(varargin)
%VL_SIMPLENN  Evaluate a SimpleNN network.
%   RES = VL_SIMPLENN(NET, X) evaluates the convnet NET on data X.
%   RES = VL_SIMPLENN(NET, X, DZDY) evaluates the convnent NET and its
%   derivative on data X and output derivative DZDY (foward+bacwkard pass).
%   RES = VL_SIMPLENN(NET, X, [], RES) evaluates the NET on X reusing the
%   structure RES.
%   RES = VL_SIMPLENN(NET, X, DZDY, RES) evaluates the NET on X and its
%   derivatives reusing the structure RES.
%
%   This function process networks using the SimpleNN wrapper
%   format. Such networks are 'simple' in the sense that they consist
%   of a linear sequence of computational layers. You can use the
%   `dagnn.DagNN` wrapper for more complex topologies, or write your
%   own wrapper around MatConvNet computational blocks for even
%   greater flexibility.
%
%   The format of the network structure NET and of the result
%   structure RES are described in some detail below. Most networks
%   expect the input data X to be standardized, for example by
%   rescaling the input image(s) and subtracting a mean. Doing so is
%   left to the user, but information on how to do this is usually
%   contained in the `net.meta` field of the NET structure (see
%   below).
%
%   The NET structure needs to be updated as new features are
%   introduced in MatConvNet; use the `VL_SIMPLENN_TIDY()` function
%   to make an old network current, as well as to cleanup and check
%   the structure of an existing network.
%
%   Networks can run either on the CPU or GPU. Use VL_SIMPLENN_MOVE()
%   to move the network parameters between these devices.
%
%   To print or obtain summary of the network structure, use the
%   VL_SIMPLENN_DISPLAY() function.
%
%   VL_SIMPLENN(NET, X, DZDY, RES, 'OPT', VAL, ...) takes the following
%   options:
%
%   `Mode`:: `normal`
%      Specifies the mode of operation. It can be either `'normal'` or
%      `'test'`. In test mode, dropout and batch-normalization are
%      bypassed. Note that, when a network is deployed, it may be
%      preferable to *remove* such blocks altogether.
%
%   `ConserveMemory`:: `true`
%      Aggressively delete intermediate results. This in practice has
%      a very small performance hit and allows training much larger
%      models. However, it can be useful to disable it for
%      debugging. It is also possible to preserve individual layer outputs
%      by setting `net.layers{...}.precious` to `true`.
%
%   `CuDNN`:: `true`
%      Use CuDNN when available.
%
%   `Accumulate`:: `false`
%      Accumulate gradients in back-propagation instead of rewriting
%      them. This is useful to break the computation in sub-batches.
%      The gradients are accumulated to the provided RES structure
%      (i.e. to call VL_SIMPLENN(NET, X, DZDY, RES, ...).
%
%   `SkipForward`:: `false`
%      Reuse the output values from the provided RES structure and compute
%      only the derivatives (bacward pass).
%
%   ## The result format
%
%   SimpleNN returns the result of its calculations in the RES
%   structure array. RES(1) contains the input to the network, while
%   RES(2), RES(3), ... contain the output of each layer, from first
%   to last. Each entry has the following fields:
%
%   - `res(i+1).x`: the output of layer `i`. Hence `res(1).x` is the
%     network input.
%
%   - `res(i+1).aux`: any auxiliary output data of layer i. For example,
%     dropout uses this field to store the dropout mask.
%
%   - `res(i+1).dzdx`: the derivative of the network output relative
%     to the output of layer `i`. In particular `res(1).dzdx` is the
%     derivative of the network output with respect to the network
%     input.
%
%   - `res(i+1).dzdw`: a cell array containing the derivatives of the
%     network output relative to the parameters of layer `i`. It can
%     be a cell array for multiple parameters.
%
%   ## The network format
%
%   The network is represented by the NET structure, which contains
%   two fields:
%
%   - `net.layers` is a cell array with the CNN layers.
%
%   - `net.meta` is a grab-bag of auxiliary application-dependent
%     information, including for example details on how to normalize
%     input data, the class names for a classifiers, or details of
%     the learning algorithm. The content of this field is ignored by
%     VL_SIMPLENN().
%
%   SimpleNN is aware of the following layers:
%
%   Convolution layer::
%     The convolution layer wraps VL_NNCONV(). It has fields:
%
%     - `layer.type` contains the string `'conv'`.
%     - `layer.weights` is a cell array with filters and biases.
%     - `layer.stride` is the sampling stride (e.g. 1).
%     - `layer.pad` is the padding (e.g. 0).
%
%   Convolution transpose layer::
%     The convolution transpose layer wraps VL_NNCONVT(). It has fields:
%
%     - `layer.type` contains the string `'convt'`.
%     - `layer.weights` is a cell array with filters and biases.
%     - `layer.upsample` is the upsampling factor (e.g. 1).
%     - `layer.crop` is the amount of output cropping (e.g. 0).
%
%   Max pooling layer::
%     The max pooling layer wraps VL_NNPOOL(). It has fields:
%
%     - `layer.type` contains the string `'pool'`.
%     - `layer.method` is the pooling method (either 'max' or 'avg').
%     - `layer.pool` is the pooling size (e.g. 3).
%     - `layer.stride` is the sampling stride (usually 1).
%     - `layer.pad` is the padding (usually 0).
%
%   Normalization (LRN) layer::
%     The normalization layer wraps VL_NNNORMALIZE(). It has fields:
%
%     - `layer.type` contains the string `'normalize'` or `'lrn'`.
%     - `layer.param` contains the normalization parameters (see VL_NNNORMALIZE()).
%
%   Spatial normalization layer::
%     The spatial normalization layer wraps VL_NNSPNORM(). It has fields:
%
%     - `layer.type` contains the string `'spnorm'`.
%     - `layer.param` contains the normalization parameters (see VL_NNSPNORM()).
%
%   Batch normalization layer::
%     This layer wraps VL_NNBNORM(). It has fields:
%
%     - `layer.type` contains the string `'bnorm'`.
%     - `layer.weights` contains is a cell-array with, multiplier and
%       biases, and moments parameters
%
%     Note that moments are used only in `'test'` mode to bypass batch
%     normalization.
%
%   ReLU and Sigmoid layers::
%     The ReLU layer wraps VL_NNRELU(). It has fields:
%
%     - `layer.type` contains the string `'relu'`.
%     - `layer.leak` is the leak factor (e.g. 0).
%
%     The sigmoid layer is the same, but for the sigmoid function,
%     with `relu` replaced by `sigmoid` and no leak factor.
%
%   Dropout layer::
%     The dropout layer wraps VL_NNDROPOUT(). It has fields:
%
%     - `layer.type` contains the string `'dropout'`.
%     - `layer.rate` is the dropout rate (e.g. 0.5).
%
%     Note that the block is bypassed in `test` mode.
%
%   Softmax layer::
%     The softmax layer wraps VL_NNSOFTMAX(). It has fields
%
%     - `layer.type` contains the string`'softmax'`.
%
%   Log-loss layer and softmax-log-loss::
%     The log-loss layer wraps VL_NNLOSS(). It has fields:
%
%     - `layer.type` contains `'loss'`.
%     - `layer.class` contains the ground-truth class labels.
%
%     The softmax-log-loss layer wraps VL_NNSOFTMAXLOSS() instead. it
%     has the same parameters, but `type` contains the `'softmaxloss'`
%     string.
%
%   P-dist layer::
%     The p-dist layer wraps VL_NNPDIST(). It has fields:
%
%     - `layer.type` contains the string  `'pdist'`.
%     - `layer.p` is the P parameter of the P-distance (e.g. 2).
%     - `layer.noRoot` it tells whether to raise the distance to
%     the P-th power (e.g. `false`).
%     - `layer.epsilon` is the regularization parameter for the derivatives.
%
%   Custom layer::
%     This can be used to specify custom layers.
%
%     - `layer.type` contains the string `'custom'`.
%     - `layer.forward` is  a function handle computing the block.
%     - `layer.backward` is a function handle computing the block derivative.
%
%     The first function is called as
%
%          res(i+1) = layer.forward(layer, res(i), res(i+1))
%
%     where RES is the structure array specified before. The second function is
%     called as
%
%          res(i) = layer.backward(layer, res(i), res(i+1))
%
%     Note that the `layer` structure can contain additional custom
%     fields if needed.
%
%   See also: dagnn.DagNN, VL_SIMPLENN_TIDY(),
%   VL_SIMPLENN_DISPLAY(), VL_SIMPLENN_MOVE().

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.conserveMemory = false ;
opts.sync = false ;
opts.mode = 'normal' ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false;
opts = vl_argparse(opts, varargin);


