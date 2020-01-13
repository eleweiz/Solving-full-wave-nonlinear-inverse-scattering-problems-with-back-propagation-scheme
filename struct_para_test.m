function [opts]=struct_para_test(varargin)
%  replace structure paarameters

opts.expDir = fullfile('data','exp') ;
opts.continue = true ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;

opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = true ;
opts.errorFunction = 'euclideanloss' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.plotStatistics = true;
opts.lambda = 0;
opts.waveLevel = [1 2 3];
opts.waveName = 'vk';
opts.gradMax = 1e-2;
opts.weight = 'none';
opts = vl_argparse(opts, varargin) ;




