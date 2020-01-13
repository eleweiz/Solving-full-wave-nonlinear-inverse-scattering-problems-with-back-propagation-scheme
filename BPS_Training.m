%% CNN for inverse problem of Cricle reconstruction;
% The input is the results reconstructed by Bp with 5% noise, and output is groundtruth;  
% Wirtten by Wei Zhun at ECE NUS on 20th, Nov, 2017; 

clc;clear all;close all;
% data_generate_Circle_Es;  % generate scattering field
% Data_generate_Circle_BP;  % Backpropogation field
run ./matconvnet-1.0-beta23/matlab/vl_setupnn
run ./matconvnet-1.0-beta23/matlab/vl_compilenn
load CNN_Data_Cir.mat;

%%
% -----------------------------------------------------------------------------------------------------------------------------
%                                       Basic parameters
% -----------------------------------------------------------------------------------------------------------------------------
% input
W   = 64; % size of patch
Nimg= 500; % # of train + test set
Nimg_test= fix(Nimg*0.05);
Coef=10;

id_tmp  = ones(Nimg,1);
id_tmp(Nimg-Nimg_test+1:end)=2;  % the test indx is 2, training is 1

imdb.images.set=id_tmp;             % train set : 1 , test set : 2
imdb.images.noisy=single((epsil_bp-1)*Coef);    % input  : H x W x C x N (X,Y,channel,batch)
imdb.images.orig=single((epsil_exa-1)*Coef);     % output : H x W x C x N (X,Y,channel,batch)

% opts
opts.channel_in = 1;
opts.channel_out=1;
opts.useGpu = 'false'; %'false'
opts.gpus = [] ;       % []
opts.patchSize = W;
opts.batchSize = 1;    
opts.gradMax = 1e-2;
opts.numEpochs = 203 ;
opts.momentum = 0.99 ;
opts.imdb=imdb;
opts.expDir='./training_result_1';
opts.weightInitMethod = 'gaussian' ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;
opts.contrastNormalization = false ;
opts.waveLevel = 6;
opts.waveName = 'vk';
opts.train = struct() ;
opts.weight='none';
opts.plotDiagnostics = false ;
opts.plotStatistics = true;
opts.scale = 1 ;
opts.weightDecay = 1e-6;
opts.cudnnWorkspaceLimit = 1024*1024*1204*1 ; % 1GB
opts.lambda = 1e-4;
opts.continue = true ;
opts.numSubBatches = 1 ;
opts.train = find(imdb.images.set==1) ;
opts.val = find(imdb.images.set == 2) ;
opts.prefetch = false ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;
opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = true ;
opts.errorFunction = 'euclideanloss' ;
opts.errorLabels = {} ;

%%
% --------------------------------------------------------------------------------------------------------------------------
%                                                             Build net
% --------------------------------------------------------------------------------------------------------------------------
% net para
ch_length=opts.channel_in;
net=[];
net.meta.normalization.imageSize = [opts.patchSize,opts.patchSize,ch_length] ;
net.layers = {} ;
ch_length = opts.channel_in;
ch_length_out = opts.channel_out;
KerenlSize = 3;   % conv size
zeroPad = floor(KerenlSize/2);  % padding size equal =(k-1)/2 considering k is odd value;

net = add_block_test(net, opts, '0', KerenlSize, KerenlSize, ch_length, 64, 1, zeroPad,1,1) ;
net = add_block_test(net, opts, '0', KerenlSize, KerenlSize, 64, 64, 1, zeroPad,1,1) ;
net = add_block_test(net, opts, '0', KerenlSize, KerenlSize, 64, 64, 1, zeroPad,1,1) ;
net = add_reg_toss_test(net, '0',1);
net = add_pool_test(net, opts,'0', 2, 0);



% net = add_block_test(net, opts, '1_1', KerenlSize, KerenlSize, 64, 128, 1, zeroPad,1,1) ;
% net = add_block_test(net, opts, '1_2', KerenlSize, KerenlSize, 128, 128, 1, zeroPad,1,1) ;
% net = add_reg_toss_test(net, '1',2);
% net = add_pool_test(net, opts,'1', 2, 0);


net = add_block_test(net, opts, '1_1', KerenlSize, KerenlSize, 64, 128, 1, zeroPad,1,1) ;
net = add_block_test(net, opts, '1_2', KerenlSize, KerenlSize, 128, 128, 1, zeroPad,1,1) ;
net = add_reg_toss_test(net,'1',2);
net = add_pool_test(net, opts,'1', 2, 0);


% net = add_block_test(net, opts, '3_1', KerenlSize, KerenlSize, 256, 512, 1, zeroPad,1,1) ;
% net = add_block_test(net, opts, '3_2', KerenlSize, KerenlSize, 512, 512, 1, zeroPad,1,1) ;
% net = add_reg_toss_test(net,'3',4);
% net = add_pool_test(net, opts,'3',2,0);


net = add_block_test(net, opts, '2_1', KerenlSize, KerenlSize, 128, 256, 1, zeroPad,1,1) ;
net = add_block_test(net, opts, '2_2', KerenlSize, KerenlSize, 256, 256, 1, zeroPad,1,1) ;
net = add_block_convt_test(net, opts, '2_3', KerenlSize, KerenlSize, 256, 128, 2, [0 1 0 1],1,1) ;

net = add_reg_concat_test(net, '3_0',2);
net = add_block_test(net, opts, '3_1', KerenlSize, KerenlSize, 256, 128, 1, zeroPad,1,1) ;
net = add_block_test(net, opts, '3_2', KerenlSize, KerenlSize, 128, 128, 1, zeroPad,1,1) ;
net = add_block_convt_test(net, opts, '3_3', KerenlSize, KerenlSize, 128, 64, 2,  [0 1 0 1],1,1) ;


% net = add_reg_concat_test(net, '6_0',3);
% net = add_block_test(net, opts, '6_1', KerenlSize, KerenlSize, 512, 256, 1, zeroPad,1,1) ;
% net = add_block_test(net, opts, '6_2', KerenlSize, KerenlSize, 256, 256, 1, zeroPad,1,1) ;
% net = add_block_convt_test(net, opts, '6_3', KerenlSize, KerenlSize, 256, 128, 2,  [0 1 0 1],1,1) ;


% net = add_reg_concat_test(net, '5_0',2);
% net = add_block_test(net, opts, '5_1', KerenlSize, KerenlSize, 256, 128, 1, zeroPad,1,1) ;
% net = add_block_test(net, opts, '5_2', KerenlSize, KerenlSize, 128, 128, 1, zeroPad,1,1) ;
% net = add_block_convt_test(net, opts, '5_3', KerenlSize, KerenlSize, 128, 64, 2,  [0 1 0 1],1,1) ;

net = add_reg_concat_test(net, '4_0',1);
net = add_block_test(net, opts, '4_1', KerenlSize, KerenlSize, 128, 64, 1, zeroPad,1,1) ;
net = add_block_test(net, opts, '4_2', KerenlSize, KerenlSize, 64, 64, 1, zeroPad,1,1) ;
net = add_block_test(net, opts, '4_3', 1, 1, 64, ch_length_out, 1, 0,0,0) ;


info = vl_simplenn_display(net);
vl_simplenn_display(net)
net.meta.regNum = 3;
net.meta.regSize = [info.dataSize(1,end),info.dataSize(2,end),info.dataSize(3,end)]; 
 
% final touches
switch lower(opts.weightInitMethod)
    case {'xavier', 'xavierimproved'}
        net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
end
net.layers{end+1} = struct('type', 'euclideanloss', 'name', 'loss') ;
net.meta.inputSize = net.meta.normalization.imageSize ;
net.meta.augmentation.rgbVariance = zeros(0,3) ;
net.meta.augmentation.transformation = 'stretch' ;

if ~opts.batchNormalization
    lr = 1*logspace(-5.5, -7.5, 20) ; % generate 20 points between 10^-2 and 10^-3
else
    lr = logspace(-5.5, -7.5, 20) ;
end
net.meta.trainOpts.learningRate = lr ;

% Fill in default values; fill in missing default values in NET. 
net = vl_simplenn_tidy(net) ;
 
% Switch to DagNN if requested
switch lower(opts.networkType)
    case 'simplenn'
        % done
    case 'dagnn'
        net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
        net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
            {'prediction','label'}, 'top1err') ;
        net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
            'opts', {'topK',5}), ...
            {'prediction','label'}, 'top5err') ;
    otherwise
        assert(false) ;
end

% -----------------------------------------------------------------------------------------------------------------------------
%                                              Prepare data for training                                                  
% -----------------------------------------------------------------------------------------------------------------------------
if isempty(opts.imdb) 
    imdb = load(opts.imdbPath) ;
else 
    imdb = opts.imdb;
end 
% get batch
patchSize=opts.patchSize;
opts.learningRate = net.meta.trainOpts.learningRate ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end  % set the training size;
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end
if isnan(opts.val), opts.val = [] ; end

% -----------------------------------------------------------------------------------------------------------------------------
%                                                    Network initialization
% -----------------------------------------------------------------------------------------------------------------------------

net = vl_simplenn_tidy(net); % fill in some eventually missing values
net.layers{end-1}.precious = 1; % do not remove predictions, used for error
vl_simplenn_display(net, 'batchSize', opts.batchSize) ;

evaluateMode = isempty(opts.train) ;

if ~evaluateMode
    for i=1:numel(net.layers)
        if isfield(net.layers{i}, 'weights')
            J = numel(net.layers{i}.weights) ;
            for j=1:J
                net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
            end
            if ~isfield(net.layers{i}, 'learningRate')
                net.layers{i}.learningRate = ones(1, J, 'single') ;
            end
            if ~isfield(net.layers{i}, 'weightDecay')
                net.layers{i}.weightDecay = ones(1, J, 'single') ;
            end
        end
    end
end

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
    if isempty(gcp('nocreate')),
        parpool('local',numGpus) ;
        spmd, gpuDevice(opts.gpus(labindex)), end
    end
elseif numGpus == 1
    gpuDevice(opts.gpus)
end
if exist(opts.memoryMapFile), delete(opts.memoryMapFile) ; end

% setup error calculation function
hasError = true ;
if isstr(opts.errorFunction)
    switch opts.errorFunction
        case 'none'
            opts.errorFunction = @error_none ;
            hasError = false ;
        case 'multiclass'
            opts.errorFunction = @error_multiclass ;
            if isempty(opts.errorLabels), opts.errorLabels = {'top1err', 'top5err'} ; end
        case 'binary'
            opts.errorFunction = @error_binary ;
            if isempty(opts.errorLabels), opts.errorLabels = {'binerr'} ; end
        case 'euclideanloss'
            opts.errorFunction = @error_euclideanloss_test ;
            if isempty(opts.errorLabels), opts.errorLabels = {'mse'} ; end
        case 'euclideansparseloss'
            opts.errorFunction = @error_euclideanloss ;
            if isempty(opts.errorLabels), opts.errorLabels = {'mse'} ; end
        otherwise
            error('Unknown error function ''%s''.', opts.errorFunction) ;
    end
end

%%
% -----------------------------------------------------------------------------------------------------------------------------
%                                                        Train and validate
% -----------------------------------------------------------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start_e = opts.continue * findLastCheckpoint_test(opts.expDir) ;
if start_e >= 1
    fprintf('%s: resuming by loading epoch %d\n', mfilename, start_e) ;
    load(modelPath(start_e), 'net', 'info') ;
    net = vl_simplenn_tidy(net) ; % just in case MatConvNet was updated
end


for epoch=start_e+1:opts.numEpochs    % loop for epoch
    
    % train one epoch and validate
    learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    train = opts.train(randperm(numel(opts.train))) ; % shuffle
    val = opts.val ;

    if numGpus <= 1
        [net,stats.train,prof] = process_epoch_test(opts, patchSize, epoch, train, learningRate, imdb, net,Coef) ;
        [~,stats.val] = process_epoch_test(opts, patchSize, epoch, val, 0, imdb, net,Coef) ;
        if opts.profile
            profile('viewer') ;
            keyboard ;
        end
    else
        fprintf('%s: sending model to %d GPUs\n', mfilename, numGpus) ;
        spmd(numGpus)
            [net_, stats_train_,prof_] = process_epoch_test(opts, patchSize, epoch, train, learningRate, imdb, net,Coef) ;
            [~, stats_val_] = process_epoch_test(opts, patchSize, epoch, val, 0, imdb, net_,Coef) ;
        end
        net = net_{1} ;
        stats.train = sum([stats_train_{:}],2) ;
        stats.val = sum([stats_val_{:}],2) ;
        if opts.profile
            mpiprofile('viewer', [prof_{:,1}]) ;
            keyboard ;
        end
        clear net_ stats_train_ stats_val_ ;
    end
    
     % save
    if evaluateMode, sets = {'val'} ; else sets = {'train', 'val'} ; end
    for f = sets
        f = char(f);
        n = numel(eval(f));
        info.(f).speed(epoch) = n / stats.(f)(1) * max(1, numGpus) ;
        info.(f).objective(epoch) = stats.(f)(2) / n ;
        info.(f).error(:,epoch) = stats.(f)(3:end) / n ;
    end
    if ~evaluateMode
        fprintf('%s: saving model for epoch %d\n', mfilename, epoch) ;
        tic ;
        if mod(epoch,20)==1
            save(modelPath(epoch), 'net', 'info') ;
        elseif epoch==1
            save(modelPath(epoch), 'net', 'info','opts') ;
        end
        fprintf('%s: model saved in %.2g s\n', mfilename, toc) ;
    end
    
    if opts.plotStatistics
        switchfigure_test(1) ; clf ;
        subplot(1,1+hasError,1) ;
        if ~evaluateMode
            semilogy(max(1,epoch-2000):epoch, info.train.objective(max(1,epoch-2000):end), '.-', 'linewidth', 2) ;
            hold on ;
        end
        semilogy(max(1,epoch-2000):epoch, info.val.objective(max(1,epoch-2000):end), '.--') ;
        xlabel('training epoch') ; ylabel('energy') ;
        grid on ;
        h=legend(sets) ;
        set(h,'color','none');
        title('objective') ;
        if hasError
            subplot(1,2,2) ; leg = {} ;
            if ~evaluateMode
                plot(max(1,epoch-2000):epoch, info.train.error(max(1,epoch-2000):end)', '.-', 'linewidth', 2) ;
                hold on ;
                leg = horzcat(leg, strcat('train ', opts.errorLabels)) ;
            end
            plot(max(1,epoch-2000):epoch, info.val.error(max(1,epoch-2000):end)', '.--') ;
            leg = horzcat(leg, strcat('val ', opts.errorLabels)) ;
            set(legend(leg{:}),'color','none') ;
            grid on ;
            xlabel('training epoch') ; ylabel('error') ;
            title('error') ;
        end
        drawnow ;
        print(1, modelFigPath, '-dpdf') ;
    end
    
    %}
end


%% check and plot evaluate figure;
% Display_Results;




















