function  [net_cpu,stats,prof] = check_results(opts, patchSize, epoch, subset, learningRate, imdb, net_cpu)
% -------------------------------------------------------------------------

% move the CNN to GPU (if needed)
numGpus = numel(opts.gpus) ;
if numGpus >= 1
    net = vl_simplenn_move(net_cpu, 'gpu') ;
    one = gpuArray(single(1)) ;
else
    net = net_cpu ;
    net_cpu = [] ;
    one = single(1) ;
end

% assume validation mode if the learning rate is zero
training = learningRate > 0 ;
if training
    mode = 'train' ;
    evalMode = 'normal' ;
else
    mode = 'val' ;
    evalMode = 'val' ;
end

% turn on the profiler (if needed)
if opts.profile
    if numGpus <= 1
        prof = profile('info') ;
        profile clear ;
        profile on ;
    else
        prof = mpiprofile('info') ;
        mpiprofile reset ;
        mpiprofile on ;
    end
end

res = [] ;
mmap = [] ;
stats = [] ;
start = tic ;

for t=1:opts.batchSize:numel(subset)
    fprintf('%s: epoch %02d: %3d/%3d: ', mode, epoch, ...
        fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
    batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
    numDone = 0 ;
    error = [] ;
    for s=1:opts.numSubBatches
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
        [im, labels, lowRes] = getSimpleNNBatch_test(imdb, batch, patchSize);
        
        if opts.prefetch
            if s==opts.numSubBatches
                batchStart = t + (labindex-1) + opts.batchSize ;
                batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
            else
                batchStart = batchStart + numlabs ;
            end
            nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
            getSimpleNNBatch_test(imdb, nextBatch, patchSize);
        end
        
        if numGpus >= 1
            im = gpuArray(im) ;
        end
        
        % evaluate the CNN
        net.layers{end}.class = labels ;
        net.layers{end}.lambda = opts.lambda ;
        if training, dzdy = one; else, dzdy = [] ; end
        
        [res, reg] = vl_simplenn_fbpconvnet(net, im, dzdy, res, ...
            'accumulate', s ~= 1, ...
            'mode', evalMode, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn) ;
        
        
        
        if (mod(t,5) == 1)%&&(rand>0.95)
            displayImg_test_final(labels,im,res,lowRes,opts);
        end
        
        % accumulate training errors
        error = sum([error, [...
            sum(double(gather(res(end).x))) ;
            reshape(opts.errorFunction(opts, labels, res),[],1) ; ]],2) ;
        numDone = numDone + numel(batch) ;
    end % next sub-batch
    
    % gather and accumulate gradients across labs
    if training
        if numGpus <= 1
            [net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res) ;
        else
            if isempty(mmap)
                mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
            end
            write_gradients(mmap, net, res) ;
            labBarrier() ;
            [net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res, mmap) ;
        end
    end
    
    % collect and print learning statistics
    time = toc(start) ;
    stats = sum([stats,[0 ; error]],2); % works even when stats=[]
    stats(1) = time ;
    n = t + batchSize - 1 ; % number of images processed overall
    speed = n/time ;
    fprintf('%.1f Hz%s\n', speed) ;
    
    m = n / max(1,numlabs) ; % num images processed on this lab only
    fprintf(' obj:%.3g', stats(2)/m) ;
    for i=1:numel(opts.errorLabels)
        fprintf(' %s:%.3e', opts.errorLabels{i}, stats(i+2)/m) ;
%         fprintf(' %s:%.3g', opts.errorLabels{i}, stats(i+2)/m) ;
    end
    fprintf(' [%d/%d]', numDone, batchSize);
    fprintf('\n') ;
    
    % collect diagnostic statistics
    if training & opts.plotDiagnostics
        switchfigure_test(2) ; clf ;
        diag = [res.stats] ;
        barh(horzcat(diag.variation)) ;
        set(gca,'TickLabelInterpreter', 'none', ...
            'YTickLabel',horzcat(diag.label), ...
            'YDir', 'reverse', ...
            'XScale', 'log', ...
            'XLim', [1e-5 1]) ;
        drawnow ;
    end
    
end

% switch off the profiler
if opts.profile
    if numGpus <= 1
        prof = profile('info') ;
        profile off ;
    else
        prof = mpiprofile('info');
        mpiprofile off ;
    end
else
    prof = [] ;
end

% bring the network back to CPU
if numGpus >= 1
    net_cpu = vl_simplenn_move(net, 'cpu') ;
else
    net_cpu = net ;
end



% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap)
% -------------------------------------------------------------------------
if nargin >= 6
    numGpus = numel(mmap.Data) ;
else
    numGpus = 1 ;
end

for l=numel(net.layers):-1:1
    for j=1:numel(res(l).dzdw)
        
        % accumualte gradients from multiple labs (GPUs) if needed
        if numGpus > 1
            tag = sprintf('l%d_%d',l,j) ;
            tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
            for g = setdiff(1:numGpus, labindex)
                tmp = tmp + mmap.Data(g).(tag) ;
            end
            res(l).dzdw{j} = res(l).dzdw{j} + tmp ;
        end
        
        if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
            % special case for learning bnorm moments
            thisLR = net.layers{l}.learningRate(j) ;
            net.layers{l}.weights{j} = ...
                (1-thisLR) * net.layers{l}.weights{j} + ...
                (thisLR/batchSize) * res(l).dzdw{j} ;
        else
            % standard gradient training
            thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
            thisLR = lr * net.layers{l}.learningRate(j) ;
            net.layers{l}.momentum{j} = ...
                opts.momentum * net.layers{l}.momentum{j} ...
                - thisDecay * net.layers{l}.weights{j} ...
                - (1 / batchSize) * res(l).dzdw{j} ;
            
            net.layers{l}.weights{j} = net.layers{l}.weights{j} + ...
                min(max(thisLR * net.layers{l}.momentum{j},-opts.gradMax),opts.gradMax) ;
        end
        
        % if requested, collect some useful stats for debugging
        if opts.plotDiagnostics
            variation = [] ;
            label = '' ;
            switch net.layers{l}.type
                case {'conv','convt'}
                    variation = thisLR * mean(abs(net.layers{l}.momentum{j}(:))) ;
                    if j == 1 % fiters
                        base = mean(abs(net.layers{l}.weights{j}(:))) ;
                        label = 'filters' ;
                    else % biases
                        base = mean(abs(res(l+1).x(:))) ;
                        label = 'biases' ;
                    end
                    variation = variation / base ;
                    label = sprintf('%s_%s', net.layers{l}.name, label) ;
            end
            res(l).stats.variation(j) = variation ;
            res(l).stats.label{j} = label ;
        end
    end
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, res, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.layers)
    for j=1:numel(res(i).dzdw)
        format(end+1,1:3) = {'single', size(res(i).dzdw{j}), sprintf('l%d_%d',i,j)} ;
    end
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
    f = fopen(fname,'wb') ;
    for g=1:numGpus
        for i=1:size(format,1)
            fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
        end
    end
    fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net, res)
% -------------------------------------------------------------------------
for i=1:numel(net.layers)
    for j=1:numel(res(i).dzdw)
        mmap.Data(labindex).(sprintf('l%d_%d',i,j)) = gather(res(i).dzdw{j}) ;
    end
end

