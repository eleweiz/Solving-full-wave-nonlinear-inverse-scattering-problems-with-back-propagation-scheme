function [net,res] = accumulate_gradients_test(opts, lr, batchSize, net, res, mmap)
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