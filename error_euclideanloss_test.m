function err = error_euclideanloss_test(opts, labels, res)
% -------------------------------------------------------------------------
errTemp = res(end-1).x - labels ;
err  = sum(errTemp(:).^2)/numel(errTemp);