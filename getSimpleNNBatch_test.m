function [images, labels, lowRes] = getSimpleNNBatch_test(imdb, batch, patchSize)
% -------------------------------------------------------------------------
Ny = size(imdb.images.noisy,1);
Nx = size(imdb.images.noisy,2);
pos_x = round(rand(1)*(Nx-patchSize));
pos_y = round(rand(1)*(Ny-patchSize));
images = single(imdb.images.noisy(pos_y+(1:patchSize),pos_x+(1:patchSize),:,batch)) ;
labels = single(imdb.images.orig(pos_y+(1:patchSize),pos_x+(1:patchSize),:,batch)) ;
% if rand > 0.5
%     labels=fliplr(labels);  % left right rotate
%     images=fliplr(images);
% end
% if rand > 0.5
%     labels=flipud(labels);  % up down rotate
%     images=flipud(images);
% end
lowRes = images(:,:,1,:);
labels(:,:,1,:) = labels(:,:,1,:) - lowRes;
