function displayImg_test(labels,im,res,lowRes,opts,Coef)


imTemp = im(:,:,:,1);
labelsTemp = labels(:,:,:,1);
labelsTemp(:,:,1,:) = labelsTemp(:,:,1,:) +lowRes(:,:,1);
recTemp =  res(end-1).x(:,:,:,1);
recTemp(:,:,1,:) = recTemp(:,:,1,:) +lowRes(:,:,1);


figure(12);
subplot(1,3,1); imagesc((labelsTemp/Coef)+1); colormap(gray); axis off image;title('original');
subplot(1,3,2); imagesc((imTemp/Coef)+1); colormap(gray); axis off image; title('BP');
subplot(1,3,3); imagesc((recTemp/Coef)+1); colormap(gray); axis off image;title('Reconstruction');
drawnow;
pause(.1);