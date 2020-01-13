function [x1,x2,x3]=displayImg_test_final(labels,im,res,lowRes,Coef)


imTemp = im(:,:,:,1);
labelsTemp = labels(:,:,:,1);
labelsTemp(:,:,1,:) = labelsTemp(:,:,1,:) +lowRes(:,:,1);
recTemp =  res(end-1).x(:,:,:,1);
recTemp(:,:,1,:) = recTemp(:,:,1,:) +lowRes(:,:,1);


x1=(labelsTemp/Coef)+1;x2=(imTemp/Coef)+1;x3=(recTemp/Coef)+1;
figure;
subplot(1,3,1); imagesc(x1); colormap(gray); axis off image;title('original');
subplot(1,3,2); imagesc(x2); colormap(gray); axis off image; title('BP');
subplot(1,3,3); imagesc(x3); colormap(gray); axis off image;title('CNN');
drawnow;
pause(.1);