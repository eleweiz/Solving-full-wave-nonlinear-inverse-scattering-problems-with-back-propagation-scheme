clc; clear all;close all;
load ('./training_result/net-epoch-201.mat');
load CNN_Data_Cir.mat;

load CNN_Data_Cir_S1_N20.mat;

% load ('./training_result/net-epoch-201.mat');
% load CNN_Data_Cir_N20.mat;

% run D:/Software/Matlab/Neural_Network/5_CNN_test/FBPConvNet-master/matconvnet-1.0-beta23/matlab/vl_setupnn
% run D:/Software/Matlab/Neural_Network/5_CNN_test/FBPConvNet-master/matconvnet-1.0-beta23/matlab/vl_compilenn
%% opts
W   = 64; % size of patch
Nimg= 500; % # of train + test set
Nimg_test= fix(Nimg*0.05);
Coef=10;

id_tmp  = ones(Nimg,1);
id_tmp(Nimg-Nimg_test+1:end)=2;  % the test indx is 2, training is 1

imdb.images.set=id_tmp;             % train set : 1 , test set : 2
imdb.images.noisy=single((epsil_bp-1)*Coef);    % input  : H x W x C x N (X,Y,channel,batch)
imdb.images.orig=single((epsil_exa-1)*Coef);     % output : H x W x C x N (X,Y,channel,batch)
imdb.images.noisy(:,:,1,500)=single((epsil_bpS1-1)*Coef);    % input  : H x W x C x N (X,Y,channel,batch)
imdb.images.orig(:,:,1,500)=single((epsil_exaS1-1)*Coef);     % output : H x W x C x N (X,Y,channel,batch)
% opts
opts.channel_in = 1;
opts.channel_out=1;
opts.useGpu = 'false'; %'false'
opts.gpus = [] ;       % []
patchSize = W;
%%
tic
for t=500
batch=500;
[im, labels, lowRes] = getSimpleNNBatch_test(imdb, batch, patchSize);
dzdy=[];res=[];s=1; evalMode = 'val' ;
opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = true ;
[res, reg] = vl_simplenn_fbpconvnet(net, im, dzdy, res, ...
            'accumulate', s ~= 1, ...
            'mode', evalMode, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn) ;
  
[epsil_exa,epsil_bp,epsil_rec]=displayImg_test_final(labels,im,res,lowRes,Coef);
toc
err_bp=norm(reshape(epsil_bp,[],1)-reshape(epsil_exa,[],1))/norm(reshape(epsil_exa,[],1));
err_rec=norm(reshape(epsil_rec,[],1)-reshape(epsil_exa,[],1))/norm(reshape(epsil_exa,[],1));

end
err_rec
mean(err_rec)

mean(err_rec)
MAX = 1; Mx = 64; % discretization parameter
tmp_domain = linspace(-MAX,MAX,Mx);
[x_dom,y_dom] = meshgrid(tmp_domain, -tmp_domain);
figure
set(0,'DefaultaxesFontSize',22);
set(0,'DefaulttextFontSize',22);
xlabel('x (m)');
ylabel('y (m)');
pcolor(x_dom,y_dom,epsil_exa); axis square; axis tight; shading flat;colorbar;colormap(jet);
xlabel('x (m)');
ylabel('y (m)');
print('-djpeg','-r200','-painters','Fig11');
figure
pcolor(x_dom,y_dom,epsil_bp); axis square; axis tight; shading flat;colorbar;colormap(jet);
set(0,'DefaultaxesFontSize',22);
set(0,'DefaulttextFontSize',22);
xlabel('x (m)');
ylabel('y (m)');
print('-djpeg','-r200','-painters','Fig12');
% title('BP Results');
figure
pcolor(x_dom,y_dom,epsil_rec); axis square; axis tight; shading flat;colorbar;colormap(jet);
set(0,'DefaultaxesFontSize',22);
set(0,'DefaulttextFontSize',22);

xlabel('x (m)');
ylabel('y (m)');
print('-djpeg','-r200','-painters','Fig13');
