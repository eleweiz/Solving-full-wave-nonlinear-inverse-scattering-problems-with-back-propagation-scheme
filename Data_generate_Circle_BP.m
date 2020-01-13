% This code is used to produce ground truth pattern and backprojection
% initializations;
% Written by Wei Zhun at ECE, NUS, 20th, Nov, 2017.

clc; clear all;
close all;

load('basic_para.mat');
load('Forward_Circ1.mat');
N_t=size(Pro_Para,2);
MAX = 1; Mx = 64; % discretization parameter
epsil_exa=zeros(Mx,Mx,1,N_t,'single');
epsil_bp=zeros(Mx,Mx,1,N_t,'single');
NS_L=0.05;  % noise level;
% NS_L=0.2;  % noise level;
%% discritization 
tmp_domain = linspace(-MAX,MAX,Mx); 
[x_dom,y_dom] = meshgrid(tmp_domain, -tmp_domain);
N_cell_dom = size(x_dom,1)*size(x_dom,2);
x0 = x_dom; y0 = y_dom;
step_size = 2*MAX/(Mx-1);
cell_area = step_size^2; % the area of the sub-domain
a_eqv = sqrt(cell_area/pi); % the equivalent radius of the sub-domain for the circle approximatio

%% reciver and incidence
% reciver
N_rec = 32;    % Nb. of Receiver
theta_tmp = linspace(0, 2*pi, N_rec+1); theta_tmp(end) = []; theta_tmp = theta_tmp(:);
[theta,rho] = meshgrid(theta_tmp,3); theta = theta(:); rho = rho(:);
[x,y] = pol2cart(theta,rho);
% Incident field 
if bool_plane ==1
    %Plane wave incidence
    E_inc_dom = exp(i*x_dom*k_x.' +i*y_dom*k_y.'); %  N_cell_dom x N_inc
else
    %Line source incidence
    theta_inc = linspace(0, 2*pi, N_inc+1); theta_inc(end) = []; theta_inc = theta_inc(:);
    [theta_t,rho_t] = meshgrid(theta_inc,3); theta_t = theta_t(:); rho_t = rho_t(:);
    [x_t,y_t] = pol2cart(theta_t,rho_t);
    % Point source incidence
    [x_dom_tmp_t, x_tmp_t] = meshgrid(x_dom,x_t);
    [y_dom_tmp_t, y_tmp_t] = meshgrid(y_dom,y_t);
    rho_dom_mat_t = sqrt((x_dom_tmp_t-x_tmp_t).^2 +(y_dom_tmp_t-y_tmp_t).^2); % N_inc x N_cell_dom
    T_mat_dom = i*k_0*eta_0 *(i/4)*besselh(0,1,k_0*rho_dom_mat_t); % N_inc x N_cell_dom
    E_inc_dom = T_mat_dom.'; % N_cell_dom x N_inc
    clear x_dom_tmp_t x_tmp_t y_dom_tmp_t y_tmp_t rho_dom_mat_t T_mat_dom
end

%% Gs matrix
x_dom = x_dom(:);
y_dom = y_dom(:);
[x_dom_tmp, x_tmp] = meshgrid(x_dom,x);
[y_dom_tmp, y_tmp] = meshgrid(y_dom,y);
rho_dom_mat = sqrt((x_dom_tmp-x_tmp).^2 +(y_dom_tmp-y_tmp).^2); % N_rec x N_cell_dom
R_dom_mat = i*k_0*eta_0 *(i/4)*besselh(0,1,k_0*rho_dom_mat);   % N_rec x N_cell_dom

%% GD matrix
[x_dom_cell, x_dom_cell_2] = meshgrid(x_dom,x_dom);
x_dist_tmp = (x_dom_cell-x_dom_cell_2).^2;
clear x_dom_cell x_dom_cell_2
[y_dom_cell, y_dom_cell_2] = meshgrid(y_dom,y_dom);
y_dist_tmp = (y_dom_cell-y_dom_cell_2).^2;
clear y_dom_cell y_dom_cell_2 
dist_tmp = x_dist_tmp + y_dist_tmp;
clear x_dist_tmp y_dist_tmp
dist_dom_cell = sqrt(dist_tmp);
clear dist_tmp
dist_dom_cell = dist_dom_cell+ eye(N_cell_dom);
Phi_mat_dom = i*k_0*eta_0 *(i/4)*besselh(0,1,k_0*dist_dom_cell);
clear dist_dom_cell
diag_zero = ones(N_cell_dom)-eye(N_cell_dom);
Phi_mat_dom = Phi_mat_dom .*diag_zero; % Diagonal is set to zero
I2=(i/4)*(2/(k_0*a_eqv)*besselh(1,1,k_0*a_eqv)+4*1i/(k_0^2*cell_area));   % diagonal integral of Green's function;
S1=Coef*I2;  % Analytical sigular value for self-cell integral(exclude cell area);
Phi_mat_dom=Phi_mat_dom+S1*eye(N_cell_dom);  % Add the self-contribution.
clear diag_zero

for nn=1:N_t

%% grondtruth pattern
epsil_exa(:,:,1,nn)= annulus_gen_exa(MAX,Mx,Pro_Para{nn});

%% BP pattern
% adding noise
E_sG=E_s(:,:,nn);  % no noise signal
rand_real = randn(N_rec,N_inc);
rand_imag = randn(N_rec,N_inc);
E_Gaussian = 1/sqrt(2) *sqrt(1/N_rec/N_inc)*norm(E_sG(:)) *NS_L*(rand_real +i*rand_imag);
E_sN = E_sG +E_Gaussian;

% Back propagation initial guess
J_init = zeros(N_cell_dom,N_inc);
for ii = 1:N_inc
    gamma_tmp = (R_dom_mat*(R_dom_mat'*E_sN(:,ii)))\E_sN(:,ii);  %scalar
    J_init(:,ii) = gamma_tmp*(R_dom_mat'*E_sN(:,ii)); 
end
E_tot_init = E_inc_dom +Phi_mat_dom*J_init;  % N_cell_dom x N_inc
up_tmp = sum(conj(E_tot_init).*J_init,2);
clear J_init
down_tmp = sum(conj(E_tot_init).*E_tot_init,2);
clear E_tot_init
x_int = imag(up_tmp./down_tmp);
clear up_tmp down_tmp
x_int = x_int/(-omega*eps_0*step_size^2)+1;  % change to epsilon
epsil_bp(:,:,1,nn) = reshape(x_int, Mx, Mx);
nn
end

%% display a rand one parameter
Indx=round(1+(N_t-1)*rand);  % produce a rand integer between 1 and N_t;
[x_dom,y_dom] = meshgrid(tmp_domain, -tmp_domain);
figure
pcolor(x_dom,y_dom,epsil_bp(:,:,1,Indx)); axis square; axis tight; shading flat;
title('BP Results');
colorbar
figure
pcolor(x_dom,y_dom,epsil_exa(:,:,1,Indx)); axis square; axis tight; shading flat;
colorbar
title('Ground Truth Results');

clearvars -except epsil_bp epsil_exa;
save('CNN_Data_Cir.mat')



