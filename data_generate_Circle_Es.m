%% save forward data including Profile parameters and scattered field;
% Wirtten by Wei Zhun at NUS on 18 Nov, 2017


clc; clear all;close all;
%% basic paramters
eta_0 = 120*pi;
c = 3e8;
eps_0 = 8.85e-12;
lam_0 = 0.75; % lambda:  0.75 m is for 400 MHz
k_0 = 2*pi/lam_0;
omega = k_0*c;
bool_plane = 0;  % 1: Plane wave incidence; 0: Line source incidence
Coef=i*k_0*eta_0;
N_rec = 32;    % Nb. of Receiver
N_inc = 16;  % Nb. of Incidence
% save('basic_para.mat')
% return
% discritization 
MAX = 1; Mx = 64;
step_size = 2*MAX/(Mx-1);
cell_area = step_size^2; % the area of the sub-domain
a_eqv = sqrt(cell_area/pi); % the equivalent radius of the sub-domain for the circle approximation
Mnr=0.15; Mxr=0.4;  % radius range, center range will be decided by specific radius value;
Ep_s=1+1e-3; Ep_l=1.5;  % permittivity range
N_t=500; % total data number;
E_s=zeros(N_rec,N_inc,N_t);
Pro_Para=cell(1,N_t);

for nn=1:N_t

%% generate total profile 
tmp_domain = linspace(-MAX,MAX,Mx); 
[x_dom,y_dom] = meshgrid(tmp_domain, -tmp_domain);
N_cell_dom = size(x_dom,1)*size(x_dom,2);
x0 = x_dom; y0 = y_dom;

%% generate perturbations
N_cir=ceil((3*rand)+1e-2);
[epsil,Para] = annulus_gen_rand(MAX,Mx,N_cir,[Mnr,Mxr],[Ep_s,Ep_l]);
Pro_Para{1,nn}=Para;

xi_all = -i*omega*(epsil-1)*eps_0*cell_area;  
bool_eps = epsil==1; in_anulus = find(bool_eps==0); in_anulus = in_anulus(:);
x0(bool_eps) = []; y0(bool_eps) = []; 
x0 = x0(:); y0 = y0(:);
xi_forward = xi_all; xi_forward(bool_eps) =[]; xi_forward = xi_forward(:);
N_cell = length(x0);

epsil_t=epsil;
epsil_t(bool_eps)=[];epsil_t=epsil_t(:);

% reciver

theta_tmp = linspace(0, 2*pi, N_rec+1); theta_tmp(end) = []; theta_tmp = theta_tmp(:);
[theta,rho] = meshgrid(theta_tmp,3); theta = theta(:); rho = rho(:);
[x,y] = pol2cart(theta,rho);

if bool_plane == 1
    %Plane wave incidence
    theta_inc = linspace(0, 2*pi, N_inc+1); theta_inc(end) = []; theta_inc = theta_inc(:);
    k_x = k_0*cos(theta_inc); k_y = k_0*sin(theta_inc);
    E_inc = exp(i*x0*k_x.' +i*y0*k_y.'); %  N_cell x N_inc
else
    %Line source incidence
    theta_inc = linspace(0, 2*pi, N_inc+1); theta_inc(end) = []; theta_inc = theta_inc(:);
    [theta_t,rho_t] = meshgrid(theta_inc,3); theta_t = theta_t(:); rho_t = rho_t(:);
    [x_t,y_t] = pol2cart(theta_t,rho_t);

    [x0_tmp_t, x_tmp_t] = meshgrid(x0,x_t);
    [y0_tmp_t, y_tmp_t] = meshgrid(y0,y_t);
    rho_mat_t = sqrt((x0_tmp_t-x_tmp_t).^2 +(y0_tmp_t-y_tmp_t).^2); % N_inc x N_cell
    T_mat = i*k_0*eta_0 *(i/4)*besselh(0,1,k_0*rho_mat_t); % N_inc x N_cell
    E_inc = T_mat.'; % N_cell x N_inc

    clear x0_tmp_t x_tmp_t y0_cell_t y0_cell_2_t rho_mat_t T_mat
end

% MOM results -----------------------------------------
Phi_mat = zeros(N_cell);  % GD matrix
[x0_cell, x0_cell_2] = meshgrid(x0,x0);
[y0_cell, y0_cell_2] = meshgrid(y0,y0);
dist_cell =  sqrt((x0_cell-x0_cell_2).^2 +(y0_cell-y0_cell_2).^2);
dist_cell = dist_cell + eye(N_cell);
I1=(i/4)*besselh(0,1,k_0*dist_cell);  % off-diagonal integral of Green's function;
Phi_mat = Coef * I1;
Phi_mat = Phi_mat .*(ones(N_cell)-eye(N_cell)); % Diagonal is set to zero
I2=(i/4)*(2/(k_0*a_eqv)*besselh(1,1,k_0*a_eqv)+4*1i/(k_0^2*cell_area));   % diagonal integral of Green's function;
S1=Coef*I2;  % Analytical sigular value for self-cell integral(exclude cell area);
Phi_mat=Phi_mat+S1*eye(N_cell);  % Add the self-contribution.
E_tot = (eye(N_cell)-Phi_mat*diag(xi_forward))\E_inc;  % E_tot = E_inside for 2D   % N_cell x N_inc

[x0_tmp, x_tmp] = meshgrid(x0,x);
[y0_tmp, y_tmp] = meshgrid(y0,y);
rho_mat = sqrt((x0_tmp-x_tmp).^2 +(y0_tmp-y_tmp).^2); % N_rec x N_cell
R_mat = Coef *(i/4)*besselh(0,1,k_0*rho_mat); % N_rec x N_cell
E_CDM = R_mat *diag(xi_forward) *E_tot; % N_rec x N_inc
E_s(:,:,nn)=E_CDM;
nn
end
clearvars -except E_s Pro_Para;
save('Forward_Circ1.mat')


