function [epsil,Para] = annulus_gen_rand(MAX,Mx,N_cir,R_range,Ep_range)
% Generate N_cir circles
% MAX: For the region of length , say Length, MAX=Length/2;
%       (if the domain of investigation is assumed to be square region centred at (0,0), and the size of the region is –MAX to MAX (thus a total of 2 MAX))
% Mx: length of (-MAX:Step_size:MAX)
% N_cir: The number of circular;
% R_range: The radius range
% Ep_range:range.

tmp = linspace(-MAX,MAX,Mx); [x0, y0] = meshgrid(tmp, -tmp); % Mx by Mx
epsil = ones(size(y0));
Para=zeros(N_cir,5);
for nn=1:N_cir
 
radius=R_range(1)+(R_range(2)-R_range(1))*rand; % radius of the circle

L_range=[-1+radius+0.05,1-radius-0.05];  % location range
L_x=L_range(1)+(L_range(2)-L_range(1))*rand;  % x coordinate for center
L_y=L_range(1)+(L_range(2)-L_range(1))*rand;  % x coordinate for center 

dist = sqrt((x0-L_x).^2+(y0-L_y).^2);
bool_annul =dist<radius;

Ep=Ep_range(1)+(Ep_range(2)-Ep_range(1))*rand; % relative permittivity of the circle 
epsil(bool_annul) = Ep;

Para(nn,:)=[N_cir,radius,L_x,L_y,Ep];

end