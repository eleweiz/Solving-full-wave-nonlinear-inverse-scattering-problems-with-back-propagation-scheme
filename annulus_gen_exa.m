function epsil = annulus_gen_exa(MAX,Mx,Para)
% Generate N_cir circles
% MAX: For the region of length , say Length, MAX=Length/2;
%       (if the domain of investigation is assumed to be square region centred at (0,0), and the size of the region is –MAX to MAX (thus a total of 2 MAX))
% Mx: length of (-MAX:Step_size:MAX)
% N_cir: The number of circular;
% R_range: The radius range
% Ep_range:range.

tmp = linspace(-MAX,MAX,Mx); [x0, y0] = meshgrid(tmp, -tmp); % Mx by Mx
epsil = ones(size(y0));

for nn=1:Para(1,1)
 
radius=Para(nn,2); % radius of the circle
L_x=Para(nn,3);  % x coordinate for center
L_y=Para(nn,4);  % x coordinate for center 

dist = sqrt((x0-L_x).^2+(y0-L_y).^2);
bool_annul =dist<radius;

Ep=Para(nn,5); % relative permittivity of the circle 
epsil(bool_annul) = Ep;


end