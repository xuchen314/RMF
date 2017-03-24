% This script is for the papers:
% Zhouchen Lin, Chen Xu,and Hongbin Zha, Robust Matrix Factorization by Majorization-Minimization, T-PAMI, accepted
% Chen Xu, Zhouchen Lin,and Hongbin Zha, Relaxed Majorization-Minimization for Non-smooth and Non-convex Optimization, AAAI-16
% If you have any questions, feel free contract Chen Xu (xuen@pku.edu.cn)
%
% Copyright: Peking University


nview=7;  % Number of views, [5,6,7]

%% data preprocessing
viff = load('viff.txt'); %The raw dinosaur tracks from http://www.robots.ox.ac.uk/?vgg/data1.html
II = (viff == -1);       % find where ends of tracks marked - indicator var
viff(II) = nan;          %changes these to nan
x = viff(1:end,1:2:72)'; % pull out x coord of all tracks
II = isfinite(x);        % selects tracks apart from nans
JJ = sum(II) >=nview;    % tracks longer than  nviews
M = viff(JJ,:)';
II = find(~isfinite(M));
M(II) = 0;
W = (M>0); % The mask matrix

%Register the image origin to the image center, (360, 288),
%by which the intrinsic rank remains to be 4 under the affine SfM model
M_cen = M;
M_cen(1:2:end,:) = M(1:2:end,:)-360;  
M_cen(2:2:end,:) = M(2:2:end,:)-288;
M_cen = W.*M_cen;

%% Inilialization
[U,S,V] = svds(M_cen,4); 
U0 = U*sqrt(S);
V0 = V*sqrt(S);

%%
clear para
[m,n] = size(M);
para.lambda_u = 20/(m+n);
para.lambda_v = 20/(m+n);
para.rho =1.5;
tic
[U_mm,V_mm] = RMF_MM(W,M_cen,U0,V0,para);

time_mm = toc;
M_mm = U_mm*V_mm';
Err1 = sum(sum(abs(W.*(M_cen-M_mm))))/sum(W(:));
fprintf('Err1=%1.2d,  Time=%2.2d \n',Err1,time_mm);

%% Show the recovered tracks
x = M_mm(1:2:72,:); % pull out x coord of all tracks
y = M_mm(2:2:72,:); % pull out y coord of all tracks
fig = figure('PaperSize',[5,4],'PaperPosition',[0,0,5,4],'color',[1,1,1]);
plot(x,y)
axis ij
axis([-350,350,-300,300])
axis off
box off
