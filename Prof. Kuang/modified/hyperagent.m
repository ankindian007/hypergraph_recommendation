%{
% Hyper-incidence matrix.
m = 29; % # Vertices
n = 7; % # Edges
H = zeros(n,m);
H(1,1)=1; H(1,2)=1;
H(2,2:7)=1;
H(3,5:13)=1; H(3,27:29)=1;
H(4,26:29)=1; H(4,14:15)=1;
H(5,14:20)=1;
H(6,16:17)=1; H(6,21:23)=1;
H(7,18:20)=1; H(7,24:25)=1;
H=H';

%Mul = ones(n,1); % Multiplicity
Mul = [3 7 5 4 2 2 2]';
Car = sum(H,1)'; % Nothing but D_e
W = Mul./Car;

label = zeros(29, 1);
label(1:2,1)=1; label(26:29,1)=1; label(14:15,1)=1;

% randomly select 20% of data as test set and set the parameters for HyperPrior
[Train, Test] = crossvalind('HoldOut', label, 0.2);
mu = 0.1; rho = 1;

% semi-supervised learning by HyperPrior
[F] = HyperModified(H, label, Test, mu, W);
[~,idx] = sort(F,'descend');
idx
%disp(['The AUC of HyperPrior is ' num2str(AUC) ' for this experiment.']);
%}

% +++++++++++++++++++++++ Temporal Recommendation ++++++++++++++++++++++++
% Hyper-incidence Tensor

% Code for the sparse tensor incidence matrix.
%{
subs = [1 1 1; 1 1 3; 2 2 2; 4 4 4; 1 1 1; 1 1 1]
vals = [0.5; 1.5; 2.5; 3.5; 4.5; 5.5]
siz = [4 4 4];
H = sptensor(subs,vals,siz);
%}

% Hyper-incidence matrix.
m = 29; % # Vertices
n = 7; % # Edges
H = zeros(n,m);
H(1,1)=1; H(1,2)=1;
H(2,2:7)=1;
H(3,5:13)=1; H(3,27:29)=1;
H(4,26:29)=1; H(4,14:15)=1;
H(5,14:20)=1;
H(6,16:17)=1; H(6,21:23)=1;
H(7,18:20)=1; H(7,24:25)=1;
H=H';

% Temporal Incidence Non-Sparse Tensor.
HT(:,:,1)=H;
HT(:,:,2)=H/2;
HT(:,:,3)=H/3;
HT(:,:,4)=H/4;

% Multiplicities / Cardinalities.
Mul = [3 7 5 4 2 2 2]';
Car = sum(H,1)'; % Nothing but D_e
W = Mul./Car;

% Weight Matrix.
WM(:,1)=W;
WM(:,2)=W/2;
WM(:,3)=2*W;
WM(:,4)=W;

% Label Matrix.
k = 4;
label = zeros(29, k);
label(1,1)=1; label(1,2)=1; label(1,3)=1; label(1,4)=1;

% Parameters.
Alpha = 1; mu = 0.5; gamma = 0.5; J_user = zeros(29,1); J_user(1:5,:) = 1;

% Calling the main code.
[F] = HyperTemporalRecom(HT, label, 0, Alpha, WM, k, mu, gamma, J_user);
[~,idx] = sort(F, 1, 'descend');
idx