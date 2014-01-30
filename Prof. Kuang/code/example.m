clc
clear

% generate an artifical dataset
mu = 0.6 * ones(1, 10);
SIGMA = rand(10, 10);
SIGMA = (SIGMA + SIGMA') / 4;
for i = 1 : 10
    SIGMA(i, i) = 1;
end
data = [[mvnrnd(mu, SIGMA, 50); mvnrnd(-mu, SIGMA, 50)] rand(100, 90)];
label = [zeros(50, 1); ones(50, 1)];

% use an artifical graph Laplacian matrix as prior knowledge
A = 0.1 * (ones(100, 100) - eye(100));
for i = 1 : 10
    for j = i + 1 : 10
        A(i, j) = 1;
        A(j, i) = 1;
    end
end
A = kron(A, ones(2, 2));
D = sum(A);
D(D == 0) = 1;
L = diag(D) - A;
for i = 1 : length(D)
    for j = 1 : length(D)
        L(i, j) = L(i, j) / sqrt(D(i)) / sqrt(D(j));
    end
end

% randomly select 20% of data as test set and set the parameters for HyperPrior
[Train, Test] = crossvalind('HoldOut', label, 0.2);
mu = 1;
rho = 1;

% semi-supervised learning by HyperPrior
[AUC, accuracy] = HyperPrior(data, label, Test, mu, rho, L);
disp(['The AUC of HyperPrior is ' num2str(AUC) ' for this experiment.']);