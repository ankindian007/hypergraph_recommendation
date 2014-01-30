function[AUC, accuracy, F, W, cost] = HyperPrior(data, label, test, alpha, rho, GL)

[m, n] = size(data);

% generate adjacency matirx
H = zeros(m, n * 2);
for i = 1 : n
    H(data(:, i) > 0, 2 * i - 1) = 1;
    H(data(:, i) < 0, 2 * i) = 1;
end

AUC_all = [];
accuracy_all = [];

% hold the labels of samples in the test set
Y_0 = label;
Y_0(Y_0 == 0) = -1;
Y_0(test) = 0;
Y_0(Y_0 == 1) = 1 / sum(Y_0 == 1);
Y_0(Y_0 == -1) = -1 / sum(Y_0 == -1);

D_e = sum(H, 1);
D_v = sum(H, 2);
W = ones(size(H, 2), 1);

f_a = Get_f_a(Y_0, H, D_e, D_v);
cost_all = [f_a * W + rho * W' * GL * W, f_a * W, rho * W' * GL * W, 0];
W_all = W';

% do alternative optimization
for i = 1 : 20
    % optimize F
    F = OptimizeF(H, D_e, D_v, W, Y_0, alpha);

    % calculate AUC and accuracy
    [B IX] = sort(F(test), 'ascend');
    ranked_label = label(test);
    ranked_label = ranked_label(IX);
    Num_Positive = sum(ranked_label == 1);
    AUC_tmp = (sum(find(ranked_label == 1)) - Num_Positive * (Num_Positive + 1) / 2) / (Num_Positive * (length(ranked_label) - Num_Positive));

    class = zeros(sum(test), 1);
    class(F(test) > 0) = 1;
    if sum(test) ~= 0
        cp = classperf(label, class, test);
        accuracy_tmp = cp.CorrectRate;
    else
        accuracy_tmp = 0;
    end

    AUC_all = [AUC_all; AUC_tmp];
    accuracy_all = [accuracy_all; accuracy_tmp];

    f_a = Get_f_a(F, H, D_e, D_v);

    cost_all = [cost_all; f_a * W + rho * W' * GL * W + (1 - alpha) / alpha * (F - Y_0)' * (F - Y_0), f_a * W, rho * W' * GL * W, (1 - alpha) / alpha * (F - Y_0)' * (F - Y_0)];

    % optimize W
    W = OptimizeW(f_a, H, rho, GL);
    W_all = [W_all; W'];

    cost_all = [cost_all; f_a * W + rho * W' * GL * W + (1 - alpha) / alpha * (F - Y_0)' * (F - Y_0), f_a * W, rho * W' * GL * W, (1 - alpha) / alpha * (F - Y_0)' * (F - Y_0)];

    % check the convergence
    if i > 1 && abs(cost_all(2 * i + 1, 1)- cost_all(2 * (i - 1) + 1, 1)) / cost_all(2 * (i - 1) + 1, 1) < 0.01
        break
    end
end

if i == 20
    disp('HyperPrior didn''t converge!')
end

AUC = AUC_all(end);
accuracy = accuracy_all(end);
cost = cost_all;

function[f_a] = Get_f_a(F, H, D_e, D_v)

[m, n] = size(H);
Nf = F ./ sqrt(D_v);

f_a = zeros(1, n);
for i = 1 : n
    ind = find(H(:, i));
    for j = 1 : length(ind)
        for k = 1 : length(ind)
            f_a(i) = f_a(i) + (Nf(ind(j)) - Nf(ind(k))) ^ 2;
        end
    end
    if f_a(i) ~= 0
        f_a(i) = f_a(i) / D_e(i) / 2;
    end
end

function[F] = OptimizeF(H, D_e, D_v, W, Y, alpha)

[m, n] = size(H);
tmp = zeros(m, n);
for i = 1 : n
    if D_e(i) ~= 0
        tmp(:, i) = H(:, i) * sqrt(W(i)) / sqrt(D_e(i));
    end
end
S = tmp * tmp';
for i = 1 : m
    for j = 1 : m
        if S(i, j) ~=0
            S(i, j) = S(i, j) / sqrt(D_v(i)) / sqrt(D_v(j));
        end
    end
end

F = Y;
for i = 1 : 10000
    F_old = F;
    F = alpha * S * F + (1 - alpha) * Y;

    if max(abs(F - F_old)) < 1e-9
        break
    end
end
if i == 10000
    disp('OptimizeF didn''t converge!')
end

function[W] = OptimizeW(f_a, H, rho, GL)

n = size(H, 2);
Aeq = H;
beq = sum(H, 2);
lb = zeros(n, 1);

options = optimset('LargeScale', 'off', 'MaxIter', 10000, 'Display', 'off');
W = quadprog(rho * (GL + GL'), f_a, [], [], Aeq, beq, lb, [], [], options);