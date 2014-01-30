
function[F] = HyperModified(H, label, test, alpha, W)

% Initial Labels
Y_0 = label;

D_e = sum(H, 1);
D_v = sum(H, 2);

% optimize F
F = OptimizeF(H, D_e, D_v, W, Y_0, alpha);

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