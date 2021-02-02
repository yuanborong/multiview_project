function [values pri_terms2] = computeLpri2(Beta,U,J,lambdaR,lambdaS,train_label,train_data)

s=length(train_data);
m=length(train_label);

pri_terms2 = [];

% term1: (1/2m)||Y-\sum_v diag(F^v*U{v}')||_2^2
tmp = zeros(m,1);
for v = 1:s
    vec = diag(train_data{v}*U{v}');
    tmp = tmp + vec;
end
term = norm(train_label - tmp);
term = 0.5/m*term*term;
pri_terms2 = [pri_terms2 term];

% term2
term = 0;
n = 0;
for v = 1:s
    term = term + norm(Beta{v},1);
    n = n + length(Beta{v});
end
%term = term * lambdaS/n;
term = term * lambdaS/s;
pri_terms2 = [pri_terms2 term];

% term3
sval = svd(J);
term = lambdaR*sum(sval);
pri_terms2 = [pri_terms2 term];

values = sum(pri_terms2);