function [values pri_terms1] = computeLpri1(Beta,W,lambdaR,lambdaS,train_label,train_data)

s=length(train_data);
m=length(train_label);

pri_terms1 = [];

% term1: (1/2m)||Y-\sum_v W(:,v).*(F^v*Beta{v})||_2^2
tmp = zeros(m,1);
for v = 1:s
    vec = train_data{v}*Beta{v};
    tmp = tmp + W(:,v).*vec;
end
term = norm(train_label - tmp);
term = 0.5/m*term*term;
pri_terms1 = [pri_terms1 term];

% term2
term = 0;
n = 0;
for v = 1:s
    term = term + norm(Beta{v},1);
    n = n + length(Beta{v});
end
%term = term * lambdaS/n;
term = term * lambdaS/s;
pri_terms1 = [pri_terms1 term];

% term3
Wr = roundn(W,-4);
term = lambdaR*rank(Wr);
pri_terms1 = [pri_terms1 term];

values = sum(pri_terms1);