function [values dual_terms] = computeLdual(pri_terms2,Beta,W,U,J,P,Q,R,rho,mu,xi)

[m s]=size(W);

dual_terms = [];

% term1-3
dual_terms = [dual_terms pri_terms2];

% term4-5
term = 0; term1 = 0;
for v = 1:s
    mat = -U{v} + W(:,v)*Beta{v}';
    %term = term + trace(P{v}'*mat);
    term = term + sum(diag(P{v}'*mat));
    term1 = term1 + rho/2*norm(mat,'fro');
end
dual_terms = [dual_terms term term1];

% term6
mat = Q - mu*W;
term = 0;
for i = 1:m
    for v = 1:s
        if mat(i,v)>0
            term = term + mat(i,v)*mat(i,v) - Q(i,v)*Q(i,v);
        else
            term = term - Q(i,v)*Q(i,v);
        end
    end
end
term = term*0.5/mu;
dual_terms = [dual_terms term];

% % term6-7
% mat = Q - mu*W;
% term = 0;term1 = 0;
% for i = 1:m
%     for v = 1:s
%         if mat(i,v)>0
%             term = term - W(i,v)*Q(i,v);
%             term1 = term1 + (mu*W(i,v))^2;
%         else
%             term = term - 1/mu*Q(i,v)*Q(i,v);
%             term1 = term1 + Q(i,v)*Q(i,v);
%         end
%     end
% end
% term1 = term1*0.5/mu;
% dual_terms = [dual_terms term term1];

% term8-9
mat = J-W;
term = trace(R'*mat);
term1 = xi/2*norm(mat,'fro');
dual_terms = [dual_terms term term1];

values = sum(dual_terms);