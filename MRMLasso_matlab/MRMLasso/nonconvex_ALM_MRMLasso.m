function [Beta,W]=nonconvex_ALM_MRMLasso(train_data,train_label,Lasso_para)

    tol = 1e-4;
    s=length(train_data);
    m=length(train_label);

    lambdaR=Lasso_para.lambdaR;
    lambdaS=Lasso_para.lambdaS;

    spliced_data = [];
    for v = 1:s
       spliced_data = [spliced_data train_data{v}];
    end
    
%% initializing W,beta, lagrange multipliers P,Q,R, /rho,/mu,/xi
    num_feas=[];
    Betapre = cell(s,1);
    for v = 1:s
        Betapre{v} = rand(size(train_data{v},2),1);
        num_feas=[num_feas; size(train_data{v},2)];
    end
    all_features = sum(num_feas);
    randW = rand(m,s);
    Wpre = zeros(m,s);W = zeros(m,s);
    for i=1:m
        Wpre(i,:)=randW(i,:)./sum(randW(i,:));
    end
    for v = 1:s
        P{v} = zeros(m,num_feas(v,1));
    end
    Q = zeros(m,s); R = zeros(m,s);
    rho = 1;mu = 1;xi= 1; step = 1.1;
    rho_bar = rho * 1e4;mu_bar = mu * 1e6;xi_bar = xi * 1e6;
    
    % compute \Gamma and J
    %GApre = Wpre;
    while 1
        [UU SS VV] = svd(Wpre, 'econ');
        diagS = diag(SS);
        svp = length(find(diagS > lambdaR/xi));
        if svp < 1
            %due to the zero matrix
            if xi == xi_bar
                disp('error, mu is the max, svp is zero')
                Jpre = zeros(m,s);
                break;
            end
            xi = min(xi*step, xi_bar);
            continue;
        end
        sv = s;
        if svp < sv
            sv = min(svp + 1, s);
        else
            sv = min(svp + round(0.05*s), s);
        end
        Jpre = UU(:, 1:svp) * diag(diagS(1:svp) - lambdaR/xi) * VV(:, 1:svp)'; 
        break;
    end
    
    for v = 1:s
        Upre{v} = zeros(m,num_feas(v));
    end
    
    Lpri1 = []; Lpri2 = []; Ldual = [];
    
    residual1_list = cell(1,s);
    residual4_list = cell(1,s);
    residual6_list = cell(1,s);
    residual7_list = cell(1,s);
    
    residual2_list = []; residual5_list = []; stop_list = []; Ldual = inf;
    %% alternating direction multpliers of methods and augmented lagrange multipliers
    %U -> beta -> W -> Gamma&J
    iter = 1;
    Betav_iters = cell(1,s);
    Wv_iters = cell(1,s); 
    Uv_iters = cell(1,s);
    while iter<10000   

        % solve U
        C = [];
        spliced_P = [];
        for v = 1:s
            C = [C Wpre(:,v)*Betapre{v}'];
            spliced_P = [spliced_P P{v}];
        end

        spliced_U = zeros(m,all_features);
        for i = 1:m
            x_i = spliced_data(i,:)';
            q = train_label(i,1)*x_i + m*spliced_P(i,:)' + rho*m*C(i,:)';
            tmp = q'*x_i/(rho*m+x_i'*x_i);
            spliced_U(i,:) = 1/(rho*m)*(q - tmp*x_i)';
        end
        
        % solve Beta S_(lambdaS/(s*d*rho))(D/(d*rho))
        sta = 1;
        for v = 1:s
            U{v} = spliced_U(:,sta:sta+num_feas(v)-1);
            sta = sta + num_feas(v);
            D = (rho*U{v} - P{v})'*Wpre(:,v);
            d = Wpre(:,v)'*Wpre(:,v);
            mat = D/(rho*d);
%             par = lambdaS/(all_features*d*rho);
            par = lambdaS/(s*d*rho);
            Beta{v} = zeros(num_feas(v),1);
            for j = 1:num_feas(v)
%                 if mat(j) < - par
%                     Beta{v}(j) = mat(j) + par;
%                 else
                    if mat(j) > par
                        Beta{v}(j) = mat(j) - par;
                    end
%                  end
            end
            Betav_iters{v} = [Betav_iters{v} Beta{v}]; 
            Uv_iters{v} = [Uv_iters{v} U{v}(:)];
        end
       
        % solve W
        A = xi*Jpre + R;
        for v = 1:s
            par = rho*Beta{v}'*Beta{v} + xi;
            Bv = (rho*U{v}-P{v})*Beta{v};
            cond1 = 1/par*(A(:,v)+Bv);
            cond2 = 1/(par+mu)*(A(:,v)+Bv+Q(:,v));
            cond3 = 1/mu*Q(:,v);
            for i = 1:m
                if cond1(i) >= cond3(i) && cond2(i) >= cond3(i)
                    W(i,v) = cond1(i);
                else if cond2(i) < cond3(i) && cond1(i) < cond3(i)
                        W(i,v) = cond2(i);
                    end
                end
            end
        end
        % normalize
        W = W./repmat(sum(W,2),1,s);
        for v = 1:s
            Wv_iters{v} = [Wv_iters{v} W(:,v)];
        end
        
        % solve GA
        GA = W - 1/mu*Q;
        for i = 1:m
            for v = 1:s
                if GA(i,v) < 0
                    GA(i,v) = 0;
                end
            end
        end
        
       % solve J
        while 1
%             if choosvd(s, sv) == 1
%                 %[UU SS VV] = lansvd(W-R/xi, sv, 'L');
%             else
                [UU SS VV] = svd(W-R/xi, 'econ');
%            end
            diagS = diag(SS);
            svp = length(find(diagS > lambdaR/xi));
            if svp < 1
                %due to the zero matrix
                if xi == xi_bar
                    disp('error, mu is the max, svp is zero')
                    J = zeros(m,s);
                    break;
                end
                xi = min(xi*step, xi_bar);
                continue;
            end
            if svp < sv
                sv = min(svp + 1, s);
            else
                sv = min(svp + round(0.05*s), s);
            end
            J = UU(:, 1:svp) * diag(diagS(1:svp) - lambdaR/xi) * VV(:, 1:svp)';
            break;
        end
%%        
        % update lagrange multipliers P Q R
        for v = 1:s
            Ppre{v} = P{v};
            P{v} = P{v} + rho*(-U{v} + W(:,v)*Beta{v}');
            for i = 1:m
                if Q(i,v) - mu*W(i,v) <= 0
                    Q(i,v) = 0;
                else
                    Q(i,v) = Q(i,v) - mu*W(i,v);
                end
            end
        end      
        R = R+xi*(J-W);%Q = Q+mu*(GA-W);      
    
        % the function value varies
        [values pri_terms1] = computeLpri1(Beta,W,lambdaR,lambdaS,train_label,train_data);
        Lpri1 = [Lpri1 values];
        [values pri_terms2] = computeLpri2(Beta,U,J,lambdaR,lambdaS,train_label,train_data);
        Lpri2 = [Lpri2 values];
        [values dual_terms] = computeLdual(pri_terms2,Beta,W,U,J,P,Q,R,rho,mu,xi);
        Ldual = [Ldual values];
        
        % stop conditions
        stop_i = zeros(1,3*s+2);p=0;
        stop = 0;
        err_rel = 1e-1; err_abs = 1e-4;
        for v = 1:s    
            p=p+1;
            residual1{v} = norm(U{v}-W(:,v)*Beta{v}','fro'); 
            err_pri = sqrt(m*num_feas(v))*err_abs + err_rel * max([norm(U{v},'fro'),norm(W(:,v)*Beta{v}','fro')]);
            if residual1{v} <= err_pri
                stop = stop + 1;
                stop_i(p) = 1;
            end     
            residual1_list{v} = [residual1_list{v}; residual1{v}];
        end
        
        p=p+1;
        residual2 = norm(W-J);
        err_pri = sqrt(m*s)*err_abs + err_rel*max([norm(W,'fro'),norm(J,'fro')]);
        if residual2 <= err_pri
            stop = stop + 1;
            stop_i(p) = 1;
        end
        residual2_list = [residual2_list; residual2];
        
        mat = W-Wpre;
        for v = 1:s
            p=p+1;
            residual4{v} = norm((Ppre{v}+rho*Wpre(:,v)*Beta{v}')'*mat(:,v));
            err_dual = sqrt(num_feas(v))*err_abs + err_rel*norm(P{v}'*W(:,v));
            if residual4{v} <= err_dual
                stop = stop + 1;
                stop_i(p) = 1;
            end
            residual4_list{v} = [residual4_list{v}; residual4{v}];
         end
      

        mat = J-Jpre;
        p=p+1;
        residual5 = norm(-xi*mat,'fro');
        err_dual = sqrt(m*s)*err_abs + err_rel*norm(Q+R);
        if residual5 <= err_dual
            stop = stop + 1;
            stop_i(p) = 1;
        end
        residual5_list = [residual5_list; residual5];
        
        for v = 1:s
            p=p+1;
            residual6{v} = norm(-rho*(W(:,v)*Beta{v}'-Wpre(:,v)*Betapre{v}'),'fro');
            err_dual = sqrt(m*num_feas(v))*err_abs + err_rel*norm(P{v},'fro');
            if residual6{v} <= err_dual
                stop = stop + 1;
                stop_i(p) = 1;
            end
            residual6_list{v} = [residual6_list{v}; residual6{v}];
        end
        stop_list = [stop_list;stop_i];
        
        for v = 1:s
            residual7{v} = norm(U{v}-Upre{v},'fro'); 
            residual7_list{v} = [residual7_list{v}; residual7{v}];
        end
        % terminal
        if stop == 2 + 3*s || iter == 200
            break;
        end        
        if abs(Ldual(1,end)-Ldual(1,end-1)) < tol
            break;
        end
        
        % update rho,mu,xi
        if abs(dual_terms(5))/rho > 1e-5
            rho = min(rho*step, rho_bar);
        end
 
        mu = min(mu*step, mu_bar);

        xi = min(xi*step, xi_bar);  
        
        % variable updating
        Upre = U;
        Betapre = Beta;
        Wpre = W;
        %GApre = GA;
        Jpre = J;
        
        disp(['Iteration: ' num2str(iter) '; dual function value : ' num2str(abs(Ldual(1,end)-Ldual(1,end-1))) ';']);
        iter = iter + 1;
    end
end
