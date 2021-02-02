clc;close all; clear all;

train_frac=0.75;
load('colon-cancer/view_fea_idx.mat','view_fea_idx');
    
for iter=1:10

    train_test_data=colon_datapreproc('colon-cancer/',train_frac,view_fea_idx); 
    
    %% multi-view data
    train_data=train_test_data.train_data;
    test_data=train_test_data.test_data;
    train_label=train_test_data.train_label;
    test_label=train_test_data.test_label;

    num_test = size(test_label,1);

    proc_train_label = train_label;
    proc_train_label(find(proc_train_label == -1)) = 0;
    %% feature nomalization
    [train_data, test_data] = feat_norm2(train_data, test_data);
    num_class=2; s = 3;
    
    %% multiview feature selection and SVM classifier
    Lasso_para.lambdaS=0.5;
    Lasso_para.lambdaR=100;
    tic
    [Beta,W]=nonconvex_ALM_MRMLasso(train_data,proc_train_label,Lasso_para);
    toc
    gamma=1e-4;

    for v=1:s
        train_mat=train_data{v};
        test_mat=test_data{v};

        beta = Beta{v};
        selected_features{v} = find(beta>1e-4);
                  
        train_mat = train_mat(:,selected_features{v});
        test_mat = test_mat(:,selected_features{v});

        model = svmtrain(train_label, train_mat, ['-c 10 -t 2 -g ' num2str(gamma)]);
        [predict_label, accur, dec_values] = svmpredict(test_label, test_mat, model);
        predict_labels{v}=predict_label;            
    end

    %%% voting AND fusion
    vote_label=voting(predict_labels,num_class);
    fusion_label=fusion(predict_labels,num_class,W);
    %%% evaluation
    accuracy(iter,1)=length(find(vote_label-test_label==0))/num_test;
    accuracy(iter,2) = length(find(fusion_label-test_label==0))/num_test;
    [avgF1(iter,1) f1score]= computeF1_binary(vote_label, test_label);
    neg_f1score(iter,1)=f1score(1);
    pos_f1score(iter,1)=f1score(2);
    [avgF1(iter,2) f1score]= computeF1_binary(fusion_label, test_label);
    neg_f1score(iter,2)=f1score(1);
    pos_f1score(iter,2)=f1score(2);
end
avg_accu=mean(accuracy);
var_accu=var(accuracy);
avg_negf1=mean(neg_f1score);
var_negf1=var(neg_f1score);
avg_posf1=mean(pos_f1score);
var_posf1=var(pos_f1score);
