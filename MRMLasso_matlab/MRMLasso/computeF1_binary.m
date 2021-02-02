function [F1, f1_score]=computeF1_binary(predict_label,test_labels)
f1_score=zeros(2,1);
acc_idx=find(predict_label-test_labels==0);

    label=-1;
    precision=length(find(test_labels(acc_idx)==label))/length(find(predict_label==label));
    recall=length(find(test_labels(acc_idx)==label))/length(find(test_labels==label));
    if (precision+recall~=0)&&~isnan(precision)&&~isnan(recall)
        f1_score(1)=2*precision*recall/(precision+recall);
    end
    
    label=1;
    precision=length(find(test_labels(acc_idx)==label))/length(find(predict_label==label));
    recall=length(find(test_labels(acc_idx)==label))/length(find(test_labels==label));
    if (precision+recall~=0)&&~isnan(precision)&&~isnan(recall)
        f1_score(2)=2*precision*recall/(precision+recall);
    end
    
F1=mean(f1_score);