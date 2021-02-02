function vote_label=voting(predict_label,num_class)
s=length(predict_label);
num_test=length(predict_label{1});
predict_label_mat=cell2mat(predict_label);
vote_label=zeros(num_test,1);
if num_class~=2
    for i=1:num_test
        vote=zeros(num_class,1);
        for j=1:num_class
            for v=1:s
                if (predict_label_mat(i,v)==j)
                    vote(j)=vote(j)+1;
                end
            end
        end
        maxx=max(vote);
        [r,c]=find(vote == maxx);
        randlist=randperm(length(r));
        vote_label(i)=r(randlist(1));
    end
else
    for i=1:num_test
        vote=zeros(2,1);        
        for v=1:s
            if (predict_label_mat(i,v)==-1)
                vote(1)=vote(1)+1;
            end
            if (predict_label_mat(i,v)==1)
                vote(2)=vote(2)+1;
            end
        end
        maxx = max(vote);
        [r,c]=find(vote == maxx);
        randlist=randperm(length(r));
        vote_label(i)=r(randlist(1));
    end
    vote_label(find(vote_label==1))=-1;
    vote_label(find(vote_label==2))=1;
end