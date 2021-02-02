function fusion_label = fusion(predict_label,num_class,Weight)
s=length(predict_label);
num_test=length(predict_label{1});
predict_label_mat=cell2mat(predict_label);
fusion_label=zeros(num_test,1);
b = zeros(s,1);
allsum = sum(Weight(:));
for v=1:s
    b(v,1) = sum(Weight(:,v))/allsum;
end
% if num_class~=2
%     for i=1:num_test
%         vote=zeros(num_class,1);
%         for j=1:num_class
%             for v=1:s
%                 if (predict_label_mat(i,v)==j)
%                     vote(j)=vote(j)+1;
%                 end
%             end
%         end
% %         [maxx idx]=max(vote);
% %         vote_label(i)=idx;
%         maxx=max(vote);
%         [r,c]=find(vote == maxx);
%         randlist=randperm(length(r));
%         vote_label(i)=r(randlist(1));
%     end
% else
if num_class==2
    for i=1:num_test
        vote=zeros(2,1);        
        for v=1:s
            if (predict_label_mat(i,v)==-1)
                vote(1)=vote(1)+b(v);
            end
            if (predict_label_mat(i,v)==1)
                vote(2)=vote(2)+b(v);
            end
        end
        maxx = max(vote);
        [r,c]=find(vote == maxx);
        randlist=randperm(length(r));
        fusion_label(i)=r(randlist(1));
    end
    fusion_label(find(fusion_label==1))=-1;
    fusion_label(find(fusion_label==2))=1;
end