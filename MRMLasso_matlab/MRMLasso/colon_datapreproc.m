function train_test_data=colon_datapreproc(path,train_frac,view_fea_idx)

org_data=load(strcat(path,'colon-cancer'));
labels=org_data(:,1);
m_index=find(labels==-1);
b_index=find(labels==1);

for i=1:2000
    data(:,i)=org_data(:,i*2+1);
end

% construct mult-view
s=length(view_fea_idx);
for v=1:s
    view{v}=data(:,view_fea_idx{v});
end

% partition training and testing data randomly
b_trainnum=floor(length(b_index)*train_frac);
tmp=randperm(length(b_index));
train_normal_index=b_index(tmp(1:b_trainnum));
test_normal_index=b_index(tmp(b_trainnum+1:end));

m_trainnum=floor(length(m_index)*train_frac);
tmp=randperm(length(m_index));
train_abnormal_index=m_index(tmp(1:m_trainnum));
test_abnormal_index=m_index(tmp(m_trainnum+1:end));

idx=[train_normal_index; train_abnormal_index];
tmp=randperm(length(idx));
train_index=idx(tmp);

idx=[test_normal_index; test_abnormal_index];
tmp=randperm(length(idx));
test_index=idx(tmp);

for v=1:s
    train_test_data.train_data{v}=view{v}(train_index,:);
    train_test_data.test_data{v}=view{v}(test_index,:);
    train_test_data.train_label=labels(train_index);
    train_test_data.test_label=labels(test_index);
end