function [ train_data, test_data ] = feat_norm2(train_data, test_data)

s=length(train_data);
  for v = 1 : s
           
      test_mat = test_data{v};       
      train_mat = train_data{v};
      train_num = size(train_data{v},1);
      
      data_mat=[train_mat; test_mat];
      [m dim]=size(data_mat);
      
       %# normalize the mean and variance as zero and one via the instance-wise     
       
      mean_vec=zeros(1,dim);
      var_vec=zeros(1,dim);
      for d=1:dim
          mean_vec(d)=mean(data_mat(:,d));
          var_vec(d)=var(data_mat(:,d));  
      end
      data_mat=data_mat-repmat(mean_vec,m,1);
      var_vec(find(var_vec==0))=1;
      data_mat=data_mat./repmat(var_vec,m,1);
      
      %# normalize the mean and variance as zero and one via the feature-wise
    
      mean_vec=zeros(m,1);
      var_vec=zeros(m,1);
      for i=1:m
          mean_vec(i)=mean(data_mat(i,:));  
          var_vec(i)=var(data_mat(i,:));  
      end
      data_mat=data_mat-repmat(mean_vec,1,dim);
      data_mat=data_mat./repmat(var_vec,1,dim);
      
      train_data{v}=data_mat(1:train_num,:);
      test_data{v}=data_mat(train_num+1:end,:);
      
  end
  
end

