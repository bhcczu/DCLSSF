%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%函数功能：建立KNN图，即确定W矩阵
%参数说明：
%data：数据
%sigma: 高斯核函数宽度
%k:近邻数
%输出：
%权值矩阵W


function [w, KernelMat] = ConstructKNNGraph(data, sigma, k)

sample_total=size(data,1);           %所有样本总数

%求距离矩阵（sample_total*sample_total）
distance_matrix = dist_mat(data,data);
KernelMat = exp(-distance_matrix/(2*sigma^2)); KernelMat=(KernelMat+KernelMat')/2;

distance_matrix = distance_matrix + diag(inf(1,sample_total));


%求近邻矩阵（k*k）
neighbor_matrix=zeros(sample_total,k);                        %此矩阵存储各样本点的最近邻索引
for i=1:sample_total
    
    [~,index]=sort(distance_matrix(i,:),'ascend');
    neighbor_matrix(i,:)=index(1:k);
    
end

%权重矩阵
w = zeros(sample_total,sample_total);

for i = 1:sample_total
    
    for n = 1:k
        Euclidean_distance = distance_matrix(i,neighbor_matrix(i,n));
        w(i,neighbor_matrix(i,n)) = exp(-Euclidean_distance/(2*sigma^2));    
        w(neighbor_matrix(i,n), i) = w(i,neighbor_matrix(i,n));
    end
    
end

w=(w+w')/2;


