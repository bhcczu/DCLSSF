%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�������ܣ�����KNNͼ����ȷ��W����
%����˵����
%data������
%sigma: ��˹�˺������
%k:������
%�����
%Ȩֵ����W


function [w, KernelMat] = ConstructKNNGraph(data, sigma, k)

sample_total=size(data,1);           %������������

%��������sample_total*sample_total��
distance_matrix = dist_mat(data,data);
KernelMat = exp(-distance_matrix/(2*sigma^2)); KernelMat=(KernelMat+KernelMat')/2;

distance_matrix = distance_matrix + diag(inf(1,sample_total));


%����ھ���k*k��
neighbor_matrix=zeros(sample_total,k);                        %�˾���洢������������������
for i=1:sample_total
    
    [~,index]=sort(distance_matrix(i,:),'ascend');
    neighbor_matrix(i,:)=index(1:k);
    
end

%Ȩ�ؾ���
w = zeros(sample_total,sample_total);

for i = 1:sample_total
    
    for n = 1:k
        Euclidean_distance = distance_matrix(i,neighbor_matrix(i,n));
        w(i,neighbor_matrix(i,n)) = exp(-Euclidean_distance/(2*sigma^2));    
        w(neighbor_matrix(i,n), i) = w(i,neighbor_matrix(i,n));
    end
    
end

w=(w+w')/2;


