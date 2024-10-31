
function [CT, L_Inverse] = ComputeCommuteTimeMatrix(L)
% Compute commute time matrix

n = size(L,1);   CT = zeros(size(L));


L_Inverse = pinv(L);

for i = 1:n
    for j =i+1:n
        CT(i,j) = L_Inverse(i,i) + L_Inverse(j,j)- 2*L_Inverse(i,j);
        CT(j,i) = CT(i,j);
    end
end

