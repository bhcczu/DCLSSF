GroundTruth=zeros(2100,1);
for i=1:21
    GroundTruth((i-1)*100+1:i*100,:)=i;
end
save GroundTruth.mat GroundTruth;