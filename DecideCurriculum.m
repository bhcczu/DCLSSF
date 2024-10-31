
function [Curriculum,ReconError1,ReconError2,ReconError3] = DecideCurriculum(LabeledIndex,UnlabeledIndex,Y,LearningEval,Alpha,...
                KNNMat1,KNNMat2,KNNMat3,KernelMat1,KernelMat2,KernelMat3,...           
                InvKernelLabeledMat1,InvKernelLabeledMat2,InvKernelLabeledMat3,...
                CT1,CT2,CT3)

% This function decides the curriculum for current learning round
% Input:
% LabeledIndex: the indices of labeled examples
% UnlabeledIndex: the indices of unlabeled examples
% Y: binary label matrix, each row represents an example
% LearningEval: Conf(.) in Eq.18
% Alpha: The trade-off parameter in Eq.8
% KNNMat: the n*n matrix encoding the neighborhood information
% KernelMat: Kernel matrix
% InvKernelLabeledMat: inv(KernelMat);
% CT: commute time matrix

% Output:
% Curriculum: the index of decided examples for this loop
% ReconError: ||S^{(v)[t]}-S^{(*)[t]}|| in Eq.16

LabeledTotal = length(LabeledIndex); % the total number of labeled examples
UnlabeledTotal = length(UnlabeledIndex); % the total number of unlabeled examples

%% --------------------------Find candidates of each view based on KNNMat---------%
KNNMat1(LabeledIndex,LabeledIndex) = 0;NNofLabeled = KNNMat1(LabeledIndex,:);CandidateIndex1 = (find(sum(NNofLabeled)~=0))';
KNNMat2(LabeledIndex,LabeledIndex) = 0;NNofLabeled = KNNMat2(LabeledIndex,:);CandidateIndex2 = (find(sum(NNofLabeled)~=0))';
KNNMat3(LabeledIndex,LabeledIndex) = 0;NNofLabeled = KNNMat3(LabeledIndex,:);CandidateIndex3 = (find(sum(NNofLabeled)~=0))';

CandidateIndex = union(union(CandidateIndex1,CandidateIndex2),CandidateIndex3);
CandidateTotal = length(CandidateIndex);

%% Decide the amount of Curriculum points 
CurriculumSize = ceil(CandidateTotal*LearningEval)
%CurriculumSize = min(100,ceil(CandidateTotal*LearningEval));

%% Prepare diagonal commute time matrix M_C
ClassTotal = size(Y, 2);
M1 = ComputeM(ClassTotal,CT1,CandidateIndex,CandidateTotal,Y);M2 = ComputeM(ClassTotal,CT2,CandidateIndex,CandidateTotal,Y);
M3 = ComputeM(ClassTotal,CT3,CandidateIndex,CandidateTotal,Y);

%% --------------Decide Curriculum -----------------%
R1 = ComputeR(KernelMat1,InvKernelLabeledMat1,M1,CandidateIndex,LabeledIndex);
R2 = ComputeR(KernelMat2,InvKernelLabeledMat2,M2,CandidateIndex,LabeledIndex);
R3 = ComputeR(KernelMat3,InvKernelLabeledMat3,M3,CandidateIndex,LabeledIndex);



%% --------------------Find B using optimization on Stiefel manifold-------------------
B1=zeros(CandidateTotal, CurriculumSize);B1(1:CurriculumSize,:)=eye(CurriculumSize);B2=B1;B3=B1;B_Star=B1;
Sigma = 1;% initial value of penalty coefficient 
ObjFunValue = 1e8;
Converge = false; iter=0; tol=1e-4; MaxIter=50;
while ~Converge
    iter=iter+1; 
    
    B1_Old=B1;B2_Old=B2;B3_Old=B3;B_Star_Old=B_Star;ObjFunValue_Old=ObjFunValue;
    
    % Update View 1 to View 3
    B1 = MyONMFOptimizationForB(R1, B1, B_Star, Alpha, Sigma);
    B2 = MyONMFOptimizationForB(R2, B2, B_Star, Alpha, Sigma);
    B3 = MyONMFOptimizationForB(R3, B3, B_Star, Alpha, Sigma);
    

    % Update B_Star
    B_Star = MyONMFOptimizationForBStar(B1,B2,B3,B_Star,Sigma); 
    
    % Compute the value of objective function
    [ObjFunValue,ReconError1,ReconError2,ReconError3]=ComputeObjFunValue(B1,B2,B3,B_Star,R1,R2,R3,Alpha);
        
    % Check termination
    ObjFunValueDiff(iter)=ObjFunValue_Old-ObjFunValue; 
    if (ObjFunValueDiff(iter)/ObjFunValue_Old<=tol) || (iter==MaxIter)
%     if  (norm(B_Star-B_Star_Old)/norm(B_Star_Old)<=tol) || (iter==MaxIter)
        Converge  = true;
        B_Star_Optimal = B_Star;
    end   
     
    % display details
%     disp(['IterTime ' num2str(iter) ...
%           ' ObjFunValue ' num2str(ObjFunValue)...   
%           ' ObjFunValueDiff/ObjFunValue_Old ' num2str(ObjFunValueDiff(iter)/ObjFunValue_Old)]);
end

%% Discretization
Curriculum = zeros(CurriculumSize, 1);
for i = 1:CurriculumSize
    [m, ind] = max(B_Star_Optimal(:));
    if m==0
        break;
    end
    [r,c] = ind2sub([CandidateTotal, CurriculumSize],ind);
    Curriculum(i) = CandidateIndex(r);
    B_Star_Optimal(r,:) = -10000;
end
Curriculum = sort(Curriculum,'ascend');
Curriculum(Curriculum==0) = [];


%% -------------------------Decide curriculum -----------------------%
if UnlabeledTotal == 1
    Curriculum = UnlabeledIndex;
end
disp(['CurriculumSize= ',num2str(size(Curriculum))]);
end


function M = ComputeM(ClassTotal, CT, CandidateIndex, CandidateTotal,Y)
for i = 1:ClassTotal
    AverageCTtoClass(:,i) = mean(CT(CandidateIndex, Y(:,i)==1),2);
end

AverageCTtoClass_Sort = sort(AverageCTtoClass,2,'descend');
M = sparse(1:CandidateTotal, 1:CandidateTotal,1./(AverageCTtoClass_Sort(:,end-1)-AverageCTtoClass_Sort(:,end)));

end

function R = ComputeR(KernelMat,Inv_KernelMat_LL,M,CandidateIndex,LabeledIndex)

KernelMat_CC = KernelMat(CandidateIndex, CandidateIndex);
KernelMat_CL = KernelMat(CandidateIndex, LabeledIndex);
KernelMat_LC = KernelMat(LabeledIndex, CandidateIndex);
% KernelMat_LL = KernelMat(LabeledIndex, LabeledIndex);
%简化为矩阵，论文公式4-8
R = KernelMat_CC-KernelMat_CL*Inv_KernelMat_LL*KernelMat_LC + M; R=(R+R')/2;
%R = KernelMat_CC-KernelMat_CL*Inv_KernelMat_LL*KernelMat_LC ; R=(R+R')/2;
R(R<0)=0;
end

function [ObjFunValue,ReconError1,ReconError2,ReconError3]=ComputeObjFunValue(B1,B2,B3,B_Star,R1,R2,R3,Alpha)
ReconError1=sum(dot(B_Star-B1,B_Star-B1,1));ReconError2=sum(dot(B_Star-B2,B_Star-B2,1));
ReconError3=sum(dot(B_Star-B3,B_Star-B3,1));

G1=ReconError1+ReconError2+ReconError3; 
G2=sum(dot(R1*B1,B1,1)+dot(R2*B2,B2,1)+dot(R3*B3,B3,1));

ObjFunValue = Alpha*G1 + G2;

end

