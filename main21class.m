%% Multi-modal Curriculum Learning


%%
clear all
close all;
clc;

%matlabpool open 4
% poolobj = gcp('nocreate'); % If no pool, do not create new one.
% if isempty(poolobj)
%     poolsize = 0;
% else
%     poolsize = poolobj.NumWorkers
% end

%-----------------------parametric settings---------------------%
Alpha                  = 1;           % The trade-off parameter in Eq.8
Gamma                  = 2;           % Learning rate in Eq.17
Kappa                  = 1.1;         % Gamma := Gamma/Kappa in every propagation
Theta                  = 0.05;        % theta in Eq.20

% Parameters for graph constructino via AEW
param.k = 10; % The number of neighbors in kNN graph
param.sigma = 'median'; % Kernel parameter heuristics 'median' or 'local-scaling'
param.max_iter = 100;



%-----------------------load data--------------------------------%


load('D:\matlabcode\MMCL\21class\GroundTruth.mat');
ClassTotal = max(GroundTruth);

load D:\matlabcode\MMCL\21class\DatasetSplitIdx1
% %===21class===%
% % Load View 1 (densenet201) Features
 load densenet201_UCMERCED_fc.mat; F1=sparse(double(denseFeatures));
% % Load View 2 (inceptionresnetv2) Features
 load inceptionresnetv2_UCMERCED_fc.mat; F2=sparse(double(denseFeatures));
% % Load View 3 (resnet101) Features
 load resnet101_UCMERCED_fc.mat; F3=sparse(double(denseFeatures));
[Ax, Ay, trainXdca12, trainYdca12] = dcaFuse(F1, F2, GroundTruth);
%Feature_1=[trainXdca12 ; trainYdca12]';
Feature_1=[trainXdca12 + trainYdca12]';
[Ax, Ay, trainXdca13, trainYdca13] = dcaFuse(F1, F3, GroundTruth);
%Feature_2=[trainXdca13 ; trainYdca13]';
Feature_2=[trainXdca13 + trainYdca13]';
[Ax, Ay, trainXdca23, trainYdca23] = dcaFuse(F2, F3, GroundTruth);
%Feature_3=[trainXdca23 ; trainYdca23]';
Feature_3=[trainXdca23 + trainYdca23]';
clear Ax Ay trainXdca12 trainXdca13 trainXdca23 trainYdca12 trainYdca13 trainYdca23 F1 F2 F3


tic
DataTotal = size(Feature_1,1);
%% Graph Construction
[W1, ~] = AEW(Feature_1',param);[W2, ~] = AEW(Feature_2',param);[W3, ~] = AEW(Feature_3',param);

% Binary n*n matrix encoding the neighbor information
KNNMat1=zeros(DataTotal,DataTotal);KNNMat2=zeros(DataTotal,DataTotal);KNNMat3=zeros(DataTotal,DataTotal);
KNNMat1(W1~=0)=1; KNNMat2(W2~=0)=1; KNNMat3(W3~=0)=1;

% Compute Laplacian Matrix and Kernel Matrix
D1 = sparse(1:DataTotal, 1:DataTotal,sum(W1)); D2 = sparse(1:DataTotal, 1:DataTotal,sum(W2));
D3 = sparse(1:DataTotal, 1:DataTotal,sum(W3));
L1=D1-W1; L2=D2-W2; L3=D3-W3;
I_DataTotal = eye(DataTotal,DataTotal);
KernelMat1=(L1+0.01*I_DataTotal)\I_DataTotal; KernelMat1=0.5*(KernelMat1+KernelMat1');
KernelMat2=(L2+0.01*I_DataTotal)\I_DataTotal; KernelMat2=0.5*(KernelMat2+KernelMat2');
KernelMat3=(L3+0.01*I_DataTotal)\I_DataTotal; KernelMat3=0.5*(KernelMat3+KernelMat3');

% Compute Transition Matrix
P1 = bsxfun(@rdivide,W1,sum(W1,2)); P1(isnan(P1))=0;
P2 = bsxfun(@rdivide,W2,sum(W2,2)); P2(isnan(P2))=0;
P3 = bsxfun(@rdivide,W3,sum(W3,2)); P3(isnan(P3))=0;

% Compute Cummute Time Matrix
[CT1, ~] = ComputeCommuteTimeMatrix(L1);
[CT2, ~] = ComputeCommuteTimeMatrix(L2);
[CT3, ~] = ComputeCommuteTimeMatrix(L3);

display('Graph Construction Completed!');

%% Y: initially labeled examples
Y = zeros(DataTotal,ClassTotal);
for i=1:length(LabeledIndex)
    Y(LabeledIndex(i),GroundTruth(LabeledIndex(i)))=1;
end


%% Label Propagation via Curruculum Learning
Iteration = 1;  F  = Y; InitialLabeledIndex = LabeledIndex; LearningEval = 0.01;
InvKernelLabeledMat1=KernelMat1(LabeledIndex,LabeledIndex)\eye(length(LabeledIndex)); InvKernelLabeledMat1=(InvKernelLabeledMat1+InvKernelLabeledMat1')/2;
InvKernelLabeledMat2=KernelMat2(LabeledIndex,LabeledIndex)\eye(length(LabeledIndex)); InvKernelLabeledMat2=(InvKernelLabeledMat2+InvKernelLabeledMat2')/2;
InvKernelLabeledMat3=KernelMat3(LabeledIndex,LabeledIndex)\eye(length(LabeledIndex)); InvKernelLabeledMat3=(InvKernelLabeledMat3+InvKernelLabeledMat3')/2;


while 1
    
    %% Decide curriculum for this iteration
    [Curriculum,ReconError1,ReconError2,ReconError3] = DecideCurriculum(LabeledIndex,UnlabeledIndex,Y,LearningEval,Alpha,...
        KNNMat1,KNNMat2,KNNMat3,KernelMat1,KernelMat2,KernelMat3,...
        InvKernelLabeledMat1,InvKernelLabeledMat2,InvKernelLabeledMat3,...
        CT1,CT2,CT3);    
    
    %% Label propagation & fusion
    F1 = LabelPropagation(Curriculum, LabeledIndex, InitialLabeledIndex, P1, F);
    F2 = LabelPropagation(Curriculum, LabeledIndex, InitialLabeledIndex, P2, F);
    F3 = LabelPropagation(Curriculum, LabeledIndex, InitialLabeledIndex, P3, F);
    
    [weight1, weight2, weight3]=ComputeLabelFusionWeights(ReconError1,ReconError2,ReconError3);
    F = weight1*F1+weight2*F2+weight3*F3;
    
    %% Evaluate the learning performance of this learning round
    F_Curriculum = F(Curriculum,:);
    LearningEval = ComputeLearningFeedback(F_Curriculum,Gamma);
  %  LearningEval = 0.1;
    display(['LearningEval = ' num2str(LearningEval)])
    
    %% find Y for next loop
    Y = zeros(DataTotal, ClassTotal); % The labels of learned curriculum
    [~,Y_temp] = max(F,[],2);
    IndexForCurrentPropgation = [LabeledIndex; Curriculum];
    Y_temp2 = Y_temp(IndexForCurrentPropgation);
    Y(sub2ind(size(Y),IndexForCurrentPropgation,Y_temp2)) = 1;
    
    %% Update variables incrementally
    InvKernelLabeledMat1 = UpdateInvKernelLabeledMat(InvKernelLabeledMat1, KernelMat1, LabeledIndex, Curriculum); % update inv_K_LL
    InvKernelLabeledMat2 = UpdateInvKernelLabeledMat(InvKernelLabeledMat2, KernelMat2, LabeledIndex, Curriculum); % update inv_K_LL
    InvKernelLabeledMat3 = UpdateInvKernelLabeledMat(InvKernelLabeledMat3, KernelMat3, LabeledIndex, Curriculum); % update inv_K_LL
    
    Gamma = Gamma/Kappa;
    
    LabeledIndex = [LabeledIndex; Curriculum]; LabeledIndex = sort(LabeledIndex,'ascend');
    display('LabeledIndex')
    if length(LabeledIndex) == DataTotal
        break;
    else        
        AllIndex = (1:DataTotal)';
        AllIndex(LabeledIndex)=[];
        UnlabeledIndex = AllIndex;      % find the index of unlabeled examples
        display(['Iteration = ' num2str(Iteration)])
        Iteration = Iteration + 1;
    end
end

%% Iterate Untill Convergence
F1 = (eye(DataTotal)-Theta*P1)\F; F2 = (eye(DataTotal)-Theta*P2)\F;
F3 = (eye(DataTotal)-Theta*P3)\F;
F  = (F1+F2+F3)/3;

%% Output
[~,Classification] = max(F,[],2);

%% Evaluation
classes = (1:ClassTotal)';
[confus,Accuracy,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (GroundTruth,Classification,classes);
toc

display(['Classification Accuracy = ' num2str(Accuracy)])









