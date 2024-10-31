function  F_New  = LabelPropagation(Curriculum, LabeledIndex, InitialLabeledIndex, P, F)
% Conduct label propagation via Eq.13
% Input:
% Curriculum: Indices of curriculum examples
% LabeledIndex: Indices of currently labeled examples
% InitialLabeledIndex: Indices of initial labeled examples
% P: Iteration matrix 
% F: label matrix

% Output:
% F_New: the updated label matrix


[DataTotal, ClassTotal] = size(F);

%% find soft labels
IndexForCurrentPropgation = [LabeledIndex; Curriculum];
M = zeros(1, DataTotal); M(IndexForCurrentPropgation)=1; M = sparse(1:DataTotal,1:DataTotal,M);

F_New = M*P*F;
F_New(InitialLabeledIndex,:) = F(InitialLabeledIndex,:);  %Clamp the labeled examples
F_New(sum(F_New,2)==0,:) = 1/ClassTotal; %Clamp the labels of unlabeled examples to 1/ClassTotal

F_New = bsxfun(@rdivide,F_New,sum(F_New,2)); F_New(isnan(F_New))=0; % Row normalization










