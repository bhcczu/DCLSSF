
function LearningEval = ComputeLearningFeedback(F_Curriculum,Gamma)
%   Compute learning feedback via Eq.17

% Entropy of labels
[CurriculumTotal,ClassTotal] = size(F_Curriculum);
SumEntropy = zeros(CurriculumTotal,1);
for i = 1:ClassTotal
    SumEntropy = SumEntropy-F_Curriculum(:,i).*(log(F_Curriculum(:,i))/log(ClassTotal));
    %%ybh F_Curriculum(:,i)=0导致log(F_Curriculum(:,i)）有可能为-inf 以至
    %%SumEntropy=NAL  所以添加下面这句话，让其为0
    SumEntropy(isnan(SumEntropy))=0;
end
SumEntropy(isnan(SumEntropy))=0;
AverageEntropy = mean(SumEntropy);
LearningEval = exp(-Gamma*AverageEntropy);