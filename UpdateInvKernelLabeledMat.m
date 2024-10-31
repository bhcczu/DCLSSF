function InvKernelLabeledMat = UpdateInvKernelLabeledMat(InvKernelLabeledMat, KernelMat, LabeledIndex, Curriculum)

LabeledTotal = length(LabeledIndex);
CurriculumTotal = length(Curriculum);
UpdatedLabeledTotal = LabeledTotal+CurriculumTotal;

%% Blockwise inversion update
B = KernelMat(LabeledIndex,Curriculum);
C = KernelMat(Curriculum,LabeledIndex);
D = KernelMat(Curriculum,Curriculum);
invA = InvKernelLabeledMat;

tempInverse = (D-C*invA*B)\eye(length(Curriculum));

InvKernelLabeledMat_11 = invA+invA*B*tempInverse*C*invA;
InvKernelLabeledMat_12 = -invA*B*tempInverse;
InvKernelLabeledMat_21 = InvKernelLabeledMat_12';
InvKernelLabeledMat_22 = tempInverse;

%InvKernelLabeledMat = [InvKernelLabeledMat_11 InvKernelLabeledMat_12;InvKernelLabeledMat_21 InvKernelLabeledMat_22];
InvKernelLabeledMat = zeros(UpdatedLabeledTotal, UpdatedLabeledTotal);
InvKernelLabeledMat(1:LabeledTotal,1:LabeledTotal) = InvKernelLabeledMat_11;
InvKernelLabeledMat(1:LabeledTotal,LabeledTotal+1:end) = InvKernelLabeledMat_12;
InvKernelLabeledMat(LabeledTotal+1:end,1:LabeledTotal) = InvKernelLabeledMat_21;
InvKernelLabeledMat(LabeledTotal+1:end,LabeledTotal+1:end) = InvKernelLabeledMat_22;

InvKernelLabeledMat = 0.5*(InvKernelLabeledMat+InvKernelLabeledMat');

%% permutation
PermutedIndex = [LabeledIndex; Curriculum];
[OrderedIndex, II]=sort(PermutedIndex,'ascend');

PermutationMatTranspose = zeros(UpdatedLabeledTotal,UpdatedLabeledTotal);
ElementOneIndex=sub2ind([UpdatedLabeledTotal,UpdatedLabeledTotal],(1:UpdatedLabeledTotal)',II);
PermutationMatTranspose(ElementOneIndex)=1;

InvKernelLabeledMat = PermutationMatTranspose*InvKernelLabeledMat*PermutationMatTranspose';


