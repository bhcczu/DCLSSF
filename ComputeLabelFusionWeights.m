function [weight1, weight2, weight3]=ComputeLabelFusionWeights(ReconError1,ReconError2,ReconError3)

% compute fusion weight via Eq.16

Beta  = 1;         

View1=exp(-Beta*ReconError1);
View2=exp(-Beta*ReconError2);
View3=exp(-Beta*ReconError3);


Denominator = View1+View2+View3;

weight1=View1/Denominator;
weight2=View2/Denominator;
weight3=View3/Denominator;
