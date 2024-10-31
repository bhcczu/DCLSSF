function [B,Fvalue,iter] = MyONMFOptimizationForB(R, B, B_Star, Alpha, Sigma)

% User Settings

C = 1.2;             %Sigma  = Sigma*C
minOpts.maxMinIters = 1;
lsOpts.stepBeta = 1.1;
lsOpts.stepLowerThreshold = 1e-15;

SigmaUpperThreshold = 1e10;
tol = 1e-4; MaxIter = 100;

%==========================================================================
Lembda = zeros(size(B));
stepB = 1;

converged = false;
iter = 0;

T = max(0, B+Lembda/Sigma);
while ~converged
       
    iter = iter + 1;
    
    %% update B
    [B_New,stepB,Fvalue] = updateB(B,B_Star,R,T,Lembda,Alpha,Sigma,stepB,minOpts,lsOpts);
        
    %% stop Criterion
    error(iter) = norm(B_New-B,'fro'); 
    if ((error(iter)/norm(B,'fro'))<tol) || (iter == MaxIter)  %
        converged = true;
    end
    
        
    %% update parameters
    Lembda = max(0, Lembda-Sigma*B_New);
    
    Sigma = min(Sigma*C,SigmaUpperThreshold);
    T = max(0, B_New+Lembda/Sigma);
    B = B_New;               
    
end


%==========================================================================
function B = closestOrthogonalMatrix(A)

% computes B s.t. A = B*P, B'*B = I
P = sqrtm(A'*A);
B = A/P;


%==========================================================================
function F = ComputeLagrangianObjFunValue(B,B_Star,R,T,Alpha,Sigma,Lembda)
BMinusT = B-T;
G1=2*R*B; G2=Lembda; G3=Sigma*BMinusT; G4=Alpha*(B-B_Star);
%G = G1+G2+G3;
% F = trace(X'*R*X + Lembda'*(X-T) + Sigma*((X-T)'*(X-T))/2);
F = sum(0.5*(dot(G1,B,1))+dot(G2,BMinusT,1)+0.5*(dot(G3,BMinusT,1))+dot(G4,B-B_Star,1));



%==========================================================================
function [B,step,lagrValue] = updateB(B,B_Star,R,T,Lembda,Alpha,Sigma,step,minOpts,lsOpts)

agrad = @(B)(Sigma)*(B-T)+Lembda+(2*R*B)+2*Alpha*(B-B_Star); %Compute the gradient

stepMove = @(B,step)stepMoveB(B,step,agrad);
ObjFunValue = @(B)ComputeLagrangianObjFunValue(B,B_Star,R,T,Alpha,Sigma,Lembda);

[B,step,lagrValue] = minimizeFun(B,step,stepMove,ObjFunValue,minOpts,lsOpts);

%==========================================================================
function B_new = stepMoveB(B,step,agrad)

B_euc = B - step*agrad(B);
B_new = closestOrthogonalMatrix(B_euc);

%==========================================================================
function [x,step,lagrValue] = minimizeFun(x,step,stepMove,ObjFunValue,minOpts,lsOpts)

tolFun = 1e-4;

diffFun = +Inf; % difference in subsequent function values
iter = 1;
while diffFun > tolFun && iter <= minOpts.maxMinIters    
    [x,step,lagrValue,lagrVal0] = lineSearch(x,step,stepMove,ObjFunValue,lsOpts);    
    diffFun = abs(lagrVal0 - lagrValue);
    iter = iter + 1;
end % outer iter

%==========================================================================
function [x,step,lagrValue,startLagrangValue] = lineSearch(x,step,stepMove,computeLagrangianValueForB,lsOpts)

startLagrangValue = computeLagrangianValueForB(x);
lastLagrangVal = startLagrangValue;
isStepAccepted = 0;
j = 1;

while ~isStepAccepted && step > lsOpts.stepLowerThreshold
    x_new = stepMove(x,step);
    lagrVal_candidate = computeLagrangianValueForB(x_new);
    hasImproved = lagrVal_candidate < startLagrangValue && lagrVal_candidate < lastLagrangVal;
    if j == 1
       keepIncreasing = hasImproved;           
       last_x = x_new;
    end
    if keepIncreasing
        if hasImproved
            last_x = x_new;
            lastLagrangVal = lagrVal_candidate;
            step = lsOpts.stepBeta*step;
        else                
            step = step/lsOpts.stepBeta;
            x = last_x;
            isStepAccepted = 1;
        end                
    else
        if hasImproved
            lastLagrangVal = lagrVal_candidate;
            x = x_new;
            isStepAccepted = 1;
        else
            step = step/lsOpts.stepBeta;
        end
    end
    j = j + 1;
end % end line search
lagrValue = lastLagrangVal;

%==========================================================================

