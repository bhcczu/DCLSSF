function [B_Star,Fvalue,iter] = MyONMFOptimizationForBStar(B1,B2,B3,B_Star,Sigma)

% User Settings

C = 1.2;             %Sigma  = Sigma*C
minOpts.maxMinIters = 1;
lsOpts.stepBeta = 1.1;
lsOpts.stepLowerThreshold = 1e-15;

SigmaUpperThreshold = 1e10;
tol = 1e-4; MaxIter = 100;

%==========================================================================
Lembda = zeros(size(B_Star));
stepB = 1;

converged = false;
iter = 0;

T = max(0, B_Star+Lembda/Sigma);
while ~converged
       
    iter = iter + 1;
    
    %% update B
    [B_New,stepB,Fvalue] = updateB(B1,B2,B3,B_Star,Sigma,T,Lembda,stepB,minOpts,lsOpts);
        
    %% stop Criterion
    error(iter) = norm(B_New-B_Star,'fro');
    if (error(iter)/norm(B_Star,'fro')<tol) || (iter == MaxIter)
        converged = true;
    end
         
        
    %% update parameters
    Lembda = max(0, Lembda-Sigma*B_New);
    
    Sigma = min(Sigma*C,SigmaUpperThreshold);
    T = max(0, B_New+Lembda/Sigma);
    B_Star = B_New;               
    
end


%==========================================================================
function B = closestOrthogonalMatrix(A)

% computes B s.t. A = B*P, B'*B = I
P = sqrtm(A'*A);
B = A/P;


%==========================================================================
function F = ComputeLagrangianObjFunValue(B1,B2,B3,B,Sigma,T,Lembda)

BMinusT = B-T;
G1=sum(dot(B-B1,B-B1,1)+dot(B-B2,B-B2,1)+dot(B-B3,B-B3,1)); 
G2=sum(dot(Lembda,BMinusT,1));
G3=0.5*Sigma*sum(dot(BMinusT,BMinusT,1));
F = G1 + G2 + G3;

%==========================================================================
function [B,step,lagrValue] = updateB(B1,B2,B3,B,Sigma,T,Lembda,step,minOpts,lsOpts)

agrad = @(B)(Sigma)*(B-T)+Lembda+2*(3*B-B1-B2-B3); %Compute the gradient
stepMove = @(B,step)stepMoveB(B,step,agrad);
ObjFunValue = @(B)ComputeLagrangianObjFunValue(B1,B2,B3,B,Sigma,T,Lembda);

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

