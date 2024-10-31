%
% Adaptive Edge Weighting for Label Propagation
% 
% INPUT
%  X: d (features) times n (instances) input data matrix
%  param: The structure variable containing the following field:
%    max_iter: The maximum number of iteration for gradient descent
%    k: The number of nearest neighbors
%    sigma: Initial width parameter setting 'median'|'local-scaling'
% OUTPUT
%  W: The optimized weighted adjacency matrix
%  W0: The initial adjacency matrix
% REFERENCE
%  M. Karasuyama and H. Mamitsuka, "Manifold-based similarity 
%  adaptation for label propagation", NIPS 2013.
%
function [W W0] = AEW(X, param)

  [d n] = size(X);

  ex = 0;
  tol = 1e-4;

  % Parameters for line-search
  beta = 0.1;
  beta_p = 0;
  max_beta_p = 8;
  rho = 1e-3;

  [W0 sigma0] = generate_nngraph(X,param.k,param.sigma);
  L = eye(d);

  Xori = X;
  if length(sigma0) > 1
    dist = squareform(pdist(X').^2);
    sigma0 = reshape(sigma0,n,1);
    dist = dist ./ (sigma0 * sigma0');
  else
    X = X ./ (sqrt(2)*sigma0);
    dist = squareform(pdist(X').^2);
  end

  edge_idx = find(W0);
  W = zeros(n);
  W(edge_idx) = exp(-dist(edge_idx));  
  
  Gd = zeros(n,n,d);
  for i = 1:n
    W_idx{i} = find(W(i,:));
    for j = W_idx{i};
      if W(i,j)
        Gd(i,j,:) = -(X(:,i) - X(:,j)).*(X(:,i) - X(:,j));
        if length(sigma0) > 1
          Gd(i,j,:) = Gd(i,j,:) ./ (sigma0(i)*sigma0(j));
        end
      end
    end
  end

  % --------------------------------------------------
  % Gradient Descent
  % --------------------------------------------------
  d_W = zeros(n,n,d);
  d_WDi = zeros(n,n,d);
  for iter = 1:param.max_iter
    D = sum(W);
    for i = 1:n
      d_W(i,W_idx{i},:) = 2*diag(W(i,W_idx{i}))* ...
          (reshape(Gd(i,W_idx{i},:),length(W_idx{i}),d) ...
           .*(ones(length(W_idx{i}),1)*diag(L)'));
    end    

    for i = 1:n
      sum_d_W(i,:) = sum(d_W(i,W_idx{i},:));
      d_WDi(i,W_idx{i},:) = d_W(i,W_idx{i},:)./D(i) - ...
          reshape((W(i,W_idx{i})'./(D(i).^2)) ...
                  *reshape(sum_d_W(i,:),1,d),1,length(W_idx{i}),d);
    end

    Xest = (diag(1./D)*W*Xori')';
    err = (Xori - Xest);
    sqerr = sum(sum(err.^2));

    grad = -(reshape(d_WDi,n^2,d)'*vec(err'*Xori))';
    grad = grad ./ norm(grad); % Normalize

    fprintf('Iter = %d, MSE = %e\n', ...
            iter, sqerr/(d*n));

    step = (beta^beta_p)*1;
    sqerr_prev = sqerr;
    L_prev = L;
    while 1 % Line-search
      L = L_prev - step*diag(grad);
      dist = squareform(pdist((L*X)').^2);
      if length(sigma0) > 1
        dist = dist ./ (sigma0 * sigma0');
      end
      W(edge_idx) = exp(-dist(edge_idx));

      D = sum(W);
      Xest = (diag(1./D)*W*Xori')';
      err = (Xori - Xest);
      sqerr_temp = sum(sum(err.^2));
      if sqerr_temp - sqerr_prev <= -rho*step*grad'*grad
        break; 
      end

      beta_p = beta_p + 1;
      if beta_p > max_beta_p
        ex = 1;
        break;
      end
      step = step * beta;
    end

    if ((sqerr_prev - sqerr_temp) / sqerr_prev) < tol || ex
      break;
    end
    % pause
  end
  
end

function x = vec(X)

%
% This file is part of SeDuMi 1.1 by Imre Polik and Oleksandr Romanko
% Copyright (C) 2005 McMaster University, Hamilton, CANADA  (since 1.1)
%
% Copyright (C) 2001 Jos F. Sturm (up to 1.05R5)
%   Dept. Econometrics & O.R., Tilburg University, the Netherlands.
%   Supported by the Netherlands Organization for Scientific Research (NWO).
%
% Affiliation SeDuMi 1.03 and 1.04Beta (2000):
%   Dept. Quantitative Economics, Maastricht University, the Netherlands.
%
% Affiliations up to SeDuMi 1.02 (AUG1998):
%   CRL, McMaster University, Canada.
%   Supported by the Netherlands Organization for Scientific Research (NWO).
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc.,  51 Franklin Street, Fifth Floor, Boston, MA
% 02110-1301, USA
%

[m n] = size(X);
x = reshape(X,m*n,1);
end
  