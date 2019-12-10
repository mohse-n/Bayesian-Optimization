clear;
close all;
L = 1;

%% Plot sample functions from the prior.
% Sample points.
xs = (-5:0.2:5)';
% Covariance matrix. 
cov =  Kfn(xs, xs);
mu = muFn(xs);
Xiknow = [0 1 2 3 4 5];
ftrain = 0

for iter = 1:5
  
iter
    %% Obtain the new evaluation point.
    figure; hold on
    if iter ~= 1
        % Obtain the EI funtion at sample points. Only for visualization.
        % It wouldn't be practical to do this in higher dimensional problems.
        ei = expectedImprovement(xs, Xtrain, ftrain, mu, cov);
        % Optimization the acquision function:
        % Now that we have the  acquisitionfunction evaluated on a grid, do a grid search
        % to find the maximum of EI. Normally, in higher dimensional problems,
        % we would optimize it using a more efficient method.
        [max_val,max_index] = max(ei);
        % Our new point for evaluation is one at which EI is maximum.
        Xnew = xs(max_index)
        Xtrain(end+1,1) = Xiknow(iter);
        % Plot the expected improvement function.
        plot(xs,ei)
    else
        % First iteration, evaluate a random point in the domain (or 
        % use an existing desing.)
        Xnew = 0;
        Xtrain = Xnew;
    end
    
    %% Evaluate the function at the new point.
    % Noiseless observation. Not realistic IRL because give a set
    % of design parameters, our evaluation is never exactly the same
    % as the "true" underlying function.
    ftrain = Xtrain.^2 .* sin(Xtrain)
    plot(Xtrain, ftrain, 'ok')
    %% Obtain the posterior, given the observations.
    
    [postMu, postCov] = computePosterior(xs, Xtrain, ftrain);
    
    %% Various plots for visualization purposes only.
    
    % Plot the posterior Gaussian process with two standard deviation bounds.
    
    mu = postMu(:);
    S2 = diag(postCov);
    f = [mu+2*sqrt(S2);flip(mu-2*sqrt(S2),1)];
    fill([xs; flip(xs,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);

    % Sample the posterior and plot the sample functions.
    for i=1:3
        fs = sampleGuassianProcess(postMu, postCov);
        % Uncomment to plot a 3 sample functions.
        %plot(xs, fs, 'k-', 'linewidth', 2)
        hold on
    end
    
    % Plot the mean.
    plot(xs, mu, 'r', 'LineWidth', 2)
    
    %% The posterior in current iteration will be the the prior in the next.
   
    mu = postMu;
    cov = postCov;
    
end

%% Function definitions.

% Sample points will have a mean of 0
function mu = muFn(x)
    mu = 0*x(:).^2;
end

% Kernel function for defining a Covariance matrix
function cov = Kfn(x,z)
    L = 1;
    cov = 1*exp(-pdist2(x/L,z/L).^2/2);
end

function fs = sampleGuassianProcess(mu, sigma)
% This function samples the Gaussian process once.
% Inputs: Mean (mu) and covariance matrix (sigma) of samples.
% Output: Vector containing value of the (sampled) function at each sample.

    % Number of samples.
    n = length(mu);
    % chol is senstivie to poorly conditioned matrices which sigma is often is.
    % Add small number to diagonal elements to improve condition number. 
    sigma = sigma + 1e-15*eye(n);
    % Obtain the cholesky matrix.
    A = chol(sigma, 'lower');
    Z = randn(n, 1);
    fs = bsxfun(@plus, mu(:), A*Z)';
end

function ei = expectedImprovement(xs, Xtrain, ftrain, mu, cov)
% Returns the value of expected imrovment function at the sample points.
    % Best result yet.
    zeta = 0.01;
    t = min(ftrain);
    imp = mu - t - zeta;
    sigma = diag(cov);
    Z = imp ./ sigma;
    ei = imp .* cdf('Normal',Z,0,1) + sigma .* pdf('Normal',Z,0,1);
    ei(sigma == 0 ) = 0; 
end

function [postMu, postCov] = computePosterior(xs, Xtrain, ftrain)
% Inputs
% xs: sample points.
% Xtrain, ftrain: observation points.
% muFn: Function for obtaining mean at the points. (Which is zero)
% Kfn: Kernel function (gives us the covariance matrices).
% Output
% postMu, postCov: Mean and covariance matrix of sample points in the
% posterior distribution.

    keps = 1e-8;
    % Compute correlation matrices between traning data and previous data.
    K = Kfn(Xtrain, Xtrain); % K
    Ks = Kfn(Xtrain, xs); %K_*
    Kss = Kfn(xs, xs) + keps*eye(length(xs)); % K_** (keps is essential!)
    Ki = inv(K);
    % Mean of the posterior.
    postMu = muFn(xs) + Ks'*Ki*(ftrain - muFn(Xtrain));
    % Covariance of the posterior.
    postCov = Kss - Ks'*Ki*Ks;


end
