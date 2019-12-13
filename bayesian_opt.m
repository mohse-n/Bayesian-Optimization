clear;
close all;

%% Plot sample functions from the prior.
% Sample points.
x_grid = (-1:0.05:2)';
% Initial covariance matrix and mean (the very first posterior). 
cov = kFn(x_grid, x_grid);
mu = muFn(x_grid);
% Value of the objective function.
f_real = objFunction(x_grid);

for iter = 1:6
  
    %% Obtain the new evaluation point.
   
    if iter ~= 1
        % Obtain the EI funtion at sample points. Only for visualization.
        % It wouldn't be practical to do this in higher dimensional problems.
        ei = expectedImprovement(f_observe, mu, cov);
        % Optimization the acquision function:
        % Now that we have the  acquisitionfunction evaluated on a grid, do a grid search
        % to find the maximum of EI. Normally, in higher dimensional problems,
        % we would optimize it using a more efficient method.
        [max_val,max_index] = max(ei);
        % Our new point for evaluation is one at which EI is maximum.
        new_observe = x_grid(max_index);
        x_observe(end+1,1) = new_observe;
    else
        % First iteration, evaluate random points in the domain (or 
        % use an existing designs.)
        new_observe = [-0.9; 1.1];
        x_observe = new_observe;
    end

    %% Evaluate the function at the new point.
    % Noiseless observation. Not realistic IRL because give a set
    % of design parameters, our evaluation is never exactly the same
    % as the "true" underlying function.
    f_observe = objFunction(x_observe);
    % Plot the vertical line, showing the next observation point.
    y = -3:0.05:2;
    w = plot(x_observe(end)*ones(length(y),1),y ,'g', 'LineWidth', 2);
  
    %% Obtain the posterior, given the observations.
    
    [postMu, postCov] = computePosterior(x_grid, x_observe, f_observe);
    
    %% Various plots for visualization purposes only.
    
    % Plot the posterior Gaussian process with two standard deviation bounds.
   
    fig = figure;
    hold on;
    grid on;
    set(fig, 'Position', [500 0 1000 400])
    set(fig,'Color',[1 1 1]);
    xlabel('x');
    ylabel('y');
    figTitle = strcat("Bayesian optimization, iteration ", int2str(iter));
    title(figTitle);
    set(gca,'FontName','Cambria','FontSize',14);
   
    sigma = sqrt(diag(postCov));
    f_plus = postMu + 2*sigma;
    f_minus = postMu - 2*sigma;
    f = [f_plus, f_minus];
    p = plot(x_grid, f, '--k', 'LineWidth', 2);
    q = plot(x_grid, postMu, 'r', 'LineWidth', 2);
    r = plot(x_grid, f_real, 'b');
  
    legend([p(1) p(2) q r], "Mean + 2 stddev", "Mean - 2 stddev", "Mean", "Objective function");
    
    %% The posterior in current iteration will be the the prior in the next.
   
    mu = postMu;
    cov = postCov;
    
end

%% Function definitions.

% In Gaussian processes, usually mu = 0;
function mu = muFn(x)
    mu = 0*x(:).^2;
end

% Kernel function for defining a covariance matrix
function cov = kFn(x,z)
    % L: some type of "length distance". Lower L: Sample function are more jaggedy.
    % higher L: Sample functions are smoother.
    L = 1;
    cov = 1*exp(-pdist2(x/L,z/L).^2/2);
end
% The function we're trying to optimize/approximate.
function f = objFunction(x)

    f = -sin(3*x) - x.^2 + 0.7*x;

end

function fs = sampleGuassianProcess(mu, sigma)
% This function samples the Gaussian process once.
% Inputs: Mean (mu) and covariance matrix (sigma) of samples.
% Output: Vector containing value of the (sampled) function at each sample.

    % Number of samples.
    n = length(mu);
    % chol is senstivie to poorly conditioned matrices which sigma often is.
    % Add small number to diagonal elements to improve condition number. 
    sigma = sigma + 1e-15*eye(n);
    % Obtain the cholesky matrix.
    A = chol(sigma, 'lower');
    Z = randn(n, 1);
    fs = mu(:) +  A*Z;
end

function ei = expectedImprovement(f_observe, mu, cov)
% Returns the value of expected improvement function at the sample points.
    
    % The best (smallest) observation yet.
    t = min(f_observe);
    imp = mu - t;
    sigma = sqrt(diag(cov));
    u = imp ./ sigma;
    ei = imp .* cdf('Normal',u,0,1) + sigma .* pdf('Normal',u,0,1);
    ei(sigma == 0) = 0; 
end

function [postMu, postCov] = computePosterior(x_grid, x_observe, f_observe)
% Inputs
% x_grid: sample points.
% x_observe, f_observe: observation points.
% muFn: Function for obtaining mean at the points. (Which is zero)
% kFn: Kernel function (gives us the covariance matrices).
% Output
% postMu, postCov: Mean and covariance matrix of sample points in the
% posterior distribution.

    keps = 1e-8;
    % Compute correlation matrices between traning data and previous data.
    K = kFn(x_observe, x_observe); 
    Ks = kFn(x_observe, x_grid); 
    % We add a small value to diagonal elements to improve the condition
    % number.
    Kss = kFn(x_grid, x_grid) + keps*eye(length(x_grid)); 
    Ki = inv(K);
    % Mean of the posterior.
    postMu = muFn(x_grid) + Ks'*Ki*(f_observe - muFn(x_observe));
    % Covariance of the posterior.
    postCov = Kss - Ks'*Ki*Ks;
end
