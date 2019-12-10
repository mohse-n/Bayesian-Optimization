clear;
L = 1;

%% Plot sample functions from the prior.
% Sample points.
xs = (-5:0.2:5)';
% Number of sample points.
ns = length(xs);
% Covariance matrix. 
cov =  Kfn(xs, xs);
mu = muFn(xs);

Xnew = -5 + (5+5)*rand;

for iter = 1:5
  
    
iter
figure

%% Generate observation (training) data.

if iter ~= 1
    ei = expectedImprovement(xs, mu, cov);
    [a,b] = min(ei);
    Xnew = xs(b)
end

if iter == 1
    Xtrain = Xnew;
else
    Xtrain(end+1,1) = Xnew; 
end
ftrain = sin(Xtrain);
%% Obtain the posterior, given the observations.
[postMu, postCov] = computePosterior(xs, Xtrain, ftrain);

%% Evaluate expected improvement at sample points
%ei = expectedImprovement(ftrain, postMu, postCov)

%% Plot the posterior Gaussian process with two standard deviation bounds.
figure; hold on
mu = postMu(:);
S2 = diag(postCov);
f = [mu+2*sqrt(S2);flip(mu-2*sqrt(S2),1)];
fill([xs; flip(xs,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);

%% Sample the posterior and plot the sample functions.
for i=1:3
  fs = sampleGuassianProcess(postMu, postCov);
  plot(xs, fs, 'k-', 'linewidth', 2)
  hold on
end

% Plot the mean.
plot(xs, mu, 'r', 'LineWidth', 2)

mu = postMu;
cov = postCov;
 

end

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

function ei = expectedImprovement(xs, mu, cov)
% Returns the value of expected imrovment function at the sample points.
t = min(mu);
imp = mu - t;
Z = imp ./ diag(cov);
ei = imp .* cdf('Normal',Z,0,1) + diag(cov) .* pdf('Normal',Z,0,1);

end
