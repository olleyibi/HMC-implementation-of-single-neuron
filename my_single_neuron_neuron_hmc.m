clc;
clear all;

load A3.dat
X = [ones(8,1) A3(:,1:2)]; 
t = A3(:,3); 
w = [-17.5 1 2];
%w = [-3 1 2];
alpha = 0.1;
epsilon = 0.1;
% Plot data figure(1); 
clf 
plot(X(1:4,2),X(1:4,3),'ks'); hold on 
plot(X(5:8,2),X(5:8,3),'k*') 
xlim([0 10]); ylim([0 10]); axis square; 
xlabel('x1'); ylabel('x2')

% HMC implementation of bayesian inference
[g,y] = gradM ( w,X,t,alpha ) ; % set gradient using initial w
M = findM ( w,t,y,alpha ) ; % set objective function too


lag = 200;
burnin = 10000;
L = burnin + 30*lag; 
Tau = 150;
w_stored = zeros(L,3);
w_stored(1,:) = w;
accepted = 0;
for l = 1:L % loop L times
    p = randn ( size(w) ) ; % initial momentum is Normal(0,1)
    H = p' * p / 2 + M ; % evaluate H(w,p)
    wnew = w ; gnew = g ;
    for tau = 1:Tau
        p = p - epsilon * gnew / 2 ; % make half-step in p
        wnew = wnew + epsilon * p ; % make step in w
        gnew = gradM ( wnew,X,y,alpha ) ; % find new gradient
        p = p - epsilon * gnew / 2 ; % make half-step in p
    end
    Mnew = findM ( wnew,t,y,alpha ) ; % find new objective function
    Hnew = p' * p / 2 + Mnew ; % evaluate new value of H
    dH = Hnew - H ; % decide whether to accept
    if ( dH < 0 ) accept = 1 ;
    elseif ( rand() < exp(-dH) ) accept = 1 ; % compare with a uniform
    else accept = 0 ; % variate
    end
    if ( accept ) g = gnew ; w = wnew ; M = Mnew ; 
    end
    accepted = accepted + accept;
    w_stored(l+1,:)=w;
end



% Plot learned function figure(1); 
hold on 
learned_y = @(x) sigmf(w*x',[1 0]); 
x1 = linspace(0,10); 
x2 = x1; 
[x1 x2] = meshgrid(x1, x2); 
learned_y_cont = reshape(learned_y([ones(10000,1) x1(:), x2(:)]), 100, 100); 
contour(x1, x2, learned_y_cont, [0.27 0.73],'--k'); 
hold on 
contour(x1, x2, learned_y_cont, [0.5 0.5], 'k')


function [gM,y] = gradM ( w,x,t,alpha ) % gradient of objective function
a = x * w' ; % compute activations
y = sigmf(a,[1 0]) ; % compute outputs
e = t - y ; % compute errors
g = - x' * e ; % compute the gradient of G(w)
gM = alpha * w + g ;
end


function M = findM ( w,t,y,alpha ) % objective function
G = - (t' * log(y) + (1-t') * log( 1-y )) ;
EW = w' * w / 2 ;
M = G + alpha * EW ;
end