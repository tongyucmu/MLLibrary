function [ f_h ] = appro_hessian(s, W, c)

global X_train Y_train

tmp = exp(-Y_train(:).*(X_train(:,:)*W'));

D = diag(sparse(tmp./(1+tmp).^2));

f_h = full(s'+ c*((D*(X_train*s'))'*X_train)');

end

