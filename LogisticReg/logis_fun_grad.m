function [fun_value] = logis_fun_grad(W, c)

global X_train Y_train

fun_value = W;
coef = Y_train.*(1./(1+exp(-Y_train(:).*(X_train(:,:)*W')))-1);
coef_list = diag(sparse(coef));

fun_value_list = coef_list*X_train;
%use sparse to aviod OUT OF MEMORY
fun_value2 = sparse(fun_value) + c*sum(fun_value_list);
%finally change to full because the output is not a sparse vector at all
fun_value = full(fun_value2);


% the following is the original implementaion using for loop
% num_ins = size(Y_train, 1);
% for j =1:num_ins
% 	fun_value = fun_value + (-Y_train(j)*X_train(j,:))*exp(-Y_train(j)*W*X_train(j,:)')/(1+exp(-Y_train(j)*W*X_train(j,:)'));; 
% end




