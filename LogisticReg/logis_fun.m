function [fun_value] = logis_fun(W, c)

global X_train Y_train

fun_value = 1/2*W*W';

fun_value = fun_value + c*sum(log(1+exp(-Y_train.*(X_train*W'))));

% the following is the original implementaion using for loop
% num_ins = size(Y_train, 1);
% for j =1:num_ins
% 	fun_value = fun_value + log(1+exp(-Y_train(j)*W*X_train(j,:)')); 
% end
% fun_value

