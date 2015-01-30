function [W, f_list] = train_test_nt(file_train, c)
% set long to display more bits in terminal
format long
% the list record function values at each iteration
f_list = [];
% define the two as global so that we don't need to pass them to function
% every time
global X_train Y_train

% load data 
[Y_train X_train] = libsvmread(file_train);
% change label from 0 to -1, as negative data is regarded as -1 in my
% implementaion
Y_train(find(Y_train == 0)) = -1;

% initialization
eta = 0.01;
Xi = 0.1;
dim = size(X_train, 2);
num_ins = size(Y_train, 1);
W = zeros(1, dim);
W_start = W;
g = zeros(1, dim);


g1 = logis_fun_grad(W_start, c);
g2 = logis_fun_grad(W, c);

count_iter = 0;
while norm(g1)*0.01 < norm(g2)
    count_iter = count_iter + 1;
    %n1 = norm(g1)*0.01
    %n2 = norm(g2)
    
    % initialization in CG
    s = 0;
    r = -g2;
    d = r;
    
    % CG to find the solution to newton linear system
    count_cg = 0;
    while 1    
        if norm(r) < Xi*norm(g2)
            break
        end
        count_cg =  count_cg + 1;
        tmp =  appro_hessian(d, W, c);

        a = norm(r)^2/(d*tmp);
        s = s + a*d;
        r1 = r - (a*tmp)';
        b = norm(r1)^2/norm(r)^2;
        r = r1;
        d = r + b*d;
    end
   
    v = s;
    
    % line search to find step size
    count_linesearch = 0;
    while 1 
        alpha = 1/(2^count_linesearch);
        if logis_fun(W + alpha*v, c) <= logis_fun(W, c) + eta*alpha*g2*v'
            break;
        end
        count_linesearch = count_linesearch + 1;
    end
    
    % calculate function value
    f_value = logis_fun(W, c);
    f_list = [f_list f_value];
    
    % print the information at each iteration
    info = ['iter '  num2str(count_iter) ' f ' num2str(f_value) ' |g| ' num2str(norm(g2))  ' CG ' num2str(count_cg) ' alpha ' num2str(alpha)];  
    disp(info);
    
    % update the W
    W = W +alpha*v;
    % update the g2 to use in the next iteration
    g2 = logis_fun_grad(W, c);
    
end
% calculate function value
f_value = logis_fun(W, c);
f_list = [f_list f_value];
% print the information at each iteration
info = ['end of iter '  num2str(count_iter) ' f ' num2str(f_value) ' |g| ' num2str(norm(g2))];  
    disp(info);

% a brief test on the original training data
Y_test = zeros(num_ins, 1);

Y_test(W*X_train'>=0) = 1;
Y_test(W*X_train'<0) = -1;

acc = sum(Y_test == Y_train)/num_ins

