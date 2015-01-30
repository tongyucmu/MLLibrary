%load data
data_train = load('Training Data.txt');
data_test = load('Testing Data.txt');
%get feature and label
x_train = data_train(:,1:end-1);
y_train = data_train(:,end);
x_test = data_test(:,1:end-1);
y_test = data_test(:,end);
y_predict = zeros(size(y_test));

%the labels' list and number of class
labels = unique(y_train);
num_class = size(labels,1);
label_list = [];
label_prob = [];

for i = 1: num_class
	label_list = [ label_list labels(i)];
	label_prob = [ label_prob sum(labels(i) == y_train)];
end

%sub question1: complete the vector of prior probabilities of those classes
label_prob = label_prob/size(y_train,1);

disp('sub question1: complete the vector of prior probabilities of those classes');
label_prob

matrix_mu = zeros(size(x_train,2),num_class);
matrix_sigma = zeros(size(x_train,2),num_class);

%sub question2: calculate the matrix of mu and sigma
for i = 1:size(x_train,2)
	for j = 1:num_class	
		matrix_mu(i,j) = mean(x_train(find(label_list(j) == y_train), i));
		matrix_sigma(i,j) = var(x_train(find(label_list(j) == y_train), i),1);
	end
end

disp('sub question2: calculate the matrix of mu and sigma')
matrix_mu
matrix_sigma

% get the classification results of each testing example
num_test = size(x_test,1);
num_dim = size(x_test,2);

for t = 1: num_test
	prob_list = ones(1,num_class);
	for j = 1: num_class
		for d = 1: num_dim
			x = x_test(t, d);
			prob_list(j) = prob_list(j)*exp(-(x-matrix_mu(d,j))^2/(2*matrix_sigma(d,j)))/sqrt(2*pi*matrix_sigma(d,j));
		end
		prob_list(j) = prob_list(j)*label_prob(j);
	end
	[~, index] = max(prob_list);
	y_predict(t) = label_list(index);
end

%sub question3: print out the classifications of each testing example.
disp('sub question3: print out the classifications of each testing example')
y_predict

%sub question4: show the accuracy of your classifier
disp('sub question4: show the accuracy of your classifier')
acc = sum(y_predict == y_test)/num_test

