% Read in wine dataset
wine = dlmread('winequality-red.csv',';',1,0);

% Randomly split data into train and test sets
N = size(wine,1);
ind = randperm(N);
train_ind = (ind<(N+1)/2);
test_ind = (ind>=(N+1)/2);

train = wine(train_ind,:); 
test = wine(test_ind,:); 

mean_train = mean(train);
std_train = std(train);

%Standardize train and test sets
train = (train - mean_train) ./ std_train;
test = (test - mean_train) ./ std_train;

% Divide into x and y values
x_train = train(:,1:end-1);
y_train = train(:,end);
x_test = test(:,1:end-1);
y_test = test(:,end);

% Closed form calculation of weights
w = lr_solve_closed(x_train, y_train);

% Add bias to train and test sets
bias_train = ones(size(x_train,1),1);
bias_test = ones(size(x_test,1),1); 
x_train = [bias_train x_train];
x_test = [bias_test x_test];

% Calculate closed form solution
y_pred_closed = lr_predict(x_test, w); 

fprintf("L2 distance for closed form solution")
norm(y_pred_closed-y_test,2)


% Gradient descent for 50 iterations and different eta values
for i=1:6
    w = lr_solve_gd(x_train, y_train, 50, 10^(-i)); 
    y_pred_grad = lr_predict(x_test, w);

    fprintf("L2 distance for gradient descent solution, eta = 10^-" + i)
    norm(y_pred_grad-y_test,2)
end

