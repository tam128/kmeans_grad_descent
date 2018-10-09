function [w] = lr_solve_closed(x_train, y_train)
    w = pinv([ones(size(x_train,1),1) x_train])*y_train;