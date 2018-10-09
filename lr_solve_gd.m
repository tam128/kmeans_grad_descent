function [w] = lr_solve_gd(x_train, y_train, iters, eta)
    N = size(x_train,1);
    w = zeros(size(x_train,2),1);
    
    for i=1:iters 
        y_pred = lr_predict(x_train, w);
        w = w + x_train'*(y_train - y_pred)*eta;
    end
    