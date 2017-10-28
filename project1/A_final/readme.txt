




The code is organized according to the following structure: 

1) “main.ipynb” : it runs also the function for the project including the pre-processing;


2) Inside folder “lib” :



A- “implementations.py”: inside this file the following function are implemented:

    - least squares GD(y, tx, initial w, max iters, gamma) >>    Linear regression using gradient descent
    - least squares SGD(y, tx, initial w, max iters, gamma) >>   Linear regression using stochastic gradient descent    - least squares(y, tx)   >>>.  Least squares regression using normal equations Ridge regression using normal equations
    - ridge regression(y, tx, lambda )   >>>>. Least squares regression using normal equations Ridge regression using normal equations

    - logistic regression(y, tx, initial w, max iters, gamma).  >>>. Logistic regression using gradient descent or SGD
    - reg logistic regression(y, tx, lambda , initial w, max iters, gamma)  >>Regularized logistic regression using gradient descent or SGD 


B- “costs.py”: it contains all the loss functions: 
   

    - compute_loss(y, tx, w, fct): for a given tx input data and weights vector w , this function computes three kind of loss:  - MSE  between y and the predicted vector tx.dot(w) if “fct==mse”
 																- MAE : if “fct==mae”
																- RMSE : if “fct==rmse”  

    - compute_ridge_loss(y, tx, w, lambda_, fct='mse'):
   
      

 3) “proj1_helpers.py” : it contains all the functions need to load the data from .csv files and elaborate them : 

    - load_csv_data(data_path, sub_sample): load data from “data_path” ; if “sub_sample=True” a sub-sampling the input matrix is subsampled in respect to the rows ( default_sample_rate: 50:1) 
  
    - predict_labels(weights, data): it predicts the output ( classification ) from “data” for a given vector of weights.
   
    - create_csv_submission(ids, y_pred, name): for a given vector of predictions “y_pred” , ids? , this function create .csv file with path “name”.




 