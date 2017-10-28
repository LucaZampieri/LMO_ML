




The code is organized according to the following structure: 

1) “main.ipynb” : it runs also the function for the project including the pre-processing:
    
   This code is structured in the following 
   
   Each method considered in “implementations.py” is implemented considered the following scheme: 

2) Inside folder “lib” :



A - “implementations.py”: inside this file the following function are implemented:

    - least squares GD(y, tx, initial w, max iters, gamma) >>    Linear regression using gradient descent
    - least squares SGD(y, tx, initial w, max iters, gamma) >>   Linear regression using stochastic gradient descent    - least squares(y, tx)   >>>.  Least squares regression using normal equations Ridge regression using normal equations
    - ridge regression(y, tx, lambda )   >>>>. Least squares regression using normal equations Ridge regression using normal equations

    - logistic regression(y, tx, initial w, max iters, gamma).  >>>. Logistic regression using gradient descent or SGD
    - reg logistic regression(y, tx, lambda , initial w, max iters, gamma)  >>Regularized logistic regression using gradient descent or SGD 




B - “costs.py”: it contains all the loss functions: 
   

    - compute_loss(y, tx, w, fct): for a given tx input data and weights vector w , this function computes three kind of loss:  - MSE  between y and the predicted vector tx.dot(w) if “fct==mse”
 																- MAE : if “fct==mae”
																- RMSE : if “fct==rmse”  

    - compute_ridge_loss(y, tx, w, lambda_, fct='mse'): for a given tx input data and weights vector w , it returns the regularized mse or rmse (depending on fct parameter) for a given lambda. Used in 	ridge 	regression.


    - compute_logreg_loss(y, tx, w): for a given tx input data and weights vector w , it returns the log-likely-hood; 


    - compute_gradient(y, tx, w, fct='mse’): Compute the gradient of the MSE, MAE or the logistic regression losses.

    
    - compute_logreg_hessian(y, tx, w):  Compute the hessian of the logistic regression loss function ( used for with the newton method)

  
    - sigmoid(z): it computes the sigmoid function for a given vector z; 
 



   
    
   
   
      

 3) “proj1_helpers.py” : it contains all the functions need to load the data from .csv files and elaborate them : 

    - load_csv_data(data_path, sub_sample): load data from “data_path” ; if “sub_sample=True” a sub-sampling the input matrix is subsampled in respect to the rows ( default_sample_rate: 50:1) 
  
    - predict_labels(weights, data): it predicts the output ( classification ) from “data” for a given vector of weights.
   
    - create_csv_submission(ids, y_pred, name): for a given vector of predictions “y_pred” , ids? , this function create .csv file with path “name”.





4) “preprocessing_functions.py” : ########SPIEGARE FUNZIONI



1) “main.ipynb” : it runs all the functions for the project including the pre-processing.
    
   This code is structured in the following way:

   - load_csv_data : load the data from “.csv” file. 

   - panda library is used to show the data in a table 


   - func_(method) : it calls the method and visualize to standard output the plot of the weights including the accuracy; 

         func_least_squares 	>>>>> least_square
         func_GD            	>>>>>> gradient descent 
         func_SGD           	>>>>>> stochastic gradient descent
         func_ridge_regression   >>>>> ridge regression 
         func_logistic      	>>>>>> logistic_regression 
         func_logistic_reg  	>>>>>>. Regularized logistic regression
         
    
   - cross_validation_method: It performs the cross validation on the interesting parameters  for the specific “method”. 
                              In the cross validation on the degree of the input matrix:  ########SPIEGARE COME FA 

   
   - Each method considered in “implementations.py” is implemented with the following scheme:
   
       for each method the algorithm realizes the division of the training set in 4 jets. Cycling on each jet , the cross_validation returns the best interesting parameters 
    
    




 