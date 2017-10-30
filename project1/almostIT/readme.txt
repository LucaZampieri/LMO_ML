


!!!!!!!!IMPORTANT: TO RUN “run.py” for the optimized code that gets the submission go in “run_folder”




The .zip folder is organized according to the following structure: 




1) Inside folder “lib” :



1A - “implementations.py”: inside this file the following function are implemented:

    - least squares GD(y, tx, initial w, max iters, gamma) >>    Linear regression using gradient descent
    - least squares SGD(y, tx, initial w, max iters, gamma) >>   Linear regression using stochastic gradient descent    - least squares(y, tx)   >>>.  Least squares regression using normal equations Ridge regression using normal equations
    - ridge regression(y, tx, lambda )   >>>>. Least squares regression using normal equations Ridge regression using normal equations

    - logistic regression(y, tx, initial w, max iters, gamma).  >>>. Logistic regression using gradient descent or SGD
    - reg logistic regression(y, tx, lambda , initial w, max iters, gamma)  >>Regularized logistic regression using gradient descent or SGD 




1B - “costs.py”: it contains all the loss functions: 
   

    - compute_loss(y, tx, w, fct): for a given tx input data and weights vector w , this function computes three kind of loss:  - MSE  between y and the predicted vector tx.dot(w) if “fct==mse”
 																- MAE : if “fct==mae”
																- RMSE : if “fct==rmse”  

    - compute_ridge_loss(y, tx, w, lambda_, fct='mse'): for a given tx input data and weights vector w , it returns the regularized mse or rmse (depending on fct parameter) for a given lambda. Used in 	ridge 	regression.

    - compute_logreg_loss(y, tx, w): for a given tx input data and weights vector w , it returns the log-likely-hood; 

    - compute_gradient(y, tx, w, fct='mse’): Compute the gradient of the MSE, MAE or the logistic regression losses.
    
    - compute_logreg_hessian(y, tx, w):  Compute the hessian of the logistic regression loss function ( used for with the newton method)
  
    - sigmoid(z): it computes the sigmoid function for a given vector z; 
 



2) “helpers.py” : it contains all the functions need to load the data from .csv files and elaborate them : 

    - load_csv_data(data_path, sub_sample): load data from “data_path” ; if “sub_sample=True” a sub-sampling the input matrix is subsampled in respect to the rows ( default_sample_rate: 50:1) 
  
    - predict_labels(weights, data): it predicts the output ( classification ) from “data” for a given vector of weights.
   
    - create_csv_submission(ids, y_pred, name): for a given vector of predictions “y_pred”  this function create .csv file with path “name”.

    - standardize(x) : it standardize the matrix x





3) “preprocessing_functions.py” :

 
    - split_jets_mask(tx): it split tx in different jets according to jet column;

    - clean_missing_values(tx, mean=False): it cleans the “-999” and replace them with the median of the column;

    - keep_unique_cols(tx): it removes the  equal columns to get a full ranked training matrix; 
  
    - add_cross_prod(tx, i, j): it adds a column to tx with the cross product of i-th and j-th column;

    - add_all_cross_prod(tx, selected_cols): it calls add_cross_prod(tx, i, j) for each column in “selected_cols”;

    - add_ones(tx): add a column of ones;

    - prepare data: it runs all the functions above according to a predefinitenite scheme;
 
    - add_powers(tx, range_degrees, range_col_idx): it adds to matrix “tx”  the columns referred in “range_col_idx” powered for each degree contained in “range_degrees”. 
 



4) “All_methods.ipynb” : it runs all the functions not-optimized for the project including the pre-processing.
    
   This functions used in this code are the following:

   - load_csv_data : load the data from “.csv” file;

   - panda library is used to show the data in a table;

   - func_(method) : it calls the method and visualize to standard output the plot of the weights including the accuracy; 

         func_least_squares 	>>>>> least_square
         func_GD            	>>>>>> gradient descent 
         func_SGD           	>>>>>> stochastic gradient descent
         func_ridge_regression   >>>>> ridge regression 
         func_logistic      	>>>>>> logistic_regression 
         func_logistic_reg  	>>>>>>. Regularized logistic regression


   - build_k_indices(y, k_fold, seed): it returns a vector of indexes for creating k-folders.  
         
    
   - cross_validation_method: It performs the cross validation on the interesting parameters  for the specific “method” for a specific input training set (we use for the single jet).
                              It split the training set in k_fold number of parts; for each k_fold it realizes the preprocessing and calls “cross_validation_one_fold_method”;
			      This function returns a n-dimensional matrix of accuracies for a single k_fold and so the mean over the k realizations is computed.
                              The n-tuple of parameters which correspond to best accuracy is returned. 


 

   -cross_validation_one_fold_method: Basically this perform a “for cycle” for each parameter we want to optimize. It returns a n-dimensional matrix of accuracies, where n is the number of 
					tuning parameters.

   
					
                           

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STRUCTURE OF THE CALL FOR A SINGLE METHOD: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
     Each method considered in “implementations.py” is implemented with the following scheme:
   
      - The algorithm realizes the division of the training set in 4 jets using the function  “split_jets_mask(tx_train)”.
        
      	- For each jet “the cross_validation_method”(y_jet_i_train_i, tx_jet_i_train, k_fold,  parameters) is called to return the best parameters.

      	- Once the parameters are computed “func_(method)” is called and returns the weights for the real prediction for the single jet.

      	- “predict_labels” ( in helpers.py) realizes the prediction on the single jet for “tx_jet_i_test”.

      	- Using a mask the resulted predictions (“y_pred_jet_i_test”) are putted in the correct position of y_pred_test

      -	Once the prediction is performed for each jet y_pred_test contains the predictions for whole test_set and the results are the loaded in a .csv for the submission 
  

     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%










5) Inside “run_folder” there is the optimized version of the code. As reported in the paper we select RIDGE REGRESSION. 
    The structure of this code is very similar to the one reported above apart from the fact that the functions are optimized in terms of computational effort.

   This functions used in this code are the following:

    FOR PRE-PROCESSING:
	 
    	- split_jets_mask(tx): it split tx in different jets according to jet column;

    	- clean_missing_values(tx, mean=False): it cleans the “-999” and replace them with the median of the column;

    	- keep_unique_cols(tx): it removes the  equal columns to get a full ranked training matrix; 
  
    	- add_cross_prod(tx, i, j): it adds a column to tx with the cross product of i-th and j-th column;

    	- add_all_cross_prod(tx, selected_cols): it calls add_cross_prod(tx, i, j) for each column in “selected_cols”;

    	- add_ones(tx): add a column of ones;

    	- prepare data: it runs all the functions above according to a predefinitenite scheme;
 
    	- add_powers(tx, range_degrees, range_col_idx): it adds to matrix “tx”  the columns referred in “range_col_idx” powered for each degree contained in “range_degrees”. 


    - cleaned_ridge_regression_pred(single_degree, single_lambda, single_train_tx, single_test_tx, single_train_y, single_test_y=[], predictions=True):


    - ridge_regression_all_jets_pred(full_tx_train, full_tx_test, full_y_train, degrees, lambdas): it divides the dataset into 


    -cross_validation_single_jet_ridge_regression:	




	


	


 
         
        
    
    




 