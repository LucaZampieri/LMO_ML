*** README for the 1st Machine Learning Project, finding the Higgs boson --- Team AlmostIT. 

IMPORTANT: TO RUN “run.py” to reproduce the best submission, go in “run_folder” and type: python run.py
Data sets 'test.csv' and 'train.csv' have to be put with this name in the 'data/' folder. 
The output containing the submission will be loaded in the folder "run_folder/output/". 


The .zip folder is organized according to the following structure: 

- “All_functions/”: codes for testing all functions  (containing “implementations.py”);
- “run_folder/”: codes to reproduce the definitive submission (containing “run.py”);
- “data_visualization/”: for data understanding and plotting 


almostIT.zip
  |--All_functions
  |   |--lib
  |   |  |--implementations.py
  |   |  |--costs.py
  |   |  |--helpers.py
  |   |  |--preprocessing_functions.py
  |   |  '--All_methods.ipynb
  |   |--output
  |   |  '
  |--run_folder
  |   |--costs.py
  |   |--helpers.py
  |   |--optimal_cross_validation.py
  |   |--optimal_ridge_regression.py
  |   |--preprocessing_functions.py
  |   |--run.py
  |   |--Cross_validation_optimal_results.html
  |   |--output
  |   |  '
  |--Data_vizualization
  |   |--data_understanding.ipynb
  |   |--data_understanding.html
  |   |--data_viz_helpers.py
  |   |--Correlation_matrix.png
  |   |--train_jet_comparison_dist.png
  |   |--data
  |   |  '
  |-- data




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

INSIDE “All_function” :    subfolders: “lib”, “output”;
                           Files:    “preprocessing_functions.py”, “All_functions.ipynb”

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


1) Inside subfolder “lib” :

1a) “implementations.py”: inside this file the following functions are implemented: (dependence on “costs.py” file) 

    - least_squares GD(y, tx, initial w, max iters, gamma) >>    Linear regression using gradient descent

    - least_squares SGD(y, tx, initial w, max iters, gamma) >>   Linear regression using stochastic gradient descent
         batch_iter(y, tx, batch_size, num_batches=1, shuffle=True): it is used to create the batches for the stochastic gradient descent

    - least_squares(y, tx)   >>>.  Least squares regression using normal equations Ridge regression using normal equations

    - ridge_regression(y, tx, lambda )   >>>>. Least squares regression using normal equations Ridge regression using normal equations

    - logistic_regression(y, tx, initial w, max iters, gamma).  >>>. Logistic regression using gradient descent or SGD

    - reg_logistic_regression(y, tx, lambda , initial w, max iters, gamma)  >>Regularized logistic regression using gradient descent or SGD 


1b)  “costs.py”: it contains all the loss functions: 

    - compute_loss(y, tx, w, fct): for a given tx input data and weights vector w , this function computes three kind of loss:  - MSE  between y and the predicted vector tx.dot(w) if “fct==mse”
 																- MAE : if “fct==mae”
																- RMSE : if “fct==rmse”  

    - compute_ridge_loss(y, tx, w, lambda_, fct='mse'): for a given tx input data and weights vector w , it returns the regularized mse or rmse (depending on fct parameter) for a given lambda. Used in 	ridge 	regression.

    - compute_logreg_loss(y, tx, w): for a given tx input data and weights vector w , it returns the log-likely-hood; 

    - compute_gradient(y, tx, w, fct='mse’): Compute the gradient of the MSE, MAE or the logistic regression losses.
    
    - compute_logreg_hessian(y, tx, w):  Compute the hessian of the logistic regression loss function ( used for with the newton method)
  
    - sigmoid(z): it computes the sigmoid function for a given vector z; 


1c) “helpers.py” : it contains all the functions need to load the data from .csv files and elaborate them : 

    - load_csv_data(data_path, sub_sample): load data from “data_path” ; if “sub_sample=True” a sub-sampling the input matrix is subsampled in respect to the rows ( default_sample_rate: 50:1) 
  
    - predict_labels(weights, data): it predicts the output ( classification ) from “data” for a given vector of weights.
   
    - create_csv_submission(ids, y_pred, name): for a given vector of predictions “y_pred”  this function create .csv file with path “name”.

    - standardize(x) : it standardize the matrix x

    - load_data(): it loads the data

    - build_model_data(height, weight): "Form (y,tX) to get regression data in matrix form.
   
   

2) “preprocessing_functions.py” :

    - split_jets_mask(tx): it split tx in different jets according to jet column;

    - clean_missing_values(tx, mean=False): it cleans the “-999” and replace them with the median of the column;

    - keep_unique_cols(tx): it removes the  equal columns to get a full ranked training matrix; 
  
    - add_cross_prod(tx, i, j): it adds a column to tx with the cross product of i-th and j-th column;

    - add_all_cross_prod(tx, selected_cols): it calls add_cross_prod(tx, i, j) for each column in “selected_cols”;

    - add_ones(tx): add a column of ones;

    - prepare data: it runs all the functions above according to a predefinitenite scheme;
 
    - add_powers(tx, range_degrees, range_col_idx): it adds to matrix “tx”  the columns referred in “range_col_idx” powered for each degree contained in “range_degrees”. 
 


3) “All_functions.ipynb” : it runs all the functions not-optimized for the project including the pre-processing.
    
   This functions used in this code are the following:

   - load_csv_data : load the data from “.csv” file;

   - panda library is used ONLY to show the data in a table;

   - build_k_indices(y, k_fold, seed): it returns a vector of indexes for creating k-folders.  
    
   - cross_validation_method: It performs the cross validation on the interesting parameters  for the specific “method” for a specific input training set (we use for the single jet).
                              It split the training set in k_fold number of parts; for each k_fold it realizes the preprocessing and calls “cross_validation_one_fold_method”;
			      This function returns a n-dimensional matrix of accuracies for a single k_fold and so the mean over the k realizations is computed.
                              The n-tuple of parameters which correspond to best accuracy is returned. 

   -cross_validation_one_fold_method: Basically this perform a “for cycle” for each parameter we want to optimize. It returns a n-dimensional matrix of accuracies, where n is the number of 
				      tuning parameters.


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STRUCTURE OF THE CALL FOR A SINGLE METHOD: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     Each method considered in “implementations.py” is implemented with the following scheme:
   
      - The algorithm realizes the division of the training set in 4 sub-datasets corresponding to the jet numbers, using the function  “split_jets_mask(tx_train)”.
        
      	- For each jet “the cross_validation_method”(y_jet_i_train_i, tx_jet_i_train, k_fold,  parameters) is called to return the best parameters.

      	- Once the parameters are computed, the algorithm is called and returns the weights for the real prediction for the single jet.

      	- “predict_labels” (in helpers.py) realizes the prediction on the single jet for “tx_jet_i_test”.

      	- Using a mask the resulted predictions (“y_pred_jet_i_test”) are putted in the correct position of y_pred_test.

      -	Once the prediction is performed for each jet, y_pred_test contains the predictions for the whole test_set and the results are the loaded in a .csv for the submission.
  

     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


4) Inside “data”, put the input files “train.csv” and "test.csv". 


5) Inside “output” , “All_methods.ipynb”  loads the results for the submission







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

INSIDE  “run_folder”:   subfolders:“output”
    			files: “costs.py” , “helpers.py” , ”optimal_corss_validation.py”, “optimal_ridge_regression.py”, “preprocessing_functions.py”, “run.py”, "Cross_validation_optimal_results.html"

    Here is the optimized version of the code. As reported in the paper we select RIDGE REGRESSION. 
    The structure of this code is very similar to the one reported above apart from the fact that the functions are optimized in terms of computational effort.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 

1) “costs.py”: it contains the functions “compute_loss” and  “compute_ridge_loss” as reported above



2) “helpers.py”: same as inside "All_functions"



3) “optimal_cross_validation.py” : 

    - “cross_validation_all_jets_ridge_regression(degrees, lambdas, k_fold, seed, full_y_train, full_tx_train, full_tx_test)”: Separates the full dataset into jets and perform the cross validation on the parameters of the ridge regression on each jet.


   - “plot_accuracy_evolution(degrees, lambdas, mean_acc_cv_train, mean_acc_cv_test, var_acc_cv_train, var_acc_cv_test)”: Plots the evolution of the accuracies of the training and testing sets during the cross validation for the different chosen values of degrees and lambdas. 


   - “cross_validation_single_jet_single_param_ridge_regression(jet_id, single_jet_y, single_jet_tx, k_fold, seed, single_degree, single_lambda)”: Splits the dataset into folds and compute the accuracy of the ridge regression on each fold.


     -cross_validation_single_jet_ridge_regression(jet_id, degrees, lambdas, y_single_jet_train, tx_single_jet_train, k_fold, seed, returnAll=False): On a single jet, compute the accuracy of the ridge regression for each chosen degree and lambda and returns the best values of degree and lambda for each jet number, and the corresponding training and testing accuracies.




4) “optimal_ridge_regression.py” :

     - ridge_regression(y, tx, lambda_, fct='none'): Computes the best weights and the result of the loss function obtained by ridge regression.

     - cleaned_ridge_regression_pred(jet_id, single_degree, single_lambda, single_train_tx, single_test_tx, \
                                  single_train_y, single_test_y=[], predictions=True): Computes the predictions of a single jet number with the ridge regression, given a degree (maximum polynomial degree) and a lambda (stabilizing parameter).

     - ridge_regression_all_jets_pred(full_tx_train, full_tx_test, full_y_train, degrees, lambdas): Splits the dataset by jets and computes the ridge regression predictions on each newly created dataset. 
 


5)  “preprocessing_functions.py” : 
	 
    	- split_jets_mask(tx): it split tx in different jets according to jet column;

    	- clean_missing_values(tx, mean=False): it cleans the “-999” and replace them with the median of the column;

    	- keep_unique_cols(tx): it removes the  equal columns to get a full ranked training matrix; 
  
    	- add_cross_prod(tx, i, j): it adds a column to tx with the cross product of i-th and j-th column;

    	- add_all_cross_prod(tx, selected_cols): it calls add_cross_prod(tx, i, j) for each column in “selected_cols”;

    	- add_ones(tx): add a column of ones;

    	- prepare data: it runs all the functions above according to a predefinitenite scheme;
 
    	- add_powers(tx, range_degrees, range_col_idx): it adds to matrix “tx”  the columns referred in “range_col_idx” powered for each degree contained in “range_degrees”. 



6) “run.py”: it runs the following step:


	 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STRUCTURE OF THE CALL FOR METHOD “run.py” : %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
	 - laoading data.

         - In order to perform the cross-validation to find the best parameters, uncomment the "Cross validation" part. Otherwise, only the run with the previously found best parameters will be done.

         - compute the predictions thanks to the ridge regression. 
    
         - create the submission.

          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


7) "Cross_validation_optimal_results.html" shows the full cross validation on the ridge regression and the results obtained. 



	


	


 
         
        
    
    




 
