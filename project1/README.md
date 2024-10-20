### Code
- Your code will be executed in the same directory as the data files. Therefore, please avoid using any commands that access or reset the current directory.

- Please ensure that your script only accepts train.csv and test.csv as inputs. Do not include specific paths like "XX/Proj1_Data/fold_1/train.csv".

- Do not use PID as a feature. Train and test data should be processed separately.

- No evaluation is required , so your code should not attempt to access test_y.csv.

- Do not write your code to automatically iterate through all 10 folders. We will run your code inside each folder individually.

- Ensure you load all necessary libraries in your script.

- Name your two output files as: mysubmission1.txt and mysubmission2.txt; values in the output files are separated by commas.

- On Canvas, when you submit your file multiple times, it may rename your file from mymain to mymain-1 or mymain-2. That’s fine, no need to worry about it.

- Remove any evaluation code, runtime capture code, and corresponding libraries before your final submission.

- The error rate should be strictly less than 0.125/0.135. Avoid eyeballing the output, as it may not show sufficient precision.

- If possible, submit your code for pre-evaluation before the deadline. Please stay tuned for our email announcement this Saturday.

### Report
- Submit a PDF report, with a maximum of two pages.

- The minimum font size is 11pt; there are no specific requirements for spacing or margins.

- Include the names and netids of all teammates. If it’s a team project, add a short paragraph describing each teammate's contribution.

- You do not need to include your code. However, if a line of code is particularly illustrative, feel free to include it.

- Remember to include the 2x10 accuracy table (each of the two models has 10 test errors). Screenshots of the table are acceptable.

- Please always include your local results, even if they differ from the results of our pre-evaluation of your code.

- Feel free to include any additional results you wish to share.

- How much detail should you include? A good rule of thumb is to provide enough information so that your PSL classmates could accurately replicate your results.

- Students are permitted to use the code we provided on Campuswire or code obtained from external sources. In the case of the latter, please appropriately acknowledge the source(s) by citing them in your report.

- Report Running Time

- Report the running time for each of the 10 training/test splits. You can report the running time separately for different models or as a total for both models.

- You may or may not include preprocessing and evaluation time—just be consistent across all 10 splits, and make sure to clearly state in the report what you're reporting.


### Allowed python packages

For Project 1, we have curated a list of R/Python packages that students are permitted to use. This approach streamlines the evaluation process by ensuring all necessary packages are pre-installed. Based on feedback and experience from previous semesters, the provided package list should be comprehensive for the requirements of Project 1.

Please note that these packages apply only to the code you submit for evaluation, where we will verify that you meet the evaluation metrics. You are free to use any additional packages for exploratory data analysis, generating graphs/plots, logging execution time, or controlling the script execution, etc. However, there is no need to include these in your submitted code.

R Packages:
readr, DescTools, forcats, fastDummies, vtreat
dplyr, tidyr, reshape2, tidyverse, plyr, data.table
CARET, tidymodels
randomForest, xgboost, GBM, Catboost, lightGBM
glmnet
ggplot2

Python Packages:
pandas
scipy
numpy
xgboost, lightGBM, Catboost
sklearn
category_encoders
feature_engine.outliers
glmnet_python
rpy2
warnings

(You do not need matplotlib or seaborn)

However, if you believe an essential package is missing from our list and its inclusion is critical to your work, please bring it to our attention.

Helpful Tip: If you only need a specific function from a package, it might be more efficient to directly copy that function and include it at the beginning of your code. For instance, last year, a team wanted to use the skewness function from the R package moments. They were able to simply copy the function (in the linked below) and embed it directly into their script.
https://github.com/cran/moments/blob/master/R/skewness.R



### Referenced Kaggle Notebook

- [Random Forest, XGB, Ridge, Lasso : All in one](https://www.kaggle.com/code/janvichokshi/random-forest-xgb-ridge-lasso-all-in-one/notebook)

- [House Price Prediction: Top 4 %](https://www.kaggle.com/code/noussairmighri/house-price-prediction-top-4/notebook)

- [Regularized Linear Models](https://www.kaggle.com/code/apapiu/regularized-linear-models)