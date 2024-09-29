### Variable selection

1. Alphas and MSE in sckit-learn

[docs - LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)

[docs - RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)

Attributes:
- alpha_ -> float: The amount of penalization chosen by cross validation.
- mse_path_ -> ndarray of shaep (n_alphas, n_folds): Mean square error for the test set on each fold, varying alpha.