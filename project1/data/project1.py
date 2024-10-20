import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNetCV, RidgeCV, Ridge, LassoCV, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import scipy
from scipy import stats
from scipy.stats import skew, pearsonr

import optuna
from optuna import Trial
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')

# Set random seed
seed_val = 1160
np.random.seed(seed_val)

linear_numeric_feats = ['Lot_Frontage', 'Lot_Area', 'Year_Built', 'Year_Remod_Add',
                        'Mas_Vnr_Area', 'Bsmt_Unf_SF', 'Total_Bsmt_SF', 'First_Flr_SF',
                        'Second_Flr_SF', 'Gr_Liv_Area', 'Garage_Area', 'Wood_Deck_SF',
                        'Open_Porch_SF', 'Latitude', 'Longitude']

# linear_numeric_feats = ["Lot_Frontage", "Lot_Area", "Year_Built", "Year_Remod_Add", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "First_Flr_SF", "Second_Flr_SF",
#                         "Low_Qual_Fin_SF", "Gr_Liv_Area", "Garage_Yr_Blt", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val", "Longitude", "Latitude"]


def load_dataframe(target_fold_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # import the given fold data
    X_train_full = pd.read_csv(f'{target_fold_dir}/train.csv')
    X_train = X_train_full.iloc[:, 1:-1]
    y_train = np.log1p(X_train_full.iloc[:, -1])

    # import the test data
    X_test = pd.read_csv(f'{target_fold_dir}/test.csv').iloc[:, 1:]
    y_test = np.log1p(pd.read_csv(f'{target_fold_dir}/test_y.csv').iloc[:, -1])

    return X_train, y_train, X_test, y_test


def transform_training_target(X_train: pd.DataFrame, y_train: pd.Series, std_threshold: float = 3) -> tuple[pd.DataFrame, pd.Series]:
    # Drop the rows with extreme price larger
    mean = y_train.mean()
    std = y_train.std()
    rows_to_drop = y_train[(y_train > mean + std_threshold * std) |
                           (y_train < mean - std_threshold * std)].index
    print(f'Drop {len(rows_to_drop)} rows')
    X_train = X_train.drop(index=rows_to_drop)
    y_train = y_train.drop(index=rows_to_drop)
    return X_train, y_train, mean


def clean(df: pd.DataFrame) -> pd.DataFrame:

    # Cleaning numerical features
    max_year = 2011  # the max year of the training data
    # corrupted Garage_Yr_Blt = less than Year_Built or nan
    df['Garage_Yr_Blt'] = df['Garage_Yr_Blt'].apply(
        lambda x: x if x <= max_year else np.nan)
    df['Garage_Yr_Blt'] = df['Garage_Yr_Blt'].fillna(df['Year_Built'])

    # Cleaning categorical features
    # dropping the rows with missing values in the categorical features
    train_null = df.isnull().sum()[df.isnull(
    ).sum() != 0].sort_values(ascending=False)
    df = df.drop(columns=train_null.index)
    print(f"Dropping categorical features with missing values: {train_null}")
    
    return df


def process_numeric_features(df: pd.DataFrame, skew_threshold: float = 0.75) -> tuple[pd.DataFrame, list]:
    # log transform skewed numeric features:
    # numeric_feats = df.dtypes[df.dtypes != "object"].index

    # only consider the skewed features
    skewed_feats = df[linear_numeric_feats].apply(
        lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > skew_threshold]
    skewed_feats = skewed_feats.index

    # log transform the skewed features
    df[skewed_feats] = np.log1p(df[skewed_feats])
    print(f"Log transformed {len(skewed_feats)} skewed features")
    print(f"The skewed features are {skewed_feats}")
    return df, skewed_feats


def delete_outliers(X: pd.DataFrame, y: pd.Series, feature: str, threshold: float = 5) -> pd.DataFrame:
    z_scores = np.abs(stats.zscore(X[feature]))
    outliers = X[z_scores > threshold]
    # print(f"Dropping {len(outliers)} outliers in {feature}")
    return X.drop(outliers.index), y.drop(outliers.index)


def rmse(y_test: pd.Series, y_pred: pd.Series) -> float:
    return np.sqrt(np.mean((y_pred - y_test) ** 2))


def full_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    model = LinearRegression()
    model.fit(X_train, y_train)
    # training error
    y_pred_train = model.predict(X_train)
    train_rmse = rmse(y_train, y_pred_train)
    # prediction error
    y_pred = model.predict(X_test)
    test_rmse = rmse(y_test, y_pred)
    return train_rmse, test_rmse


def ridge_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    ridge_alphas = np.logspace(-1, 3, 100)
    model = make_pipeline(RobustScaler(), RidgeCV(
        alphas=ridge_alphas, cv=10))
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    train_rmse = rmse(y_train, y_pred_train)
    y_pred = model.predict(X_test)
    test_rmse = rmse(y_test, y_pred)
    return train_rmse, test_rmse


def lasso_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    lasso_alphas = np.logspace(-5, -2, 100)
    model = LassoCV(alphas=lasso_alphas, cv=10, random_state=seed_val)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    train_rmse = rmse(y_train, y_pred_train)
    y_pred = model.predict(X_test)
    test_rmse = rmse(y_test, y_pred)
    return train_rmse, test_rmse


def elasticnet_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    model = ElasticNetCV(alphas=np.logspace(-5, 3, 100), cv=10)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    train_rmse = rmse(y_train, y_pred_train)
    y_pred = model.predict(X_test)
    test_rmse = rmse(y_test, y_pred)
    return train_rmse, test_rmse


def xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    xgb_params = {'objective': 'reg:squarederror',
                  'max_depth': 5,
                  'learning_rate': 0.0074012209282183683,
                  'n_estimators': 2800,
                  'min_child_weight': 10,
                  'subsample': 0.5259427801631228,
                  'colsample_bytree': 0.5525252726939558,
                  'reg_alpha': 5.8503033950745276e-05,
                  'reg_lambda': 0.07447269431255081,
                  'random_state': seed_val}

    xgb_tuned = XGBRegressor(**xgb_params)
    xgb_tuned.fit(X_train, y_train)
    y_pred_train = xgb_tuned.predict(X_train)
    train_rmse = rmse(y_train, y_pred_train)
    y_pred = xgb_tuned.predict(X_test)
    test_rmse = rmse(y_test, y_pred)
    return train_rmse, test_rmse


def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    # Suggest hyperparameters
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 5000, step=100),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e-1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1e-1, log=True),
        'random_state': seed_val,
        'objective': 'reg:squarederror'
    }

    # Initialize the model with suggested hyperparameters
    model = XGBRegressor(**param)

    # Perform cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    rmse = np.sqrt(-cv_scores.mean())

    return rmse


def tune_xgboost_params(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 50) -> dict:
    sampler = TPESampler(seed=seed_val)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(lambda trial: objective(
        trial, X_train, y_train), n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  RMSE: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return trial.params


def main(target_fold_dir: str) -> None:
    # Load the dataframes
    X_train, y_train, X_test, y_test = load_dataframe(target_fold_dir)

    # Make copies for processing
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    y_train_processed = y_train.copy()

    # Log transform the training target
    X_train_processed, y_train_processed, y_train_mean = transform_training_target(
        X_train_processed, y_train_processed)

    # Clean the data
    X_train_processed = clean(X_train_processed)
    X_test_processed = clean(X_test_processed)

    # Preprocess numerical features
    X_train_processed, skewed_feats = process_numeric_features(
        X_train_processed)
    # Remarks: apply the same transformation (on training data) to the test data
    X_test_processed[skewed_feats] = np.log1p(X_test_processed[skewed_feats])

    # Delete outliers in linear numerical features
    # numeric_feats = X_train_processed.dtypes[X_train_processed.dtypes != "object"].index
    for feature in linear_numeric_feats:
        if feature in X_train_processed.columns:
            X_train_processed, y_train_processed = delete_outliers(
                X_train_processed, y_train_processed, feature)

    # Preprocess categorical features
    X_train_processed = pd.get_dummies(X_train_processed)
    X_test_processed = pd.get_dummies(X_test_processed)
    # If testing data don't have the feature, fill it with the mean of the training data
    X_test_processed = X_test_processed.reindex(
        columns=X_train_processed.columns, fill_value=0)

    # Train models
    model_params = {
        'X_train': X_train_processed,
        'y_train': y_train_processed,
        'X_test': X_test_processed,
        'y_test': y_test
    }
    full_model_train_rmse, full_model_test_rmse = full_model(**model_params)
    ridge_model_train_rmse, ridge_model_test_rmse = ridge_model(**model_params)
    lasso_model_train_rmse, lasso_model_test_rmse = lasso_model(**model_params)
    elasticnet_model_train_rmse, elasticnet_model_test_rmse = elasticnet_model(
        **model_params)
    xgboost_model_train_rmse, xgboost_model_test_rmse = xgboost_model(
        **model_params)

    # Hyperparameter Tuning
    # print("Xgboost rmse before tuning: ", xgboost_model_rmse)
    # print("Starting hyperparameter tuning for XGBoost...")
    # tuned_params = tune_xgboost_params(
    #     X_train_processed, y_train_processed, n_trials=100)
    # print(f"Tuned params: {tuned_params}")
    # xgb_tuned = XGBRegressor(**tuned_params)
    # xgb_tuned.fit(X_train_processed, y_train_processed)
    # y_pred_xgb_tuned = xgb_tuned.predict(X_test_processed)
    # xgboost_model_rmse = rmse(y_test_log, y_pred_xgb_tuned)

    # Conclusion
    print("Current fold target: ", target_fold_dir)
    print("Training Error:")
    print(f"Full model score: {full_model_train_rmse:.5f} RMSE")
    print(f"Lasso score with optimal alpha: {lasso_model_train_rmse:.5f} RMSE")
    print(f"Ridge score with optimal alpha: {ridge_model_train_rmse:.5f} RMSE")
    print(f"Elasticnet score: {elasticnet_model_train_rmse:.5f} RMSE")
    print(f"Boosting Tree score: {xgboost_model_train_rmse:.5f} RMSE")

    print("Test Error:")
    print(f"Full model score: {full_model_test_rmse:.5f} RMSE")
    print(f"Lasso score with optimal alpha: {lasso_model_test_rmse:.5f} RMSE")
    print(f"Ridge score with optimal alpha: {ridge_model_test_rmse:.5f} RMSE")
    print(f"Elasticnet score: {elasticnet_model_test_rmse:.5f} RMSE")
    print(f"Boosting Tree score: {xgboost_model_test_rmse:.5f} RMSE")


if __name__ == '__main__':

    # Train all folds
    # for fold in range(1, 11):
    #     print(f"#### Fold {fold}: ####")
    #     main(f'fold{fold}')
    #     print()

    main(f'fold4')
