import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import scipy
from scipy import stats
from scipy.stats import skew, pearsonr

import warnings
warnings.filterwarnings('ignore')


def load_dataframe(target_fold_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # import the given fold data
    X_train_full = pd.read_csv(f'{target_fold_dir}/train.csv', index_col='PID')
    X_train = X_train_full.iloc[:, :-1]  # exclude the last two columns
    y_train = np.log(X_train_full.iloc[:, -1])

    # import the test data
    X_test = pd.read_csv(f'{target_fold_dir}/test.csv', index_col='PID')
    y_test = np.log(pd.read_csv(
        f'{target_fold_dir}/test_y.csv', index_col='PID').iloc[:, -1])

    return X_train, y_train, X_test, y_test

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


def process_numeric_features(df: pd.DataFrame, skew_threshold: float = 0.5) -> tuple[pd.DataFrame, list]:
    # log transform skewed numeric features:
    numeric_feats = df.dtypes[df.dtypes != "object"].index

    # only consider the skewed features
    skewed_feats = df[numeric_feats].apply(
        lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > skew_threshold]
    skewed_feats = skewed_feats.index

    # log transform the skewed features
    df[skewed_feats] = np.log1p(df[skewed_feats])
    print(f"Log transformed {len(skewed_feats)} skewed features")
    print(f"The skewed features are {skewed_feats}")
    return df, skewed_feats


def delete_outliers(X: pd.DataFrame, y: pd.Series, feature: str, threshold: float = 3) -> pd.DataFrame:
    z_scores = np.abs(stats.zscore(X[feature]))
    outliers = X[z_scores > threshold]
    print(f"Dropping {len(outliers)} outliers in {feature}")
    return X.drop(outliers.index), y.drop(outliers.index)


def rmse(y_test: pd.Series, y_pred: pd.Series) -> float:
    return np.sqrt(np.mean((y_pred - y_test) ** 2))


def full_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    # full model with no penalty
    model = LinearRegression()
    model.fit(X_train, y_train)

    # verify the model with the test data
    y_pred = model.predict(X_test)
    score = rmse(y_test, y_pred)
    return score


if __name__ == '__main__':
    # Set random seed
    seed_val = 1160
    np.random.seed(seed_val)

    # Targeted directory
    target_fold_dir = 'fold1'

    # Load the dataframes
    X_train, y_train, X_test, y_test = load_dataframe(target_fold_dir)

    # Make copies for processing
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    y_train_processed = y_train.copy()
    y_test_processed = y_test.copy()
    
    # Drop the rows with extreme price larger
    mean = y_train.mean()
    std = y_train.std()    
    rows_to_drop = y_train[(y_train > mean + 3 * std) | (y_train < mean - 3 * std)].index
    print(f'Drop {len(rows_to_drop)} rows')
    X_train_processed = X_train_processed.drop(index=rows_to_drop)
    y_train_processed = y_train_processed.drop(index=rows_to_drop)

    # Clean the data
    X_train_processed = clean(X_train_processed)
    X_test_processed = clean(X_test_processed)

    # Preprocess numerical features
    X_train_processed, skewed_feats = process_numeric_features(
        X_train_processed)
    # Remarks: apply the same transformation (on training data) to the test data
    X_test_processed[skewed_feats] = np.log1p(X_test_processed[skewed_feats])

    # Delete outliers in linear numerical features
    numeric_feats = ['Lot_Frontage', 'Lot_Area', 'Year_Built', 'Year_Remod_Add',
                     'Mas_Vnr_Area', 'Bsmt_Unf_SF', 'Total_Bsmt_SF', 'First_Flr_SF',
                     'Second_Flr_SF', 'Gr_Liv_Area', 'Garage_Area', 'Wood_Deck_SF',
                     'Open_Porch_SF', 'Latitude']
    # numeric_feats = X_train_processed.dtypes[X_train_processed.dtypes != "object"].index
    for feature in numeric_feats:
        if feature in X_train_processed.columns:
            X_train_processed, y_train_processed = delete_outliers(
                X_train_processed, y_train_processed, feature)
            X_test_processed, y_test_processed = delete_outliers(
                X_test_processed, y_test_processed, feature)

    # Encode categorical features
    X_train_processed = pd.get_dummies(X_train_processed)
    X_test_processed = pd.get_dummies(X_test_processed)
    # If testing data don't have the feature, fill it with the mean of the training data
    X_test_processed = X_test_processed.reindex(
        columns=X_train_processed.columns, fill_value=0)

    # Train models
    full_model_rmse = full_model(X_train_processed, y_train_processed,
                                 X_test_processed, y_test_processed)
    ridge_model_rmse = 0
    lasso_model_rmse = 0
    elasticnet_model_rmse = 0
    randomforest_model_rmse = 0
    xgboost_model_rmse = 0

    # Conclusion
    print("Current fold target: ", target_fold_dir)
    print(f"Full model score: {full_model_rmse:.5f} RMSE")
    print(f"Lasso score with optimal alpha: {lasso_model_rmse:.5f} RMSE")
    print(f"Ridge score with optimal alpha: {ridge_model_rmse:.5f} RMSE")
    print(f"Elasticnet score: {elasticnet_model_rmse:.5f} RMSE")
    print(f"Random Forest score: {randomforest_model_rmse:.5f} RMSE")
    print(f"Boosting Tree score: {xgboost_model_rmse:.5f} RMSE")
