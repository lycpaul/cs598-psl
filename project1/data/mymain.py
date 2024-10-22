import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from xgboost import XGBRegressor

from scipy import stats
from scipy.stats import skew

import warnings
warnings.filterwarnings('ignore')

# Set random seed
seed_val = 1160
np.random.seed(seed_val)


def load_dataframe(target_fold_dir: str = ".") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # import the given fold data
    X_train_full = pd.read_csv(f'{target_fold_dir}/train.csv')
    X_train = X_train_full.iloc[:, 1:-1]
    y_train = np.log1p(X_train_full.iloc[:, -1])

    # import the test data
    X_test = pd.read_csv(f'{target_fold_dir}/test.csv').iloc[:, 1:]
    X_test_id = pd.read_csv(f'{target_fold_dir}/test.csv').iloc[:, 0]

    return X_train, y_train, X_test, X_test_id


def transform_training_target(X_train: pd.DataFrame, y_train: pd.Series, std_threshold: float = 3) -> tuple[pd.DataFrame, pd.Series]:
    # Drop the rows with extreme price larger
    mean = y_train.mean()
    std = y_train.std()
    rows_to_drop = y_train[(y_train > mean + std_threshold * std) |
                           (y_train < mean - std_threshold * std)].index
    # print(f'Drop {len(rows_to_drop)} rows')
    X_train = X_train.drop(index=rows_to_drop)
    y_train = y_train.drop(index=rows_to_drop)
    return X_train, y_train


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

    # suggested variable removal
    remove_cols = ['Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating',
                   'Pool_QC', 'Misc_Val', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude', 'Latitude']
    df = df.drop(columns=remove_cols)
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
    return df, skewed_feats


def delete_outliers(X: pd.DataFrame, y: pd.Series, feature: str, threshold: float = 5) -> pd.DataFrame:
    z_scores = np.abs(stats.zscore(X[feature]))
    outliers = X[z_scores > threshold]
    # print(f"Dropping {len(outliers)} outliers in {feature}")
    return X.drop(outliers.index), y.drop(outliers.index)


def xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> pd.Series:
    # tune by optuna
    xgb_params = {'objective': 'reg:squarederror',
                  'max_depth': 4,
                  'learning_rate': 0.009469809175065902,
                  'n_estimators': 4600,
                  'min_child_weight': 3,
                  'subsample': 0.4071521943355695,
                  'colsample_bytree': 0.788589645736069,
                  'reg_alpha': 0.005345576307828712,
                  'reg_lambda': 0.00019771013593545314,
                  'random_state': seed_val}
    xgb_tuned = XGBRegressor(**xgb_params)
    xgb_tuned.fit(X_train, y_train)
    y_pred = xgb_tuned.predict(X_test)
    return y_pred


def encode_categorical_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(
        include=['object', 'category']).columns

    # Perform dummy encoding on training and testing data
    X_train_encoded = pd.get_dummies(
        X_train, columns=categorical_cols, drop_first=False)
    X_test_encoded = pd.get_dummies(
        X_test, columns=categorical_cols, drop_first=False)

    # Align the training and testing dataframes by the dummy variables
    X_train_encoded, X_test_encoded = X_train_encoded.align(
        X_test_encoded, join='left', axis=1, fill_value=0)

    return X_train_encoded, X_test_encoded


def winsorize_features(X_train: pd.DataFrame, X_test: pd.DataFrame, features: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    for feature in features:
        M = X_train[feature].quantile(0.95)
        X_train[feature] = np.where(X_train[feature] > M, M, X_train[feature])
        X_test[feature] = np.where(X_test[feature] > M, M, X_test[feature])
    return X_train, X_test


def main(target_fold_dir: str) -> None:
    # Load the dataframes
    X_train, y_train, X_test, X_test_id = load_dataframe(target_fold_dir)

    # Make copies for processing
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    y_train_processed = y_train.copy()

    # Log transform the training target
    X_train_processed, y_train_processed = transform_training_target(
        X_train_processed, y_train_processed)

    # Winsorize specified features
    features_to_winsorize = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF",
                             "Total_Bsmt_SF", "Second_Flr_SF", "First_Flr_SF", "Gr_Liv_Area", "Garage_Area",
                             "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch",
                             "Screen_Porch", "Misc_Val"]
    X_train_processed, X_test_processed = winsorize_features(
        X_train_processed, X_test_processed, features_to_winsorize)

    # Clean the data
    X_train_processed = clean(X_train_processed)
    X_test_processed = clean(X_test_processed)

    # Preprocess numerical features
    X_train_processed, skewed_feats = process_numeric_features(
        X_train_processed)
    # Remarks: apply the same transformation (on training data) to the test data
    X_test_processed[skewed_feats] = np.log1p(X_test_processed[skewed_feats])

    # Delete outliers in linear numerical features
    numeric_feats = X_train_processed.dtypes[X_train_processed.dtypes != "object"].index
    for feature in numeric_feats:
        if feature in X_train_processed.columns:
            X_train_processed, y_train_processed = delete_outliers(
                X_train_processed, y_train_processed, feature)

    # Preprocess categorical features
    # Encode categorical features using dummy encoding with K dummies
    X_train_processed, X_test_processed = encode_categorical_features(
        X_train_processed, X_test_processed)

    #########################################
    ###### Train models

    # Ridge model
    ridge_alphas = np.logspace(-1, 3, 100)
    ridge_model = make_pipeline(RobustScaler(), RidgeCV(
        alphas=ridge_alphas, cv=5, scoring='neg_root_mean_squared_error'))
    ridge_model.fit(X_train_processed, y_train_processed)
    
    # parameters tuned by optuna
    xgb_params = {'objective': 'reg:squarederror',
                  'max_depth': 4,
                  'learning_rate': 0.009469809175065902,
                  'n_estimators': 4600,
                  'min_child_weight': 3,
                  'subsample': 0.4071521943355695,
                  'colsample_bytree': 0.788589645736069,
                  'reg_alpha': 0.005345576307828712,
                  'reg_lambda': 0.00019771013593545314,
                  'random_state': seed_val}
    xgboost_model = XGBRegressor(**xgb_params)
    xgboost_model.fit(X_train_processed, y_train_processed)

    #########################################
    ###### Make predictions
    y_pred_ridge = ridge_model.predict(X_test_processed)
    y_pred_xgboost = xgboost_model.predict(X_test_processed)
    
    # scale up the predictions
    y_pred_ridge = np.expm1(y_pred_ridge)
    y_pred_xgboost = np.expm1(y_pred_xgboost)
    
    # save the predictions as csv, using the PID in test index
    pd.DataFrame({'PID': X_test_id, 'Sale_Price': y_pred_ridge}).to_csv(
        f'{target_fold_dir}/mysubmission1.csv', index=False)
    pd.DataFrame({'PID': X_test_id, 'Sale_Price': y_pred_xgboost}).to_csv(
        f'{target_fold_dir}/mysubmission2.csv', index=False)


if __name__ == '__main__':
    # Train one fold
    main(f'project1/data/fold1')

    # Train all folds
    # for fold in range(1, 11):
    #     print(f"#### Fold {fold}: ####")
    #     main(f'project1/data/fold{fold}')
