import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNetCV, RidgeCV, Ridge, LassoCV, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from xgboost import XGBRegressor

from scipy import stats
from scipy.stats import skew

import optuna
from optuna import Trial
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')


# Set random seed
seed_val = 1160
np.random.seed(seed_val)

# linear_numeric_feats = ['Lot_Frontage', 'Lot_Area', 'Year_Built', 'Year_Remod_Add',
#                         'Mas_Vnr_Area', 'Bsmt_Unf_SF', 'Total_Bsmt_SF', 'First_Flr_SF',
#                         'Second_Flr_SF', 'Gr_Liv_Area', 'Garage_Area', 'Wood_Deck_SF',
#                         'Open_Porch_SF', 'Latitude', 'Longitude']

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
    # print(f"Dropping categorical features with missing values: {train_null}")

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
    # print(f"Log transformed {len(skewed_feats)} skewed features")
    # print(f"The skewed features are {skewed_feats}")
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
        alphas=ridge_alphas, cv=5, scoring='neg_root_mean_squared_error'))

    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    train_rmse = rmse(y_train, y_pred_train)
    y_pred = model.predict(X_test)
    test_rmse = rmse(y_test, y_pred)
    return train_rmse, test_rmse


def lasso_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    lasso_alphas = np.logspace(-5, -2, 100)
    model = LassoCV(alphas=lasso_alphas, cv=5, random_state=seed_val)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    train_rmse = rmse(y_train, y_pred_train)
    y_pred = model.predict(X_test)
    test_rmse = rmse(y_test, y_pred)
    return train_rmse, test_rmse


def elasticnet_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    model = ElasticNetCV(alphas=np.logspace(-1, 1, 100), cv=5)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    train_rmse = rmse(y_train, y_pred_train)
    y_pred = model.predict(X_test)
    test_rmse = rmse(y_test, y_pred)
    return train_rmse, test_rmse


def xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
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
    y_pred_train = xgb_tuned.predict(X_train)
    train_rmse = rmse(y_train, y_pred_train)
    y_pred = xgb_tuned.predict(X_test)
    test_rmse = rmse(y_test, y_pred)
    return train_rmse, test_rmse


def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    # Suggest hyperparameters
    param = {
        'max_depth': trial.suggest_int('max_depth', 4, 7),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 2500, 5000, step=100),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.4, 0.6),
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


def encode_categorical_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encodes categorical features by creating K dummy variables for each categorical feature with K levels.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Encoded training and testing features.
    """
    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(
        include=['object', 'category']).columns
    # print(f"Categorical columns to encode: {categorical_cols}")

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
        # print(f"Winsorized {feature} at 95% quantile: {M}")
    return X_train, X_test


def main(target_fold_dir: str) -> None:
    # Load the dataframes
    X_train, y_train, X_test, y_test = load_dataframe(target_fold_dir)

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

    #################
    # Train models
    model_params = {
        'X_train': X_train_processed,
        'y_train': y_train_processed,
        'X_test': X_test_processed,
        'y_test': y_test
    }
    rmse_errors = {
        'full_model': ['train', 'test'],
        'ridge_model': ['train', 'test'],
        'lasso_model': ['train', 'test'],
        'elasticnet_model': ['train', 'test'],
        'xgboost_model': ['train', 'test']
    }

    # rmse_errors['full_model'] = full_model(**model_params)
    rmse_errors['ridge_model'] = ridge_model(**model_params)
    # rmse_errors['lasso_model'] = lasso_model(**model_params)
    # rmse_errors['elasticnet_model'] = elasticnet_model(**model_params)
    rmse_errors['xgboost_model'] = xgboost_model(**model_params)

    # Hyperparameter Tuning
    # print("Xgboost rmse before tuning: ", rmse_errors['xgboost_model'])
    # print("Starting hyperparameter tuning for XGBoost...")
    # tuned_params = tune_xgboost_params(
    #     X_train_processed, y_train_processed, n_trials=50)
    # print(f"Tuned params: {tuned_params}")
    # xgb_tuned = XGBRegressor(**tuned_params)
    # xgb_tuned.fit(X_train_processed, y_train_processed)
    # y_pred_xgb_tuned = xgb_tuned.predict(X_test_processed)
    # xgboost_model_rmse = rmse(y_test, y_pred_xgb_tuned)

    # Conclusion
    # print("Training Error:")
    # print(f"Full model score: {rmse_errors['full_model'][0]:.5f} RMSE")
    # print(f"Lasso score with optimal alpha: {
    #       rmse_errors['lasso_model'][0]:.5f} RMSE")
    # print(f"Ridge score with optimal alpha: {
    #       rmse_errors['ridge_model'][0]:.5f} RMSE")
    # print(f"Elasticnet score: {rmse_errors['elasticnet_model'][0]:.5f} RMSE")
    # print(f"Boosting Tree score: {rmse_errors['xgboost_model'][0]:.5f} RMSE")

    # print("Test Error:")
    # print(f"Full model score: {rmse_errors['full_model'][1]:.5f} RMSE")
    # print(f"Lasso score with optimal alpha: {
    #   rmse_errors['lasso_model'][1]:.5f} RMSE")
    print(f"Ridge score with optimal alpha: {
          rmse_errors['ridge_model'][1]:.5f} RMSE")
    # print(f"Elasticnet score: {rmse_errors['elasticnet_model'][1]:.5f} RMSE")
    print(f"Boosting Tree score: {rmse_errors['xgboost_model'][1]:.5f} RMSE")

    return rmse_errors


if __name__ == '__main__':
    # Train one fold
    # main(f'project1/data/fold7')

    # Train all folds
    rmse_errors = {}
    for fold in range(1, 11):
        print(f"#### Fold {fold}: ####")
        rmse_errors[f'fold{fold}'] = main(f'project1/data/fold{fold}')
