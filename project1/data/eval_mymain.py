import numpy as np
import pandas as pd


def load_dataframe(target_fold_dir: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    # import the ground truth test data with log transformation
    y_test = np.log1p(pd.read_csv(f'{target_fold_dir}/test_y.csv').iloc[:, -1])
    y_pred_ridge = np.log1p(pd.read_csv(
        f'{target_fold_dir}/mysubmission1.csv').iloc[:, -1])
    y_pred_xgb = np.log1p(pd.read_csv(
        f'{target_fold_dir}/mysubmission2.csv').iloc[:, -1])

    return y_test, y_pred_ridge, y_pred_xgb


def rmse(y_test: pd.Series, y_pred: pd.Series) -> float:
    return np.sqrt(np.mean((y_pred - y_test) ** 2))


if __name__ == "__main__":
    # evaluate all folds
    for fold in range(1, 11):
        target_fold_dir = f"project1/data/fold{fold}"
        y_test, y_pred_ridge, y_pred_xgb = load_dataframe(target_fold_dir)
        print(f"Target fold directory: {target_fold_dir}")
        print(f"Ridge RMSE: {rmse(y_test, y_pred_ridge):.6f}")
        print(f"XGB   RMSE: {rmse(y_test, y_pred_xgb):.6f}")
