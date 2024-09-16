import numpy as np

## Three type of Bayes Rule implementation

sigma_sd = 1/np.sqrt(5)

# 1. Naive Bayes Rule with for loop
def bayes_predict(X, m0, m1):
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        exp_m1 = np.sum(
            np.exp(-np.sum((X[i] - m1)**2, axis=1) / (2 * sigma_sd**2)))
        exp_m0 = np.sum(
            np.exp(-np.sum((X[i] - m0)**2, axis=1) / (2 * sigma_sd**2)))
        predictions[i] = (exp_m1 >= exp_m0).astype(int)
    return predictions


def bayes_predict_vectorized1(X, m0, m1):
    # Compute distances for all samples at once
    dist_m1 = np.sum((X[:, np.newaxis, :] - m1[np.newaxis, :, :])**2, axis=2)
    dist_m0 = np.sum((X[:, np.newaxis, :] - m0[np.newaxis, :, :])**2, axis=2)

    # Compute exponentials
    exp_m1 = np.sum(np.exp(-dist_m1 / (2 * sigma_sd**2)), axis=1)
    exp_m0 = np.sum(np.exp(-dist_m0 / (2 * sigma_sd**2)), axis=1)

    # Make predictions
    return (exp_m1 >= exp_m0).astype(int)


def bayes_predict_vectorized2(X, m0, m1):
    # # Compute distances for all samples at once
    # dist_m1 = np.sum((X[:, np.newaxis, :] - m1[np.newaxis, :, :])**2, axis=2)
    # dist_m0 = np.sum((X[:, np.newaxis, :] - m0[np.newaxis, :, :])**2, axis=2)

    # Compute the Euclidean distance with vectorization
    x2 = np.sum(X**2, axis=1, keepdims=True)
    m1_sq = np.sum(m1**2, axis=1)
    m0_sq = np.sum(m0**2, axis=1)
    x_m1 = np.matmul(X, m1.T)
    x_m0 = np.matmul(X, m0.T)
    x2 = x2.reshape(-1, 1)
    dist_m1 = x2 + m1_sq - 2 * x_m1
    dist_m0 = x2 + m0_sq - 2 * x_m0

    # Compute exponentials
    exp_m1 = np.sum(np.exp(-dist_m1 / (2 * sigma_sd**2)), axis=1)
    exp_m0 = np.sum(np.exp(-dist_m0 / (2 * sigma_sd**2)), axis=1)

    # Make predictions
    return (exp_m1 >= exp_m0).astype(int)
