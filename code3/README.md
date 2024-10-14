### Task 3


original method
```python
num_products = 811
num_weeks = 52
num_knots = 10

# Step 1: Create the design matrix F for each time series
F = np.zeros((num_products, num_weeks, num_knots))
for i in range(num_products):
    F[i] = ns(X_centered[i], df=num_knots, include_intercept=True)

# Step 2: Compute B for each time series
B = np.zeros((num_products, num_knots))
for i in range(num_products):
    B_t = np.linalg.pinv(F[i].T @ F[i]) @ F[i].T @ X_centered[i].T
    B[i] = B_t.T  # Transpose to get the 811-by-9 matrix

# Step 3: Drop the first column of B
B = B[:, 1:]
```