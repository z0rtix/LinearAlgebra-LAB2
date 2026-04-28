import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def generate_data(n_samples=500, n_features=2, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_informative=2,
        random_state=random_state,
        n_clusters_per_class=1
    )
    return X, y

def generate_linear_separable(n_samples=200, centers=[[-1,-1],[1,1]], cov=[[1,0],[0,1]], random_state=42):
    np.random.seed(random_state)
    half = n_samples // 2
    X0 = np.random.multivariate_normal(centers[0], cov, half)
    y0 = np.zeros(half)
    X1 = np.random.multivariate_normal(centers[1], cov, half)
    y1 = np.ones(half)
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    return X, y

def generate_xor(n_samples=200, scale=0.3, random_state=42):
    np.random.seed(random_state)
    centers = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])
    labels = np.array([0, 0, 1, 1])
    points_per_center = n_samples // 4
    X = []
    y = []
    for i in range(4):
        noise = np.random.randn(points_per_center, 2) * scale
        X.append(centers[i] + noise)
        y.append(np.full(points_per_center, labels[i]))
    return np.vstack(X), np.hstack(y)

def generate_circle(n_samples=200, radius=1.0, inside_class=1, noise=0.0, random_state=42):
    np.random.seed(random_state)
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    distances = np.random.uniform(0, radius*1.5, n_samples)
    X = np.column_stack([distances * np.cos(angles), distances * np.sin(angles)])
    y = (distances <= radius).astype(int)
    if inside_class == 0:
        y = 1 - y
    if noise > 0:
        flip = np.random.rand(n_samples) < noise
        y[flip] = 1 - y[flip]
    return X, y

def add_label_noise(y, noise_level=0.05, random_state=42):
    np.random.seed(random_state)
    y_noisy = y.copy()
    n = len(y)
    idx = np.random.choice(n, int(n * noise_level), replace=False)
    y_noisy[idx] = 1 - y_noisy[idx]
    return y_noisy

def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled

def split_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test