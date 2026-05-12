import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from perceptron import Perceptron
from data_utils import generate_linear_separable, generate_xor, generate_circle, standardize, split_data


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_recall_f1(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def roc_curve_manual(y_true, y_scores):
    thresholds = np.sort(y_scores)[::-1]
    tpr, fpr = [], []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

    return np.array(fpr), np.array(tpr), thresholds


def auc_manual(fpr, tpr):
    return np.trapezoid(tpr, fpr)


def plot_loss(loss_history, title='Loss', save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history['train'], label='train')
    plt.plot(loss_history['val'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_decision_boundary(model, X, y, title='Decision Boundary', save_path=None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def run_baseline(X_train, y_train, X_test, y_test, output_dir='results'):
    model = Perceptron(n_features=2)
    model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)

    plot_loss(model.loss_history, title='Baseline Loss', save_path=f'{output_dir}/baseline_loss.png')

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = accuracy(y_train, train_pred)
    test_acc = accuracy(y_test, test_pred)
    print(f'Baseline: train accuracy = {train_acc:.4f}, test accuracy = {test_acc:.4f}')

    plot_decision_boundary(model, X_train, y_train, 'Baseline Decision Boundary (Train)', save_path=f'{output_dir}/baseline_boundary_train.png')
    plot_decision_boundary(model, X_test, y_test, 'Baseline Decision Boundary (Test)', save_path=f'{output_dir}/baseline_boundary_test.png')
    
    return model


def run_lr_experiment(X_train, y_train, X_test, y_test, lr_values=[0.001, 0.01, 0.5, 1.0], output_dir='results'):
    results = {}
    plt.figure(figsize=(12, 4))

    for idx, lr in enumerate(lr_values):
        model = Perceptron(n_features=2)
        model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=lr, batch_size=32)
        train_acc = accuracy(y_train, model.predict(X_train))
        test_acc = accuracy(y_test, model.predict(X_test))
        results[lr] = {'train_acc': train_acc, 'test_acc': test_acc}

        plt.subplot(1, 4, idx+1)
        plt.plot(model.loss_history['train'], label='train')
        plt.plot(model.loss_history['val'], label='val')
        plt.title(f'lr = {lr}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/lr_experiment.png', dpi=150, bbox_inches='tight')
    plt.show()

    print('\nLearning Rate Experiment Results:')
    print('lr\tTrain Acc\tTest Acc')

    for lr, res in results.items():
        print(f'{lr}\t{res["train_acc"]:.4f}\t\t{res["test_acc"]:.4f}')


def run_batch_experiment(X_train, y_train, X_test, y_test, batch_sizes=[1, 16, 64, 256], output_dir='results'):
    results = {}
    plt.figure(figsize=(12, 4))

    for idx, bs in enumerate(batch_sizes):
        model = Perceptron(n_features=2)
        model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=bs)
        train_acc = accuracy(y_train, model.predict(X_train))
        test_acc = accuracy(y_test, model.predict(X_test))
        results[bs] = {'train_acc': train_acc, 'test_acc': test_acc}

        plt.subplot(1, 4, idx+1)
        plt.plot(model.loss_history['train'], label='train')
        plt.plot(model.loss_history['val'], label='val')
        plt.title(f'batch_size = {bs}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/batch_experiment.png', dpi=150, bbox_inches='tight')
    plt.show()

    print('\nBatch Size Experiment Results:')
    print('batch\tTrain Acc\tTest Acc')

    for bs, res in results.items():
        print(f'{bs}\t{res["train_acc"]:.4f}\t\t{res["test_acc"]:.4f}')


def run_init_experiment(X_train, y_train, X_test, y_test, output_dir='results'):
    init_methods = {'zeros': lambda: (np.zeros((2, 1)), 0.0), 'small_random': lambda: (np.random.randn(2, 1) * 0.01, 0.0), 'large_random': lambda: (np.random.randn(2, 1) * np.sqrt(10), 0.0)}
    results = {}
    plt.figure(figsize=(12, 4))

    for idx, (name, init_fn) in enumerate(init_methods.items()):
        model = Perceptron(n_features=2)
        w, b = init_fn()
        model.w = w.copy()
        model.b = b
        model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
        train_acc = accuracy(y_train, model.predict(X_train))
        test_acc = accuracy(y_test, model.predict(X_test))
        results[name] = {'train_acc': train_acc, 'test_acc': test_acc}

        plt.subplot(1, 3, idx+1)
        plt.plot(model.loss_history['train'], label='train')
        plt.plot(model.loss_history['val'], label='val')
        plt.title(f'init: {name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/init_experiment.png', dpi=150, bbox_inches='tight')
    plt.show()

    print('\nWeight Initialization Experiment Results:')
    print('init\t\tTrain Acc\tTest Acc')

    for name, res in results.items():
        print(f'{name}\t\t{res["train_acc"]:.4f}\t\t{res["test_acc"]:.4f}')


def run_custom_data_experiments(output_dir='results'):
    print("\n=== Custom Data Experiments ===")
    datasets = {'linear_separable': generate_linear_separable(n_samples=200), 'xor': generate_xor(n_samples=200), 'circle': generate_circle(n_samples=200, radius=1.0, noise=0.0)}

    for name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
        X_train_scaled, X_test_scaled = standardize(X_train, X_test)

        model = Perceptron(n_features=2)
        model.fit(X_train_scaled, y_train, X_test_scaled, y_test, epochs=100, lr=0.1, batch_size=32)
        train_acc = accuracy(y_train, model.predict(X_train_scaled))
        test_acc = accuracy(y_test, model.predict(X_test_scaled))

        print(f'{name}: train={train_acc:.4f}, test={test_acc:.4f}')
        plot_decision_boundary(model, X_test_scaled, y_test, f'Decision Boundary: {name}', save_path=f'{output_dir}/custom_{name}_boundary.png')


def run_loss_comparison(X_train, y_train, X_test, y_test, output_dir='results'):
    print("\n=== Loss Comparison: Cross-Entropy vs Hinge ===")
    models = {'cross_entropy': Perceptron(n_features=2, loss_type='cross_entropy'), 'hinge': Perceptron(n_features=2, loss_type='hinge')}
    plt.figure(figsize=(10, 5))

    for name, model in models.items():
        model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
        train_acc = accuracy(y_train, model.predict(X_train))
        test_acc = accuracy(y_test, model.predict(X_test))
        print(f'{name}: train={train_acc:.4f}, test={test_acc:.4f}')
        plt.plot(model.loss_history['train'], label=f'{name} train')
        plt.plot(model.loss_history['val'], label=f'{name} val', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Loss Curves: Cross-Entropy vs Hinge')
    plt.savefig(f'{output_dir}/loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n=== L2 Regularization Experiment ===")

    lambdas = [0.0, 0.01, 0.1, 1.0]
    plt.figure(figsize=(12, 4))

    for idx, lam in enumerate(lambdas):
        model = Perceptron(n_features=2, loss_type='cross_entropy', lambda_l2=lam)
        model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
        train_acc = accuracy(y_train, model.predict(X_train))
        test_acc = accuracy(y_test, model.predict(X_test))
        print(f'lambda={lam}: train={train_acc:.4f}, test={test_acc:.4f}, ||w||={np.linalg.norm(model.w):.4f}')
        plt.subplot(1, 4, idx+1)
        plt.plot(model.loss_history['train'], label='train')
        plt.plot(model.loss_history['val'], label='val')
        plt.title(f'lambda={lam}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/l2_regularization.png', dpi=150, bbox_inches='tight')
    plt.show()


def run_metrics_analysis(model, X_test, y_test, output_dir='results'):
    print("\n=== Additional Metrics ===")
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)
    precision, recall, f1 = precision_recall_f1(y_test, y_pred)
    print(f'Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')

    fpr, tpr, _ = roc_curve_manual(y_test, y_scores)
    roc_auc = auc_manual(fpr, tpr)
    print(f'ROC-AUC={roc_auc:.4f}')

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    errors = y_test != y_pred
    if np.any(errors):
        plt.figure(figsize=(8, 6))
        correct = ~errors
        plt.scatter(X_test[correct, 0], X_test[correct, 1], c=y_test[correct], edgecolors='k', cmap='coolwarm', marker='o', label='Correct')
        plt.scatter(X_test[errors, 0], X_test[errors, 1], c=y_test[errors], cmap='coolwarm', marker='x', s=100, linewidth=2, label='Error')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Error Analysis')
        plt.legend()
        plt.savefig(f'{output_dir}/error_points.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print("No errors on test set.")


def run_momentum_experiment(X_train, y_train, X_test, y_test, betas=[0.0, 0.5, 0.9, 0.99], output_dir='results'):
    print("\n=== Momentum Experiment ===")
    results = {}
    plt.figure(figsize=(12, 4))

    for idx, beta in enumerate(betas):
        model = Perceptron(n_features=2)
        model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32, momentum=beta)
        train_acc = accuracy(y_train, model.predict(X_train))
        test_acc = accuracy(y_test, model.predict(X_test))
        results[beta] = {'train_acc': train_acc, 'test_acc': test_acc}

        plt.subplot(1, 4, idx+1)
        plt.plot(model.loss_history['train'], label='train')
        plt.plot(model.loss_history['val'], label='val')
        plt.title(f'momentum beta={beta}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/momentum_experiment.png', dpi=150, bbox_inches='tight')
    plt.show()

    print('beta\tTrain Acc\tTest Acc')
    for beta, res in results.items():
        print(f'{beta}\t{res["train_acc"]:.4f}\t\t{res["test_acc"]:.4f}')


def run_cross_validation(X_train, y_train, output_dir='results'):
    print("\n=== 5-fold Cross-Validation ===")
    lr_candidates = [0.01, 0.1, 0.5]
    batch_candidates = [16, 32, 64]
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = -np.inf
    best_params = None
    results_table = []

    for lr in lr_candidates:
        for bs in batch_candidates:
            fold_scores = []
            for train_idx, val_idx in kfold.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                model = Perceptron(n_features=2)
                model.fit(X_tr, y_tr, X_val, y_val, epochs=50, lr=lr, batch_size=bs)
                val_pred = model.predict(X_val)
                acc = accuracy(y_val, val_pred)
                fold_scores.append(acc)
            mean_acc = np.mean(fold_scores)
            std_acc = np.std(fold_scores)
            results_table.append((lr, bs, mean_acc, std_acc))
            print(f'lr={lr}, bs={bs}: mean accuracy={mean_acc:.4f} ± {std_acc:.4f}')

            if mean_acc > best_score:
                best_score = mean_acc
                best_params = (lr, bs)

    print(f'\nBest parameters: lr={best_params[0]}, batch_size={best_params[1]}')
    
    best_model = Perceptron(n_features=2)
    best_model.fit(X_train, y_train, X_train, y_train, epochs=100, lr=best_params[0], batch_size=best_params[1])
    
    return best_model, best_params, results_table