import os
from data_utils import generate_data, standardize, split_data
from experiments import (
    run_baseline, run_lr_experiment, run_batch_experiment, run_init_experiment,
    run_custom_data_experiments, run_loss_comparison, run_metrics_analysis,
    run_momentum_experiment, run_cross_validation, accuracy
)

OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

X, y = generate_data()
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_scaled, X_test_scaled = standardize(X_train, X_test)

model = run_baseline(X_train_scaled, y_train, X_test_scaled, y_test, output_dir=OUTPUT_DIR)
run_lr_experiment(X_train_scaled, y_train, X_test_scaled, y_test, output_dir=OUTPUT_DIR)
run_batch_experiment(X_train_scaled, y_train, X_test_scaled, y_test, output_dir=OUTPUT_DIR)
run_init_experiment(X_train_scaled, y_train, X_test_scaled, y_test, output_dir=OUTPUT_DIR)

run_custom_data_experiments(output_dir=OUTPUT_DIR)
run_loss_comparison(X_train_scaled, y_train, X_test_scaled, y_test, output_dir=OUTPUT_DIR)
run_metrics_analysis(model, X_test_scaled, y_test, output_dir=OUTPUT_DIR)
run_momentum_experiment(X_train_scaled, y_train, X_test_scaled, y_test, output_dir=OUTPUT_DIR)
best_model, best_params, _ = run_cross_validation(X_train_scaled, y_train, output_dir=OUTPUT_DIR)

print(f"\nFinal model from CV trained on all training data. Test accuracy = {accuracy(y_test, best_model.predict(X_test_scaled)):.4f}")