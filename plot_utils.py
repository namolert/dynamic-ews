import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, recall_score, accuracy_score, mean_squared_error, ConfusionMatrixDisplay, classification_report
import tensorflow as tf

def create_sequences(X, y, window=30):
    Xs, ys, indices = [], [], []
    for i in range(len(X) - window):
        Xs.append(X.iloc[i:(i + window)].values)
        ys.append(y.iloc[i + window])  # predict if crash occurs after window
        indices.append(X.index[i+window-1])
    return np.array(Xs), np.array(ys), np.array(indices)

def plot_feature_importances(model, X):
    importances = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # plt.figure(figsize=(10, 5))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.show()

def plot_feature_importances_grid(results):
    num_models = len(results)
    cols = 2
    rows = int(np.ceil(num_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()

    for idx, (n, result) in enumerate(sorted(results.items())):
        model = result['model']
        importances = model.feature_importances_
        feature_names = model.feature_names_in_

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        sns.barplot(
            x='Importance',
            y='Feature',
            data=importance_df,
            ax=axes[idx],
            palette='viridis'
        )
        axes[idx].set_title(f'{n}-Day Feature Importances')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('')

    # Turn off unused subplots
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def plot_logit_coefficients_grid(results):
    num_models = len(results)
    cols = 2
    rows = int(np.ceil(num_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()

    for idx, (n, result) in enumerate(sorted(results.items())):
        coefs_df = result['coefficients']

        sns.barplot(
            x='Coefficient',
            y='Feature',
            data=coefs_df,
            ax=axes[idx],
            palette='coolwarm'
        )
        axes[idx].set_title(f'{n}-Day Coefficients')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('')

    # Remove unused subplots
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_roc_curves_grid(results, X_test_dict, y_test_dict):
    from sklearn.metrics import roc_curve, auc
    num_models = len(results)
    cols = 2
    rows = int(np.ceil(num_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()

    for idx, (n, result) in enumerate(sorted(results.items())):
        model = result['model']
        X_test = X_test_dict[n]
        y_test = y_test_dict[n]

        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        ax = axes[idx]
        ax.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{n}-Day ROC Curve')
        ax.legend(loc='lower right')
        ax.grid()

    # Turn off unused subplots
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def plot_CNN_roc_curves_grid(results, X_test_dict, y_test_dict):
    from sklearn.metrics import roc_curve, auc
    num_models = len(results)
    cols = 2
    rows = int(np.ceil(num_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()

    for idx, (n, result) in enumerate(sorted(results.items())):
        model = result['model']
        X_test = X_test_dict[n]
        y_test = y_test_dict[n]

        y_proba = model.predict(X_test).flatten()

        # Skip ROC if test set is one-class
        if len(np.unique(y_test)) < 2:
            print(f"Skipping ROC for {n}-Day model (only one class in test set)")
            continue

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        ax = axes[idx]
        ax.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{n}-Day ROC Curve')
        ax.legend(loc='lower right')
        ax.grid()

    # Turn off unused subplots
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Crash', 'Crash'], yticklabels=['No Crash', 'Crash'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
def plot_confusion_matrix_grid(conf_matrices, titles, ncols=2):
    n = len(conf_matrices)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))

    # Make sure axes is always 2D array
    axes = np.atleast_2d(axes)

    for i in range(nrows * ncols):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

        if i < n:
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrices[i], display_labels=[0, 1])
            disp.plot(ax=ax, values_format='d')
            ax.set_title(titles[i])
        else:
            ax.axis('off')  # Hide unused subplot

    plt.tight_layout()
    plt.show()

def plot_crash_probabilities(df, target_col='crash_probability'):
    # plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df[target_col], label='Crash Probability', color='red')
    plt.axhline(0.5, linestyle='--', color='gray', label='Threshold = 0.5')  # Example threshold
    plt.title("Early Warning Signal: Crash Probability Over Time")
    plt.xlabel("Date")
    plt.ylabel("Predicted Crash Probability")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()


def plot_crash_probabilities_grid(results, market_sentiment_data, target_col='future_crash'):
    num_models = len(results)
    cols = 2
    rows = int(np.ceil(num_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()

    for idx, (n, result) in enumerate(sorted(results.items())):
        model = result['model']

        # Prepare features
        feature_cols = [
            f'{n}_day_market_volatility',
            f'{n}_day_sentiment_volatility',
            f'VaR_{n}',
            f'ES_{n}',
        ]

        df_model = market_sentiment_data.dropna(subset=feature_cols + [target_col]).copy()

        X_all = df_model[feature_cols]
        y_all = df_model[target_col]
        model.fit(X_all, y_all)
        df_model['crash_probability'] = model.predict_proba(X_all)[:, 1]

        ax = axes[idx]
        ax.plot(df_model['Date'], df_model['crash_probability'], label='Crash Probability', color='red')
        ax.axhline(0.5, linestyle='--', color='gray', label='Threshold = 0.5')
        ax.set_title(f'{n}-Day Crash Probability Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Predicted Probability')
        ax.legend()
        ax.grid()

    # Hide any unused subplots
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def plot_crash_probabilities_grid_CNN(results, market_sentiment_data_with_lags, lag=10, window_size=30):
    num_models = len(results)
    cols = 2
    rows = (num_models + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows*3), sharex=False)
    axes = axes.flatten()

    for idx, (n, result) in enumerate(sorted(results.items())):
        model = result['model']
        features = result['features']

        # Recreate full sequence to get full-length predictions
        df_model = market_sentiment_data_with_lags.dropna(subset=features + ['future_crash']).copy()
        X_raw = df_model[features].fillna(0)
        y_raw = df_model['future_crash']
        X_seq, y_seq, _ = create_sequences(X_raw, y_raw, window=window_size)

        crash_prob = model.predict(X_seq).flatten()
        df_plot = df_model.iloc[window_size:].copy()
        df_plot['crash_probability'] = crash_prob

        ax = axes[idx]
        ax.plot(df_plot['Date'], df_plot['crash_probability'], color='red', label='Crash Probability')
        ax.axhline(0.5, linestyle='--', color='gray', label='Threshold = 0.5')
        ax.set_title(f'{n}-Day Crash Probability')
        ax.set_xlabel('Date')
        ax.set_ylabel('Probability')
        ax.grid(True)
        ax.legend()

    # Remove any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def compute_integrated_gradients(model, input_sequence, baseline=None, steps=50):
    if baseline is None:
        baseline = np.zeros_like(input_sequence)

    # Generate interpolated inputs between baseline and input
    interpolated_inputs = [
        baseline + (float(i) / steps) * (input_sequence - baseline)
        for i in range(steps + 1)
    ]
    interpolated_inputs = np.array(interpolated_inputs)

    # Convert to Tensor and enable gradient tracking
    interpolated_inputs = tf.convert_to_tensor(interpolated_inputs, dtype=tf.float32)
    interpolated_inputs = tf.reshape(interpolated_inputs, (steps + 1, *input_sequence.shape))

    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        predictions = model(interpolated_inputs)
    
    # Compute gradients of output w.r.t inputs
    grads = tape.gradient(predictions, interpolated_inputs).numpy()

    # Average gradients across the path
    avg_grads = np.mean(grads[:-1], axis=0)

    # Integrated gradients
    integrated_grads = (input_sequence - baseline) * avg_grads
    return integrated_grads

def plot_ig_feature_attributions_grid(cnn_results, X_test_dict, window_size):
    num_models = len(cnn_results)
    cols = 2
    rows = int(np.ceil(num_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()

    for idx, (n, result) in enumerate(sorted(cnn_results.items())):
        model = result['model']
        feature_names = result['features']

        # Get one test sample
        X_sample = X_test_dict[n][0]  # Shape: (window, features)
        
        # Compute Integrated Gradients
        attributions = compute_integrated_gradients(model, X_sample)
        avg_attributions = np.mean(attributions, axis=0)

        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Attribution': avg_attributions
        }).sort_values(by='Attribution', ascending=False)

        # Plot
        sns.barplot(
            x='Attribution',
            y='Feature',
            data=importance_df,
            ax=axes[idx],
            palette='viridis'
        )
        axes[idx].set_title(f'{n}-Day IG Attributions')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('')

    # Turn off unused subplots
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def dynamic_threshold_calculate(y_test, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # Find threshold for best F1 or based on your risk tolerance
    f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]

    print(f"Best threshold: {best_thresh:.2f}, F1: {f1_scores[best_idx]:.2f}")

    return best_thresh

def evaluate_model(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Metrics
    sensitivity = recall_score(y_true, y_pred)  # True positive rate
    specificity = tn / (tn + fp)               # True negative rate
    misclassification_error = 1 - accuracy_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_proba))

    # Print results
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Misclassification Error: {misclassification_error:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Misclassification Error": misclassification_error,
        "RMSE": rmse
    }