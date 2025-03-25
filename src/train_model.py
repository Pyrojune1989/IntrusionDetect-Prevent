import numpy as np
from sklearn.svm import SVC
import joblib
from preprocessing import load_and_preprocess_data
from quantum_kernel import compute_kernel_matrix

def train_quantum_svm(data_path, model_path="models/trained_svm.pkl"):
    """Trains a quantum SVM using the quantum kernel method."""
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

    # Compute quantum kernel matrices
    K_train = compute_kernel_matrix(X_train, X_train)
    K_test = compute_kernel_matrix(X_test, X_train)

    # Train SVM
    svm = SVC(kernel="precomputed")
    svm.fit(K_train, y_train)

    # Save model
    joblib.dump(svm, model_path)
    print(f"Model trained and saved to {model_path}")

    return svm, K_test, y_test

if __name__ == "__main__":
    train_quantum_svm("../data/raw/dataset.csv")
