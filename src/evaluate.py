import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_and_preprocess_data
from quantum_kernel import compute_kernel_matrix

def evaluate_model(data_path, model_path="models/trained_svm.pkl"):
    """Loads a trained model and evaluates it on test data."""
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

    # Compute kernel matrix for testing
    K_test = compute_kernel_matrix(X_test, X_train)

    # Load model
    svm = joblib.load(model_path)

    # Make predictions
    y_pred = svm.predict(K_test)

    # Print evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Intrusion"], yticklabels=["Normal", "Intrusion"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate_model("../data/raw/dataset.csv")
