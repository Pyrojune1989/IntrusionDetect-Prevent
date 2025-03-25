import pennylane as qml
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

# Define the number of qubits
num_qubits = 20  # Reduced to fit within the system's memory capacity
dev = qml.device("lightning.qubit", wires=num_qubits)  # PennyLane's optimized simulator

# Quantum embedding (feature map)
def feature_map(x):
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        qml.RY(x[i], wires=i)  # Encoding data into rotation gates

# Quantum kernel circuit
@qml.qnode(dev)
def quantum_kernel(x1, x2):
    feature_map(x1)
    qml.adjoint(feature_map)(x2)  # Compute inner product
    return qml.expval(qml.Identity(0))  # Compute the overlap using the identity operator

def compute_kernel_entry(i, j, X1, X2):
    return quantum_kernel(X1[i], X2[j])

# Compute the quantum kernel matrix for training data
def kernel_matrix_train(X_train):
    N = len(X_train)
    K = np.zeros((N, N))

    def compute_row(i):
        for j in range(N):
            K[i, j] = compute_kernel_entry(i, j, X_train, X_train)

    with ThreadPoolExecutor() as executor:
        executor.map(compute_row, range(N))

    return K

# Compute the quantum kernel matrix for test data
def kernel_matrix_test(X_test, X_train):
    N_test = len(X_test)
    N_train = len(X_train)
    K = np.zeros((N_test, N_train))

    def compute_row(i):
        for j in range(N_train):
            K[i, j] = compute_kernel_entry(i, j, X_test, X_train)

    with ThreadPoolExecutor() as executor:
        executor.map(compute_row, range(N_test))

    return K

# Example dataset (Replace with real network intrusion data)
X = np.random.rand(50, num_qubits)  # Reduced to 50 samples
y = np.random.choice([0, 1], size=(50,))  # Binary labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute quantum kernel matrices
K_train = kernel_matrix_train(X_train)
K_test = kernel_matrix_test(X_test, X_train)

# Train classical SVM with quantum kernel
svm = SVC(kernel="precomputed")
svm.fit(K_train, y_train)

# Predict and evaluate
y_pred = svm.predict(K_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Quantum Kernel SVM Accuracy: {accuracy:.2f}")
