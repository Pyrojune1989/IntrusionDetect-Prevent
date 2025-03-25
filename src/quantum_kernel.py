import pennylane as qml
import numpy as np

# Define quantum device
num_qubits = 4  # Adjust based on number of features
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def quantum_kernel(x1, x2):
    """Computes quantum kernel similarity between two input samples."""
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        qml.RY(x1[i], wires=i)
        qml.RY(x2[i], wires=i)
    
    return qml.probs(wires=range(num_qubits))

def compute_kernel_matrix(X1, X2):
    """Computes quantum kernel matrix."""
    kernel_matrix = np.zeros((len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            kernel_matrix[i, j] = np.sum(quantum_kernel(x1, x2))
    return kernel_matrix

if __name__ == "__main__":
    X1_sample = np.random.rand(10, num_qubits)
    X2_sample = np.random.rand(10, num_qubits)
    kernel_matrix = compute_kernel_matrix(X1_sample, X2_sample)
    print("Quantum Kernel Matrix Computed:", kernel_matrix.shape)
