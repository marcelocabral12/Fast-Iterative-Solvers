import numpy as np
import matplotlib.pyplot as plt
import time

def msr(data, x):
    n = int(data[0, 0])  # Convert size to an integer
    JM = data[1:, 0].astype(int) - 1  # Adjust for 0-based indexing
    VM = data[1:, 1]
    y = np.zeros(n)

    for i in range(n):
        y[i] = VM[i] * x[i]  # Main diagonal elements
        i1 = JM[i]
        i2 = JM[i + 1] - 1
        for j in range(i1, i2 + 1):
            y[i] += VM[j] * x[JM[j]]  # Calculating the lower half with CRS format
            y[JM[j]] += VM[j] * x[i]  # Calculating the upper half with CSC format
    return y


def lanczos(data, k, tol):
    n = int(data[0, 0])
    v = np.zeros((n, k+1))
    v[:, 0] = np.zeros(n)  # Initialize the first vector as zeros
    v[:, 1] = np.ones(n) / np.sqrt(n)  # Initialize the second vector as ones over sqrt(n)
    alph = np.zeros(k)
    bet = np.zeros(k)

    for i in range(1, k):  # Ensure i doesn't exceed min(k, n)
        w = msr(data, v[:, i]) - bet[i - 1] * v[:, i - 1]
        alph [i] = v[:, i].dot(w)
        w = w - alph [i] * v[:, i]
        bet[i] = np.linalg.norm(w)
        v[:, i + 1] = w / bet[i]

    # Creation of the matrix tridiagonal
    T = np.zeros((k, k))
    np.fill_diagonal(T, alph[1:k])
    np.fill_diagonal(T[:, 1:], bet[1:k])
    np.fill_diagonal(T[1:, :], bet[1:k])

    e = [1]
    j = 1
    q = np.ones(k) / np.sqrt(k)
    lambda_old = 0
    while e[-1] > tol:
        z = np.dot(T, q)
        q = z / np.linalg.norm(z)
        lambda_j = q.T.dot(np.dot(T, q))
        e.append(abs(lambda_j - lambda_old))
        print(f"Iteration {j}: Eigenvalue = {lambda_j}")
        j += 1
        lambda_old = lambda_j
        print(e[-1])

    return np.array(e[1:])


# Read data from file, skipping the first line
with open('cg_test_msr.txt') as file:
    lines = file.readlines()[1:]
    data = np.array([[float(val) for val in line.split()] for line in lines])

# Lanczos Iteration
tol_lanczos = 1e-6
m = 75
start = time.time()
e_lanczos = lanczos(data, m, tol_lanczos)
end = time.time()
print(f"Lanczos Iteration Time: {end - start} seconds")


# Plot
itnum = np.arange(1, len(e_lanczos)+1)  # +1 to include the final iteration
error = e_lanczos
plt.figure()
plt.semilogy(itnum, error, linewidth=0.5, color="blue")
plt.xlabel('Number of Iterations', fontsize=14)
plt.ylabel('$|\lambda_{(k)} - \lambda_{(k-1)}|$', fontsize=14)
plt.show()

# Calculate the runtime for each iteration
# runtimes = np.linspace(0, end - start, len(e))

# Plot error over runtime
# plt.figure()
# plt.semilogy(runtimes, e, linewidth=0.5, color="blue")
# plt.xlabel('Runtime (seconds)', fontsize=14)
# plt.ylabel('$|\lambda_{(k)} - \lambda_{(k-1)}|$', fontsize=14)
# plt.show()