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

def powerIt(data, tol):
    e = [1]
    k = 1
    n = int(data[0, 0])
    q = np.ones(n) / np.sqrt(n)
    lambda_old = 0
    while e[-1] > tol:
        z = msr(data, q)
        q = z / np.linalg.norm(z)
        lambda_k = q.T.dot(msr(data, q))
        e.append(abs(lambda_k - lambda_old))
        print(f"Iteration {k}: Eigenvalue = {lambda_k}")
        k += 1  # Increment k here to update the iteration count
        lambda_old = lambda_k
        print(e[-1])

    return np.array(e[1:])


# Read data from file, skipping the first line
with open('cg_test_msr.txt') as file:
    lines = file.readlines()[1:]
    data = np.array([[float(val) for val in line.split()] for line in lines])


#Power Iteration
tol_power = 1e-10
start=time.time()
e = powerIt(data, tol_power)
end = time.time()
print (end-start)


# Plot
# itnum = np.arange(1, len(e)+1)  # +1 to include the final iteration
# error = e
# plt.figure()
# plt.semilogy(itnum, error, linewidth=0.5, color="blue")
# plt.xlabel('Number of Iterations', fontsize=14)
# plt.ylabel('$|\lambda_{(k)} - \lambda_{(k-1)}|$', fontsize=14)
# plt.show()

# Calculate the runtime for each iteration
runtimes = np.linspace(0, end - start, len(e))

# Plot error over runtime
# plt.figure()
# plt.semilogy(runtimes, e, linewidth=0.5, color="blue")
# plt.xlabel('Runtime (seconds)', fontsize=14)
# plt.ylabel('$|\lambda_{(k)} - \lambda_{(k-1)}|$', fontsize=14)
# plt.show()