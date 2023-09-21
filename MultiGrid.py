#Marcelo Cabral Liberato de Mattos Filho 
#Enrollment number: 404864
import array as arr
import math
import matplotlib.pyplot as plt
import numpy as np
import time
#Go to line 182 to change the size of the mesh

# Here I start with the Poisson equation
def initial(n):
    N = int(math.pow(2, n) + 1)
    h = 1 / (N - 1)  # h of equation
    u = [[0 for i in range(0, N)] for j in range(0, N)]

    # f equation
    f = [[0 for i in range(0, N)] for j in range(0, N)]
    for i in range(0, N):
        for j in range(0, N):
            f[i][j] = (8 * math.pi * math.pi * math.sin(2 *
                       math.pi * i * h) * math.sin(2 * math.pi * j * h))

    return f, u


def substraction(u1, u2):
    N = len(u1)
    result = [[0 for i in range(0, N)] for j in range(0, N)]

    for i in range(0, N):
        for j in range(0, N):
            result[i][j] = u1[i][j] - u2[i][j]

    return result


def max_norm(m):
    N = len(m)
    norm = 0
    for i in range(0, N):
        for j in range(0, N):
            if norm < abs(m[i][j]):
                norm = abs(m[i][j])

    return norm


# gauss-seidel relaxation-smoother
def GSsmoother(u, f, n):
    N = len(u)
    h = 1 / (N - 1)
    for k in range(0, n):
        for i in range(1, N - 1):
            for j in range(1, N - 1):

                u[i][j] = 0.25 * (h * h * f[i][j] + u[i][j - 1] + u[i][j + 1] + u[i - 1][j] + u[i + 1][j])

    return u


def residual(u, f):
    N = len(u)
    h = 1 / (N - 1)
    result = [[0 for i in range(0, N)] for j in range(0, N)]

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            result[i][j] = f[i][j] + (u[i][j - 1] + u[i][j + 1] +u[i - 1][j] + u[i + 1][j] - 4 * u[i][j]) / (h * h)

    return result


def RESTR(u_h):
    Nc = int(len(u_h) * 0.5)
    result = [[0 for i in range(0, Nc + 1)] for j in range(0, Nc + 1)]

    for i in range(1, Nc):
        ii = 2 * i
        for j in range(1, Nc):
            jj = 2 * j

            result[i][j] = (u_h[ii - 1][jj - 1]+ 2 * u_h[ii][jj - 1]+ u_h[ii + 1][jj - 1]+ 2 * u_h[ii - 1][jj]+ 4 * u_h[ii][jj]
                + 2 * u_h[ii + 1][jj]+ u_h[ii - 1][jj + 1]+ 2 * u_h[ii][jj + 1]+ u_h[ii + 1][jj + 1]) * 0.0625
    
    return result


def PROLONG(u_2h):
    N = len(u_2h) - 1

    result = [[0 for i in range(0, 2 * N + 1)] for j in range(0, 2 * N + 1)]

    for i in range(1, N):
        ii = 2 * i
        for j in range(1, N):
            jj = 2 * j

            result[ii - 1][jj - 1] += u_2h[i][j] * 0.25
            result[ii - 1][jj + 1] += u_2h[i][j] * 0.25
            result[ii + 1][jj - 1] += u_2h[i][j] * 0.25
            result[ii + 1][jj + 1] += u_2h[i][j] * 0.25

            result[ii - 1][jj] += u_2h[i][j] * 0.5
            result[ii + 1][jj] += u_2h[i][j] * 0.5
            result[ii][jj - 1] += u_2h[i][j] * 0.5
            result[ii][jj + 1] += u_2h[i][j] * 0.5

            result[ii][jj] += u_2h[i][j]

    return result


def MG(u_in, f, gamma, nu1, nu2):
    u_smooth = GSsmoother(u_in, f, nu1)
    r_resid = residual(u_smooth, f)
    residu_rest = RESTR(r_resid)

    l = len(residu_rest)

    residu_rest_neg = [[0 for i in range(0, l)]for j in range(0, l)]

    for i in range(0, l):
        for j in range(0, l):
            residu_rest_neg[i][j] = -residu_rest[i][j]

    e = [[0 for i in range(0, l)]for j in range(0, l)]

    if l < 4:
        e = GSsmoother(e, residu_rest_neg, 1)
    else:
        for j in range(0, gamma):
            e = MG(e, residu_rest_neg, gamma, nu1, nu2)

    e_p = PROLONG(e)
    u_out = GSsmoother(substraction(u_smooth, e_p), f, nu2)

    return u_out


def performMG(u0, f, gamma, nu1, nu2, n):
    tol
    f, u0 = initial (n)
    r0 = max_norm(residual(u0, f))
    print("r0:", r0)

    resi = [0 for i in range(0, 25)]
    runtime = [0 for i in range(0, 25)]
    time_accumulated = 0

    for i in range(0, 25):
        start_time = time.time()  # Start measuring runtime

        if i == 0:
            u = MG(u0, f, gamma, nu1, nu2)

        else:
            u = MG(u, f, gamma, nu1, nu2)

        r = residual(u, f)
        r_norm = max_norm(r)
        convergence = r_norm / r0
        if convergence <= tol:
            continue


        # resi[i]=math.log(max_norm(residual(u,f))/r0,10)
        resi[i] = math.log(convergence, 10)
        end_time = time.time()  # Stop measuring runtime
        runtime[i] += end_time - start_time
        time_accumulated += runtime[i]
        print("Iteration", i+1, "Convergence:", convergence) #Uncomment to print the iteration count
        # print("Run Time:", time_accumulated, "Convergence:", convergence) #Uncomment to print the run-time

    print('finish')

    return u, resi, runtime, time_accumulated

time_start = time.time()
gamma = 2
nu1 = 2
nu2 = 1
n = 7 #Set 'n' manually here to make the mesh fine or coarse 
tol = 1e-10


f, u0 = initial(n)
u, resi, runtime, time_accumulated = performMG(u0, f, gamma, nu1, nu2, n)

time_end = time.time()
time_sum = time_end - time_start
print("Total runtime:", time_accumulated)