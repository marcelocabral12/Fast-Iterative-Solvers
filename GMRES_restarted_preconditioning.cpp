// Change the parameters in int main to control this code. There is also set the preconditioner.
// Marcelo Cabral Liberato de Mattos Filho 
//Enrollment number: 404864

#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <tuple>
#include <chrono>
#include <string>
#include <cstring>
#include <numeric>
using namespace std;
using namespace std::chrono;

struct gmres_return
{
    vector<double> xm;
    double rho;
};

string trim(const string &str)
{
    size_t first = str.find_first_not_of(' ');
    if (string::npos == first)
    {
        return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

class MSR_matrix // stores the MSR matrix
{
public:
    char symmetry;
    long int size, nonzeros;
    double *VM;
    int *JM;
    int success;
    MSR_matrix(char *filename);
};

MSR_matrix::MSR_matrix(char *filename)
{
    // constructor to input values from file and store it in CRS format
    ifstream msrfile;
    msrfile.open(filename, ios::in);
    if (!msrfile.is_open())
    {
        cout << " Error!. CSR file cannot be opened\n";
        success = 0;
    }
    msrfile.seekg(0, ios::beg);
    // here I take the information from the first 2 lines (Symmetry, size and # of non-zeros)
    msrfile >> symmetry >> size >> nonzeros;

    VM = new double[nonzeros];
    JM = new int[nonzeros];

    for (int i = 0; i < nonzeros; i++)
    {
        msrfile >> JM[i] >> VM[i];
    }
    for (int i = 0; i < nonzeros; i++)
    {
        JM[i] = JM[i] - 1;
    }
    for (int index = 0; index < nonzeros; index++)
    {
        // cout << JM[index] << endl;
    }

    msrfile.close();
    success = 1;
}

inline vector<double> matrix_vec_product(const MSR_matrix &A, vector<double> &x)
{

    if (A.size != x.size())
    {
        cout << " Error!. Size mismatch between matrix and vector\n";
        exit(0);
    }
    vector<double> y;
    y.assign(A.size, 0);

    for (size_t i = 0; i < A.size; i++)
    {

        y[i] += A.VM[i] * x[i];
    }

    for (size_t i = 0; i < A.size; i++)
    {

        for (size_t j = A.JM[i]; j < A.JM[i + 1]; j++)
        {

            y[i] += A.VM[j] * x[A.JM[j]];
        }
    }
    if (A.symmetry == 's')
    {

        for (size_t i = 0; i < A.size; i++)
        {

            for (size_t j = A.JM[i]; j < A.JM[i + 1]; j++)
            {

                y[A.JM[j]] += A.VM[j] * x[i];
            }
        }
    }

    return y;
}

inline double euclidean_norm(vector<double> &y)
{
    double norm_y = 0.0;
    for (size_t i = 0; i < y.size(); i++)
    {
        norm_y += y[i] * y[i];
    }
    norm_y = sqrt(norm_y);
    return norm_y;
}
vector<double> operator-(const vector<double> &v1, const vector<double> &v2)
{
    vector<double> result(v1.size(), 0.0);
    for (unsigned int i = 0; i < v1.size(); ++i)
    {
        result[i] = v1[i] - v2[i];
    }
    return result;
}

// // Implement my Get Krylov here ....
void getKrylov(MSR_matrix &A, vector<vector<double>> &V, vector<vector<double>> &H, int j, auto preconditioner)
{
    vector<double> w = matrix_vec_product(A, V[j]); // compute w_j = A V_j
    preconditioner(A, w);
    for (int i = 0; i <= j; i++)
    {
        H[i][j] = inner_product(V[i].begin(), V[i].end(), w.begin(), 0.0); // compute H_ij
        for (int k = 0; k < A.size; k++)
        {
            w[k] -= H[i][j] * V[i][k]; // subtract H_ij V_i from w
        }
    }
    H[j + 1][j] = euclidean_norm(w); // compute H_j+1,j
    if (H[j + 1][j] != 0.0)
    {
        for (int k = 0; k < A.size; k++)
        {
            V[j + 1][k] = (w[k] / H[j + 1][j]); // normalize the new vector and store it in V
        }
    }
}

void no_preconditioner(const MSR_matrix &A, vector<double> &V)
{
}

void jacobi(const MSR_matrix &A, vector<double> &V)
{
    for (int i = 0; i < V.size(); i++)
    {
        V[i] = V[i] / A.VM[i];
    }
}

void gauss_seidel(const MSR_matrix &A, vector<double> &V)
{
    for (int i = 0; i < V.size(); i++)
    {
        double temp = 0;
        for (int j = A.JM[i]; j < A.JM[i + 1]; j++)
        {
            if (A.JM[j] > i)
            {
                break;
            }
            else
                temp += V[A.JM[j]] * A.VM[j];
        }

        V[i] = (V[i] - temp) / A.VM[i];
    }
}
// Here is my ILU(0) implementation but I had too many problems trying to debug it and make it run properly, therefore I could not finish it.
// void ilu(const MSR_matrix &A, vector<double> &V)
// {
//     MSR_matrix LU = A;

//     vector<double> row_u_JM(A.JM.begin() + 1, A.JM.begin() + A.VM + 1);
//     for (int i = 0; i < V.size(); i++)
//     {
//         for (int j = A.JM[i]; j < A.JM[i + 1]; j++)
//         {
//             if (LU.JM[j] > i)
//             {
//                 row_u_JM[i] = j;
//                 break;
//             }
//             else
//                 LU.VM[j] = LU.VM[j] / LU.VM[i];
//             for (int k = A.JM[i] + 1; k < A.JM[i + 1]; k++)
//             {
//                 LU.VM[k] -= LU.VM[k] * LU.VM[j];
//             }
//         }
//     }

//     for (int i = 0; i < LU.VM; i++)
//     {
//         double tmp = 0;
//         for (int j = LU.JM[i]; j < row_u_JM[i]; j++)
//         {
//             tmp += LU.VM[j] * V[LU.JM[j]];
//         }
//         V[i] -= tmp;
//     }

//     for (int i = LU.VM - 1; i >= 0; i--)
//     {
//         double tmp = 0;
//         for (int j = row_u_JM[i]; j < LU.JM[i + 1]; j++)
//         {
//             tmp += LU.VM[j] * V[LU.JM[j]];
//         }

//         V[i] = (V[i] - tmp) / LU.VM[i];
//     }
// }

gmres_return gmres(MSR_matrix &A, vector<double> &x0, vector<double> &b, int m, double tol, auto preconditioner)
{
    vector<vector<double>> V(m + 1, vector<double>(A.size, 0)); // initialize empty matrix V
    vector<vector<double>> H(m + 1, vector<double>(m, 0));      // initialize empty matrix H
    vector<double> r0 = b - matrix_vec_product(A, x0);
    preconditioner(A, r0);
    vector<double> s(m, 0);
    vector<double> c(m, 0);
    vector<double> g(m, 0);
    vector<double> y(m, 0);
    vector<double> xm(A.size, 0);
    int m_tilda = m;
    double rel_residual;

    double beta = euclidean_norm(r0); // compute initial beta
    for (int i = 0; i < A.size; i++)
    {
        V[0][i] = (r0[i] / beta); // normalize the first vector and store it in V
    }

    g[0] = beta;
    for (int j = 0; j < m; j++)
    {
        getKrylov(A, V, H, j, preconditioner);
        for (int k = 1; k <= j; k++)
        {
            double temp = -s[k - 1] * H[k - 1][j] + c[k - 1] * H[k][j];

            H[k - 1][j] = c[k - 1] * H[k - 1][j] + s[k - 1] * H[k][j];
            H[k][j] = temp;
        }

        double temp = sqrt(H[j][j] * H[j][j] + H[j + 1][j] * H[j + 1][j]);
        c[j] = H[j][j] / temp;
        s[j] = H[j + 1][j] / temp;
        H[j][j] = temp;

        g[j + 1] = -s[j] * g[j];
        g[j] = c[j] * g[j];
        beta = abs(g[j + 1]);
        // rel_residual = beta / euclidean_norm(r0);
        // cout << rel_residual << endl;

        cout << "my residual is:" << beta << endl; //----- to check if my residual is going down (comment break as well)
        if (beta < tol)
        {
            m_tilda = j + 1;
            cout << "break here!" << endl;
            break;
        }
    }

    // cout << m_tilda << "\n";
    //   for (int i = 0; i < H.size(); i++) // Here to print my H Matrix for testing with the small matrix 'n'
    //   {
    //       for (int j = 0; j < H[0].size(); j++)
    //       {
    //           cout << setw(20) << H[i][j] << "";
    //       }
    //       cout << endl;
    //   }

    // For Orthogonality Verification which I do dot product Vector V1 over all other Vk vectors
    // double dtmp;
    // for (int j = 0; j < m; j++)
    // {
    //     // Calculate the dot product between V0 and V[j]
    //     double dotProduct = inner_product(V[0].begin(), V[0].end(), V[j].begin(), 0.0); // *******Here Dot Product calculation*******
    //     // cout << "Dot product between V0 and V" << j << ": " << dotProduct << endl;
    //     cout << dotProduct << endl;
    // }

    for (int i = m_tilda - 1; i > -1; i--)
    {
        double dtmp = 0;
        for (int k = i + 1; k < m_tilda; k++)
        {
            dtmp += H[i][k] * y[k];
        }
        y[i] = (g[i] - dtmp) / H[i][i];
    }

    for (int i = 0; i < A.size; i++)
    {
        xm[i] = x0[i];
        for (int j = 0; j < m_tilda; j++)
        {

            xm[i] += V[j][i] * y[j];
        }
    }
    // for (int i = 0; i < 4; i++)
    // {
    //     cout << setw(20) << y[i] << "\n";
    // }
    gmres_return ret;
    ret.xm = xm;
    ret.rho = beta;
    return ret;
}

vector<double> restarted_gmres(MSR_matrix &A, vector<double> &x0, vector<double> &b, int m, double tol, auto preconditioner)
{
    vector<double> r0 = b - matrix_vec_product(A, x0);
    preconditioner(A, r0);
    double rho = euclidean_norm(r0);
    tol = tol * rho;
    vector<double> x = x0;

    gmres_return ret;
    int iteration = 0;

    while (rho > tol)
    {
        std::cout << "restart " << iteration << endl; // * I print here how many restarts the code generates
        ret = gmres(A, x, b, m, tol, preconditioner);
        x = ret.xm;
        rho = ret.rho;
        iteration++;
    }

    return ret.xm;
}

int main() // change pa
{

    char filename[] = "gmres_test_msr.txt";
    MSR_matrix A = MSR_matrix(filename);
    if (!A.success)
    {
        return 0;
    }
    int m = 600; // Iterations number
    double tol = 1e-8;
    // creating vector of ones
    vector<double> x(A.size, 1);

    // creating vectors of zeros
    vector<double> x0(A.size, 0);

    // calculating matrix vector product and finding RHS
    vector<double> b = matrix_vec_product(A, x);
    // for (long int i = 0; i < A.size; i++)
    // First tests of printing vector b[i] with the small matrix 'n'
    // {
    //     cout << b[i] << "\n";
    // }
    // start the time
    auto start = chrono::high_resolution_clock::now();

    vector<double> xm(m, 0);
    xm = restarted_gmres(A, x0, b, m, tol, gauss_seidel); // Here I set the Preconditioner

    // stop the timer
    auto end = chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    // Print the duration
    cout << "Execution time: " << duration << " milliseconds" << endl;

    // for (int i = 0; i < A.size; i++)
    // {
    //     cout << setw(20) << xm[i] << "";
    //     cout << endl;
    // }

    return 0;
}
