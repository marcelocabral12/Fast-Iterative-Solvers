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

    for (int i = 0; i < A.size; i++)
    {

        y[i] += A.VM[i] * x[i];
    }

    for (int i = 0; i < A.size; i++)
    {

        for (int j = A.JM[i]; j < A.JM[i + 1]; j++)
        {

            y[i] += A.VM[j] * x[A.JM[j]];
        }
    }
    if (A.symmetry == 's')
    {

        for (int i = 0; i < A.size; i++)
        {

            for (int j = A.JM[i]; j < A.JM[i + 1]; j++)
            {

                y[A.JM[j]] += A.VM[j] * x[i];
            }
        }
    }

    return y;
}
vector<double> operator+(const vector<double> &v1, const vector<double> &v2)
{
    vector<double> result(v1.size(), 0.0);
    for (unsigned int i = 0; i < v1.size(); ++i)
    {
        result[i] = v1[i] + v2[i];
    }
    return result;
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
double operator*(const vector<double> &v1, const vector<double> &v2)
{
    double result = 0;
    for (unsigned int i = 0; i < v1.size(); ++i)
    {
        result += v1[i] * v2[i];
    }
    return result;
}
vector<double> operator*(const double a, const vector<double> &v1)
{
    vector<double> result = v1;
    for (unsigned int i = 0; i < v1.size(); ++i)
    {
        result[i] *= a;
    }
    return result;
}
vector<double> operator-(const double a, const vector<double> &v1)
{
    vector<double> result = v1;
    for (unsigned int i = 0; i < v1.size(); ++i)
    {
        result[i] = v1[i] - a;
    }
    return result;
}
vector<double> operator-(const vector<double> &v1, const double a)
{
    vector<double> result(v1.size(), 0.0);
    for (unsigned int i = 0; i < v1.size(); ++i)
    {
        result[i] = v1[i] - a;
    }
    return result;
}
void operator+=(vector<double> &v1, const vector<double> &v2)
{
    for (unsigned int i = 0; i < v1.size(); ++i)
    {
        v1[i] += v2[i];
    }
}
void operator-=(vector<double> &v1, const vector<double> &v2)
{
    for (unsigned int i = 0; i < v1.size(); ++i)
    {
        v1[i] -= v2[i];
    }
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

double cg(MSR_matrix &A, vector<double> &x0, vector<double> &b, double tol)
{
    vector<double> rm = b - matrix_vec_product(A, x0);
    vector<double> pm = b - matrix_vec_product(A, x0);
    double norm = euclidean_norm(rm);
    vector<double> x_m = x0;
    vector<double> e(A.size, 0);
    vector<double> product_A_err;
    double A_norm_err;
    double norm_residual;

    const double residual_norm = norm;
    double rel_residual_norm = 1;
    while (rel_residual_norm > tol)
    {
        double inner_product1 = rm * rm;
        vector<double> Apm = matrix_vec_product(A, pm);
        double alpha = inner_product1 / (Apm * pm);
        x_m += alpha * pm;
        e = 1.0 - x_m; // absolute error between actual sol and iterative sol
        rm -= alpha * matrix_vec_product(A, pm);
        double norm_rm = sqrt(rm*rm);

        //double beta_m = euclidean_norm(rm) / inner_product1;
        double beta_m = rm*rm / inner_product1;
        pm = rm + beta_m * pm;
        // for (int i = 0; i < pm.size(); ++i)
        // {
        //     pm[i] = rm[i] + beta_m * pm[i];
        // }

        product_A_err = matrix_vec_product(A, e); // matrix vector product for A and err

        A_norm_err = 0; // stores square root of dot product(err,A err)
        for (long int k = 0; k < e.size(); ++k)
        {
            A_norm_err += e[k] * product_A_err[k];
        }
        A_norm_err = sqrt(A_norm_err);

        rel_residual_norm = euclidean_norm(rm) / residual_norm;
        norm_residual = euclidean_norm(rm);
        //cout << rel_residual_norm << "," << A_norm_err << endl;
        cout << norm_residual << "," << A_norm_err << endl;
        //cout << A_norm_err << endl;
    }

    return rel_residual_norm;
}

int main() // change pa
{

    char filename[] = "cg_test_msr.txt";
    MSR_matrix A = MSR_matrix(filename);
    if (!A.success)
    {
        return 0;
    }
    double tol = 1e-8;
    // creating vector of ones
    vector<double> x(A.size, 1);

    // creating vectors of zeros
    vector<double> x0(A.size, 0);

    // calculating matrix vector product and finding RHS
    vector<double> b = matrix_vec_product(A, x);

    // start the time
    auto start = chrono::high_resolution_clock::now();

    double residual_norm = cg(A, x0, b, tol);

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
