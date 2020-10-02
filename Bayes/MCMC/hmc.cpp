#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>

const int niter = 10000;
const int ntau = 40;
const double dtau = 1e0;

int BoxMuller(double& p, double& q) {
    double pi = 2e0*asin(1e0);

    double r = (double)rand()/RAND_MAX;
    double s = (double)rand()/RAND_MAX;

    p = sqrt(-2e0*log(r))*sin(2e0*pi*s);
    q = sqrt(-2e0*log(r))*cos(2e0*pi*s);

    return 0;
}

double calc_action(const double x) {
    double action = 0.5e0*x*x;

    return action;
}

double calc_hamiltonian(const double x, const double p) {
    // Potential energy
    double ham = calc_action(x);
    // Add kinetic energy
    ham = ham + 0.5e0*p*p;

    return ham;
}

double calc_delh(const double x) {
    double delh = x;
    return delh;
}

// Leap frog
int Molecular_Dynamics(double &x, double &ham_init, double &ham_fin) {
    double r1, r2;
    BoxMuller(r1, r2);
    double p = r1;

    // Calculate hamiltonian
    ham_init = calc_hamiltonian(x, p);
    //Leap frong 1st step
    x = x + p*0.5e0*dtau;

    // 2, ... Ntau steps
    for (int step=1; step!=ntau; step++) {
        double delh = calc_delh(x);
        p = p - delh * dtau;
        x = x + p * dtau;
    }

    // Last step
    double delh = calc_delh(x);
    p = p - delh * dtau;
    x = x + 0.5e0 * dtau;

    // Hamiltonian again
    ham_fin = calc_hamiltonian(x, p);

    return 0;
}

int main() {
    srand((unsigned)time(NULL));

    double x = 0e0;

    // Main
    std::ofstream outputfile("output.txt");
    int naccept = 0;
    double sum_xx = 0e0;

    for (int iter = 0; iter!=niter; iter++) {
        double backup_x = x;
        double ham_init, ham_fin;
        Molecular_Dynamics(x, ham_init, ham_fin);
        double metropolis = (double)rand()/RAND_MAX;
        
        if (exp(ham_init-ham_fin) > metropolis) {
            naccept = naccept + 1;
        } else {
            x = backup_x;
        }

        sum_xx = sum_xx + x*x;

        std::cout << std::fixed << std::setprecision(6)
            << x << "   "
            << sum_xx / ((double)(iter+1)) << "   "
            << ((double)naccept) / ((double)iter+1)
            <<std::endl;
        
        outputfile << std::fixed << std::setprecision(6)
            << x << "   "
            << sum_xx / ((double)(iter+1)) << "   "
            << ((double)naccept) / ((double)iter+1)
            <<std::endl;
    }

    outputfile.close();
    return 0;
}

