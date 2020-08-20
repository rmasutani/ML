#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void) {
    int niter = 100;
    double step_size = 0.5e0;

    srand((unsigned)time(NULL));

    double x = 0e0;
    int naccept = 0;

    for (int iter=1;iter<niter+1;iter++) {
        double backup_x=x;
        double action_init = 0.5e0*x*x;

        double dx = (double)rand()/RAND_MAX;
        dx = (dx-0.5e0) * step_size * 2e0;
        x = x + dx;

        double action_fin = 0.5e0*x*x;

        // Metropolis test
        double metropolis = (double)rand()/RAND_MAX;

        if (exp(action_init - action_fin) > metropolis)
            naccept=naccept+1;
        else
            x = backup_x;
        
        printf("%.10f  %f\n", x, (double)naccept/iter);

        return 0;
     }
}