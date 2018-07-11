#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_linalg.h>
#include <math.h>

#define Wavelet_size 100
const int row = 100, col = 50;
const double pi = 3.14159265359;

void linspace(double amp, int samples, double *vector) {
    double space = (2 * amp) / samples;
    int i = 0;
    double j = -amp;

    for (i = 0; i <= samples; i++) {
        vector[i] = j;
        //printf("%f \n", vector[i]);
        j += space;
    }

}
void Ricker(float c, double *tw, double *Wavelet){
    int i = 0;
    for(i=0;i<Wavelet_size;i++){
        tw[i] = tw[i]*0.004;
        Wavelet[i] = (1/pow((2*pi*pow(c,3)), 0.5))*(1-((pow(tw[i], 2))/pow(c,2)))*exp(-pow(tw[i], 2)/(2*pow(c,2)));
    }
}
void Generates_G(gsl_matrix *tst, double *hat, gsl_matrix *G){
    /*Setting Parameters */
    int i, j, k, l;
    long int n = tst->size1;
    long int m = Wavelet_size;

    double hatn[Wavelet_size+(2*n-m)]; // Wavelet with zeros to convolution
    double G_temp[Wavelet_size+(2*n-m)][n]; //Temporary G matrix

    /*Allocating Matrices*/
    for (i=0;i<Wavelet_size+(2*n-m);i++){
        hatn[i] = 0;
    }

    for(i=0;i<Wavelet_size+(2*n-m);i++){
        for(j=0;j<n;j++){
            G_temp[i][j] = 0;
        }
    }

    /*Building Toeplitz Matrix Above G_Temp*/
    for (i=0; i<Wavelet_size;i++){
        hatn[i] = hat[i];
    }


    for(i=0;i<Wavelet_size+(2*n-m);i++){
        G_temp[i][0] = hatn[i];
    }
    for(i=1;i<Wavelet_size+(2*n-m);i++){
        for(j=1;j<n;j++){
            G_temp[i][j] = G_temp[i-1][j-1];
        }
    }

    k = (int) round(m/2.0);
    l = (int) round(n+((m-1)/2.0));

    /*Cutting Zero Paddles of toeplitz matrix to G matrix*/
    for(i=k;i<l;i++){
        for(j=0;j<n;j++){
            gsl_matrix_set(G,i-k,j,G_temp[i][j]);
        }
    }
}
void Build_L2_Operator(gsl_matrix *G, gsl_matrix *Gf){
    long int n = G->size1;
    int alpha = 5000, s;
    int i=0, j=0,it=0;
    gsl_matrix *A, *Gt, *GtG, *GtG_Inv;
    A = gsl_matrix_calloc(G->size1,G->size1 );
    gsl_matrix_set_identity(A);
    gsl_matrix_scale(A, alpha*alpha);

    /*Transposed G Matrix*/
    Gt = gsl_matrix_calloc(G->size1, G->size2);
    gsl_matrix_memcpy(Gt,G);
    gsl_matrix_transpose(Gt);


    /*Middle matrices*/
    GtG = gsl_matrix_calloc(G->size1, G->size2);
    GtG_Inv = gsl_matrix_calloc(G->size1, G->size2);
    Gf = gsl_matrix_calloc(G->size1, G->size2);
    gsl_permutation *p = gsl_permutation_alloc(G->size1);

    /*Performing Tikhonov L2 Inversion*/
    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,Gt,G,1.0, GtG); // GTG
    gsl_matrix_add(GtG,A); // GTG  + (Alpha**2)*A
    gsl_linalg_LU_decomp(GtG, p,&s);
    gsl_linalg_LU_invert(GtG,p,GtG_Inv); //(GTG  + (Alpha**2)*A)^-1

    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,GtG_Inv,Gt,1.0, Gf);

}
void Inv_l2(gsl_vector *tst, gsl_matrix *L2_Op, double *Inv_Model){
    int i=0;
    gsl_vector *M;
    M = gsl_vector_alloc(tst->size);
    gsl_blas_dgemv(CblasNoTrans,1.0,L2_Op,tst,1.0, M);

    for (i=0;i<M->size;i++){
        Inv_Model[i] = gsl_vector_get(M,i);
    }
}



int main(int argc, char **argv) {
    /*defining variables*/
    MPI_Init(&argc, &argv);
    double t1, t2;
    t1 = MPI_Wtime();
    int myrank = 0, n_process = 0;

    int i , j;
    double linspace_vector[Wavelet_size];
    double Wavelet[Wavelet_size];

    FILE *fptSint, *fptG, *fpt_Inv;


    linspace(100, Wavelet_size, linspace_vector); //Computing linspace vector for Ricker Wavelet function.
    Ricker(0.06, linspace_vector, Wavelet);

    // Generate Sintetic Model. Still Haven't done it. So I did it on Python. Gotta Work it on!!!

    /*Here starts second part of code: Inversion*/
    /*Only first processor read files*/

    MPI_Comm_size(MPI_COMM_WORLD, &n_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int step = (int) round(col/n_process);

    gsl_matrix *G = gsl_matrix_calloc(row-1,row-1);
    gsl_matrix *L2_operator = gsl_matrix_calloc(row-1,row-1);
    double Inverted_Model[row - 1][col];

    if (myrank == 0) {

        printf("Initializing Inversion with %i processes. Step: %i\n", n_process, step);
        fptSint = fopen("Sint_Model_99x50.bin", "rb");
        fptG = fopen("G_Matrix_99x99.bin", "wb");
        fpt_Inv = fopen("Inv_Model_99x50.bin", "wb");
        /*Allocating and reading Sintetic Model Matrix*/

        gsl_matrix *Sintetic_Model;
        Sintetic_Model = gsl_matrix_calloc(row - 1, col);
        gsl_matrix_fread(fptSint, Sintetic_Model);

        /* G matrix of discrete Convolution! It's square. */
        Generates_G(Sintetic_Model, Wavelet, G);
        gsl_matrix_fwrite(fptG, G); //testing
        Build_L2_Operator(G, L2_operator);

        for (i=1;i<n_process;i++){
            MPI_Send(gsl_matrix_ptr(L2_operator,0,0),L2_operator->size1*L2_operator->size2,MPI_DOUBLE,i,i+1,MPI_COMM_WORLD);
        }
        for(i=1;i<n_process-1;i++)
            MPI_Send(gsl_matrix_ptr(Sintetic_Model,i*step,0),step*G->size1,MPI_DOUBLE,i,i+2,MPI_COMM_WORLD);

        MPI_Send(gsl_matrix_ptr(Sintetic_Model,n_process*step,0), (col- (step*n_process) + step)*G->size1,MPI_DOUBLE,n_process-1,n_process+1,MPI_COMM_WORLD);

        gsl_vector *trace;
        trace = gsl_vector_alloc(L2_operator->size1);
        for (i = 0; i < step; i++){
            gsl_matrix_get_col(trace, Sintetic_Model, i);
            Inv_l2(trace, L2_operator, &Inverted_Model[0][i+step*myrank]);
            gsl_vector_set_zero(trace);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        printf("Processor %i done\n", myrank);

        for (i=0;i<row-1;i++) {
            for (j = 0; j < col; j++) {
                fwrite(&Inverted_Model[i][j], sizeof(double), 1, fpt_Inv);
            }
        }


        gsl_matrix_free(Sintetic_Model);
        fclose(fptSint);
        fclose(fptG);
        fclose(fpt_Inv);

    }else{
            MPI_Recv(gsl_matrix_ptr(L2_operator,0,0),L2_operator->size1*L2_operator->size2,MPI_DOUBLE,0,myrank+1,MPI_COMM_WORLD,0);
            gsl_matrix *Parcial_Model;
            gsl_vector *trace;
            trace = gsl_vector_alloc(L2_operator->size1);

            if(myrank < n_process-1) {
                Parcial_Model = gsl_matrix_calloc(row - 1, step);
                MPI_Recv(gsl_matrix_ptr(Parcial_Model,0,0),step*G->size1,MPI_DOUBLE,0,myrank+2,MPI_COMM_WORLD, 0);
                for (i = 0; i < step; i++){
                    gsl_matrix_get_col(trace, Parcial_Model, i);

                    Inv_l2(trace, L2_operator, &Inverted_Model[0][i+step*myrank]);
                    gsl_vector_set_zero(trace);
                }
                printf("Processor %i done\n", myrank);
                MPI_Barrier(MPI_COMM_WORLD);

            }else{
                Parcial_Model = gsl_matrix_calloc(row - 1, col- (step*n_process) + step);
                MPI_Recv(gsl_matrix_ptr(Parcial_Model,0,0),(col- (step*n_process) + step)*G->size1,MPI_DOUBLE,0,myrank+2,MPI_COMM_WORLD, 0);

                for (i = 0; i < col- (step*n_process) + step; i++){
                    gsl_matrix_get_col(trace, Parcial_Model, i);

                    Inv_l2(trace, L2_operator, &Inverted_Model[0][i+step*myrank]);
                    gsl_vector_set_zero(trace);
                }
                printf("Processor %i done\n", myrank);
                MPI_Barrier(MPI_COMM_WORLD);

            }
    }

    //
    gsl_matrix_free(G);
    gsl_matrix_free(L2_operator);

    t2 = MPI_Wtime();
    if (myrank==0)
        printf("\n\n Tempo decorrido: %f\n\n", t2-t1);
    MPI_Finalize();



    return 0;
}
