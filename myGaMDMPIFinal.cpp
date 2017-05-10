#include <iostream>
#include <string>
#include <armadillo>
#include <time.h>
#include "network.h"
#include "myGaMDMPIFinal.h"
// EXAMPLE: mpiexec -n 2 ./test_ga 1 0 10 10 0.1 0.01 10 0.00001 Hello Hello
// EXAMPLE: mpiexec -n 2 ./test_ga return finalize gen temp regu decendingTemp linContr sqContr Hello Hello
#define SIZEOF(_arr) (sizeof(_arr)/sizeof(_arr[0]))

using namespace std;
using namespace arma;

int main(int argc, char* argv[]){
  
     int rerun = strtol(argv[1],NULL,10);

     int finalizeit = strtol(argv[2],NULL,10);
     int generations = strtol(argv[3],NULL,10);

     double temperature = (double) strtol(argv[4],NULL,10);

     double regu =  atof(argv[5]);
     double decendingTemp =  atof(argv[6]);
     double linContr =  atof(argv[7]);
     double sqContr =  atof(argv[8]);
     string saveBi = argv[9];
     string loadBi = argv[10];

     int world_size;
     int world_rank;

     MPI_Init(&argc,&argv);
     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
     MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

     vec winning;
     mat DNAS;
  
     if(rerun==0){DNAS = randu<mat>(200,100);} else{DNAS.load(loadBi);}
												      
     for(int gen = 0; gen < generations; gen++){

	     DNAS = ga_eval(DNAS,0.001*temperature, 0.001*temperature, 0.0001*temperature,0.1*temperature,0.00001*temperature,1.0*temperature,finalizeit, regu,linContr,sqContr, world_rank, world_size);

       if(world_rank == 0) DNAS.save(saveBi,raw_ascii);
       if(world_rank == 0){  cout << gen << "Winning Fitness: " <<  "???"  <<endl;}

       temperature = temperature - decendingTemp; 
     }

 MPI_Finalize();
															      
return 0;

}

