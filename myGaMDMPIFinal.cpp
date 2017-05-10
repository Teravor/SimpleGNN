#include <iostream>
#include <string>
#include <armadillo>
#include <time.h>
#include "network.h"
#include "myGaMDMPI.h"
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
	mat X;

//	X.load("sinx.dat");

int num = 100;///
//vec xx = linspace<vec>(-3,-2.9,100);
//vec xxx = linspace<vec>(2.9,3,100);
// vec xxxx = linspace<vec>(-2.9,2.9,num);///
// vec x = join_cols(join_cols(xx,xxxx),xxx);
vec x = linspace<vec>(0.1,3,100);
// vec y = (sqrt(1 - (x%x/3/3))/1.3+0.7); ///
 vec y = sin(x) - 0.1; ///

 X = join_rows(x,y);

   mat DNAS;
//   X.print("a");
   
  
   ActivationFunction::Enum activators[] = {ActivationFunction::SIGMOID,ActivationFunction::SIGMOID, ActivationFunction::LINEAR};///
   int layer_sizes[] =  {50,50,1};///

   Network* net = network_create(1, SIZEOF(layer_sizes), layer_sizes, SIZEOF(activators), activators);///
//   vec parameters(net->parameter_size, fill::randn);///
   vec parameters(net->parameter_size, fill::zeros);///

  int  netDataSize = net -> parameter_size;

if(rerun==0){DNAS = randu<mat>(200,netDataSize);} else{DNAS.load(loadBi);}
// if(rerun==0){DNAS = zeros<mat>(200,netDataSize);} else{DNAS.load(loadBi);}

//    network_load_parameters(net, parameters);///

     
												      
  if(world_rank==0)   cout << net -> outputs[0] << endl; 


     for(int gen = 0; gen < generations; gen++){
      winning=  getBest(DNAS, net, X,world_rank, world_size);
//     timer.tic();
	      DNAS = ga_eval(DNAS, net, X ,0.001*temperature, 0.001*temperature, 0.0001*temperature,0.1*temperature,0.00001*temperature,1.0*temperature,finalizeit, regu,linContr,sqContr, world_rank, world_size);

     if(world_rank == 0) DNAS.save(saveBi,raw_ascii);

    if(world_rank == 0){  cout << gen << " winning value is: " <<  getVal(winning, net, X.row(1)) <<endl;
      cout << gen << " Real    value is: " <<  X(1,X.n_cols - 1) <<endl;
      cout << gen << " Diff    value is: " <<  getValDiff(winning, net, X.row(1)) <<endl;}

//     myTime = timer.toc();

      temperature = temperature - decendingTemp; 

     }

     vec winner =  getBest(DNAS, net, X, world_rank, world_size);

     vec Vals = getResults(DNAS, net, X, world_rank, world_size);
     mat Y = join_rows(x,Vals);
     mat Z = join_rows(x,y);

    if(world_rank == 0) Y.save("sinNN.dat",raw_ascii);
//    if(world_rank == 0) Y.save("EclipseTest.dat",raw_ascii);
     winner.print("Final Winner!");
     double winVal = getVal(winner, net, X.row(10));
     Z.save("sinReal.dat",raw_ascii);
//     cout << "winval: " << winVal << endl;
//      winner.save("bestDNAS.dat",raw_ascii);
														      
 MPI_Finalize();
															      
return 0;
}

