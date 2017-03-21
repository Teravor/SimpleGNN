#include <iostream>
#include <string>
#include <armadillo>
#include <time.h>
#include "network.h"
#include "myGaMDMPI.h"

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
     string saveBi = argv[7];
     string loadBi = argv[8];

     int world_size;
     int world_rank;

MPI_Init(&argc,&argv);
MPI_Comm_size(MPI_COMM_WORLD, &world_size);
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  vec winning;
	mat X;

//	X.load("sinx.dat");

int num = 100;///
 vec x = linspace<vec>(0,3.1416,num);///
 vec y = sin(x); ///

 X = join_rows(x,y);

   mat DNAS;
//   X.print("a");
   
  
   ActivationFunction::Enum activators[] = {ActivationFunction::SIGMOID,ActivationFunction::SIGMOID, ActivationFunction::SIGMOID};///
   int layer_sizes[] =  {50,50,1};///

   Network* net = network_create(1, SIZEOF(layer_sizes), layer_sizes, SIZEOF(activators), activators);///
   vec parameters(net->parameter_size, fill::randn);///

  int  netDataSize = net -> parameter_size;

 if(rerun==0){DNAS = randu<mat>(200,netDataSize);} else{DNAS.load(loadBi);}

    network_load_parameters(net, parameters);///

     
												      
  if(world_rank==0)   cout << net -> outputs[0] << endl; 


     for(int gen = 0; gen < generations; gen++){
      winning=  getBest(DNAS, net, X,world_rank, world_size);
//     timer.tic();
	      DNAS = ga_eval(DNAS, net, X ,0.001*temperature, 0.001*temperature, 0.0001*temperature,0.1*temperature,0.00001*temperature,1.0*temperature,finalizeit, regu, world_rank, world_size);

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
    if(world_rank == 0) Y.save("sinTest.dat",raw_ascii);

     winner.print("Final Winner!");
     double winVal = getVal(winner, net, X.row(10));

//     cout << "winval: " << winVal << endl;
//      winner.save("bestDNAS.dat",raw_ascii);
														      
 MPI_Finalize();
															      
return 0;
}

