#include <iostream>
#include <string>
#include <armadillo>
#include <time.h>
extern "C" {
#include "network.h"
}
#include "myGa6.h"


using namespace std;
using namespace arma;


int main(int argc, char* argv[]){

     int rerun = strtol(argv[1],NULL,10);
     int finalizeit = strtol(argv[2],NULL,10);
     int generations = strtol(argv[3],NULL,10);
     cout << finalizeit << endl;
     srand(time(NULL));
     Network *SampNet;
     int sizes[7] = {20,5,5,5,5,5,1};
     double winningVal;
     vec winning;
     vec winningCV;

	mat X;
	X.load("norm_dat_H100.dat");
	mat Xtrain = X.rows(0,3999);
	mat Xcv = X.rows(4000,4999);

     Network *net = network_mlp_create(sizes,7);
	    

    int netDataSize = net -> datasize;
			      
  double* dna = new double [netDataSize];
  mat DNAS;

  if(rerun == 0){ DNAS = randu<mat>(1100,netDataSize);}
  else{DNAS.load("DNAScv");} 
  
     
     network_mlp_loaddata(net, dna);
												      
     network_mlp_compute_relu_lin(net);
     cout << net -> outputs[0] << endl; 

     for(int gen = 0; gen < generations; gen++){
      winning=  getBest(DNAS, net, Xtrain);
      winningCV=  getBest(DNAS, net, Xcv);
	     if(finalizeit == 0){ DNAS = ga_eval(DNAS, net, Xtrain ,0.01, 0.1, 0.005, 10.0,0.0005,100.0, finalizeit,winning);}
	     else{DNAS = ga_eval(DNAS, net, Xtrain ,0.99, 0.001, 0.00001, 1.0,0.000000001,10, finalizeit,winning);}
      cout << gen << " winning TR value diff is: " <<  getVal(winning, net, Xtrain.row(10)) <<endl;
      cout << gen << " winning CV value diff is: " <<  getVal(winningCV, net, Xcv.row(10)) <<endl;
      DNAS.save("DNAScv");
      DNAS.save("DNAScv.dat",raw_ascii);
     }


     vec winner =  getBest(DNAS, net, Xtrain);

     vec Vals = getVals(DNAS, net, Xtrain.row(10));

     winner.print("Final Winner!");
     double winVal = getVal(winner, net, Xtrain.row(10));

     cout << "winval: " << winVal << endl;
      winner.save("bestDNAScv.dat",raw_ascii);
														      
															      
return 0;

}







