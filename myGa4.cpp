#include <iostream>
#include <string>
#include <armadillo>
#include <time.h>
extern "C" {
#include "network.h"
}
#include "myGa.h"


using namespace std;
using namespace arma;


int main(int argc, char* argv[]){

     int rerun = strtol(argv[1],NULL,10);
     int finalizeit = strtol(argv[2],NULL,10);
     int generations = strtol(argv[3],NULL,10);
     cout << finalizeit << endl;
     srand(time(NULL));
     Network *SampNet;
     int sizes[3] = {20,50,1};
     double winningVal;
     vec winning;

	mat X;
	X.load("norm_dat_H100.dat");

     Network *net = network_mlp_create(sizes,3);
	    

    int netDataSize = net -> datasize;
			      
  double* dna = new double [netDataSize];
  mat DNAS;

  if(rerun == 0){ DNAS = randu<mat>(1100,netDataSize);}
  else{DNAS.load("DNAS");} 
  
     
     network_mlp_loaddata(net, dna);
												      
     network_mlp_compute_relu_lin(net);
     cout << net -> outputs[0] << endl; 

     for(int gen = 0; gen < generations; gen++){
      winning=  getBest(DNAS, net, X);
	     if(finalizeit == 0){ DNAS = ga_eval(DNAS, net, X ,0.1, 0.001, 0.001, 100.0, 100,finalizeit,winning);}
	     else{DNAS = ga_eval(DNAS, net, X ,0.99, 0.001, 0.00001, 1.0, 0,finalizeit,winning);}
      cout << gen << " winning value is: " <<  getVal(winning, net, X.row(10)) <<endl;
      DNAS.save("DNAS");
      DNAS.save("DNAS.dat",raw_ascii);
     }


     vec winner =  getBest(DNAS, net, X);

     vec Vals = getVals(DNAS, net, X.row(10));
//     Vals.print("vals0");
     winner.print("Final Winner!");
     double winVal = getVal(winner, net, X.row(10));

     cout << "winval: " << winVal << endl;
      winner.save("bestDNAS.dat",raw_ascii);
														      
															      
return 0;

}







