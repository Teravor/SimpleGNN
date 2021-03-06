#include <iostream>
#include <string>
#include <armadillo>
#include <time.h>
#include "network.h"
#include "myGa7.h"

#define SIZEOF(_arr) (sizeof(_arr)/sizeof(_arr[0]))

using namespace std;
using namespace arma;


int main(int argc, char* argv[]){
     int rerun = strtol(argv[1],NULL,10);
     int decisionVal;
     if (rerun == 0)
      {
       cout << "You are abount to start a new NN, are you sure?(1/0): " ;
       cin >> decisionVal;
       if(decisionVal == 1) {} else{exit(1);};
         }


     int finalizeit = strtol(argv[2],NULL,10);
     int generations = strtol(argv[3],NULL,10);

     double temperature = (double) strtol(argv[4],NULL,10);
     if (temperature > 50 && rerun > 0)
      {
       cout << "Temperature is quite high, This could blow up the previous NN. Sure you wanna continue?(1/0)" ;
       cin >> decisionVal;
       if(decisionVal == 1) {} else{exit(1);};
         }

     double regu = (double) strtol(argv[5],NULL,10);
     double decendingTemp =  atof(argv[6]);
     string saveBi = argv[7];
     string loadBi = argv[8];
cout << 100.0 * decendingTemp << endl;
//  int finalizeit = 0;
  vec winning;
	mat X;
//  int generations = 200;
//	X.load("sinx.dat");

int num = 100;///
 vec x = linspace<vec>(0,3.1416,num);///
 vec y = sin(x); ///

 X = join_rows(x,y);

   mat DNAS;
//   X.print("a");
   
  
   ActivationFunction::Enum activators[] = {ActivationFunction::SIGMOID, ActivationFunction::SIGMOID};///
   int layer_sizes[] =  {100,1};///

   Network* net = network_create(1, SIZEOF(layer_sizes), layer_sizes, SIZEOF(activators), activators);///
   vec parameters(net->parameter_size, fill::randn);///

  int  netDataSize = net -> parameter_size;

 if(rerun==0){DNAS = randu<mat>(100,netDataSize);} else{DNAS.load(loadBi);}

    network_load_parameters(net, parameters);///

     
												      
     cout << net -> outputs[0] << endl; 


     for(int gen = 0; gen < generations; gen++){
      winning=  getBest(DNAS, net, X);
//     timer.tic();
	      DNAS = ga_eval(DNAS, net, X ,0.001*temperature, 0.001*temperature, 0.0001*temperature,0.1*temperature,0.00001*temperature,1.0*temperature,finalizeit, regu);

      DNAS.save(saveBi,raw_ascii);

      cout << gen << " winning value is: " <<  getVal(winning, net, X.row(1)) <<endl;
      cout << gen << " Real    value is: " <<  X(1,X.n_cols - 1) <<endl;
      cout << gen << " Diff    value is: " <<  getValDiff(winning, net, X.row(1)) <<endl;

//     myTime = timer.toc();

      temperature = temperature - decendingTemp; 

     }

     vec winner =  getBest(DNAS, net, X);

     vec Vals = getResults(DNAS, net, X);
     cout << size(Vals) << endl;
     mat Y = join_rows(x,Vals);
     Y.save("sinTest.dat",raw_ascii);

     winner.print("Final Winner!");
     double winVal = getVal(winner, net, X.row(10));

//     cout << "winval: " << winVal << endl;
//      winner.save("bestDNAS.dat",raw_ascii);
														      
															      
return 0;

}
