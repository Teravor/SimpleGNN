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

  int sizeofstate = 10;

     srand(time(NULL));



   ActivationFunction::Enum activators[] = {ActivationFunction::TANH, ActivationFunction::LINEAR};///
   int layer_sizes[] =  {3,sizeofstate};///

   Network* net = network_create(40, SIZEOF(layer_sizes), layer_sizes, SIZEOF(activators), activators);///
   vec parameters(net->parameter_size, fill::randn);///

  int  netDataSize = net -> parameter_size;

 for(int i=0; i <netDataSize ; i++){
     net -> parameters[i] =0.01*(double)(rand() + 1)/(RAND_MAX);
 }

    mat ln = randu<mat>(40,sizeofstate);
    cube lnu = randu<cube>(40,sizeofstate,40);
    mat xu = randu<mat>(40,sizeofstate);
    mat Xold = randu<mat>(40,sizeofstate) ; 
    mat Xdiff = ones<mat>(40,sizeofstate) ; 
    mat X = randu<mat>(40,sizeofstate);
    mat lu = randu<mat>(40,sizeofstate);
     X.print("X");
     Xold.print("Xold");
     Xdiff.print("Xdiff");

for(int i=0; i < 100; i++){
//     X.print("X");
     cout << i << endl;
     Xold = X;
     X = hw(net, ln,lnu,X, lu,40,40);
     Xdiff = Xold - X;
     X.save("GNNX.dat");
     X.save("GNNXraw.dat",raw_ascii);
     X.print("X cured");
     Xold.print("Xold cured");
     Xdiff.print("Xdiff cured");
}

     
     X.print("X");
     Xold.print("Xold");
     Xdiff.print("Xdiff");
	    
return 0;

}








