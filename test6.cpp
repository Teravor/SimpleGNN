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

  int sizeofsystem = 100;
  int sizeofstate = 10;

     srand(time(NULL));
     int sizes[3] = {sizeofstate*4,5,sizeofstate};

     Network *net = network_mlp_create(sizes,3);
 for(int i=0; i < net -> datasize; i++){
     net -> data[i] =0.01*(double)(rand() + 1)/(RAND_MAX);
 }

    mat ln = randu<mat>(100,sizeofstate);
    cube lnu = randu<cube>(100,sizeofstate,100);
    mat xu = randu<mat>(100,sizeofstate);
    mat Xold = randu<mat>(100,sizeofstate) ; 
    mat Xdiff = ones<mat>(100,sizeofstate) ; 
    mat X = randu<mat>(100,sizeofstate);
    mat lu = randu<mat>(100,sizeofstate);

for(int i=0; i < 100; i++){
//     X.print("X");
     cout << i << endl;
     Xold = X;
     X = hw(net, ln,lnu,X, lu,100,100);
     Xdiff = Xold - X;
     X.save("GNNX.dat");
     X.save("GNNXraw.dat",raw_ascii);
}

     
     X.print("X");
     Xold.print("Xold");
     Xdiff.print("Xdiff");
	    
return 0;

}







