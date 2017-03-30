#include <iostream>
#include <string>
#include <armadillo>
#include <time.h>
#include "network.h"
#include "myGaMDMPI.h"
#include "fileoperation.h"

#define SIZEOF(_arr) (sizeof(_arr)/sizeof(_arr[0]))


using namespace std;
using namespace arma;


int main(int argc, char* argv[]){
  string* myTypes = getType("au40cu40.xyz");

  mat A = getPos("au40cu40.xyz");
  int n_Atom = A.n_rows; 

  int sizeofstate = 10;

     srand(time(NULL));



   ActivationFunction::Enum activators[] = {ActivationFunction::TANH, ActivationFunction::LINEAR};///
   int layer_sizes[] =  {3,sizeofstate};///

   Network* net = network_create(sizeofstate + 3, SIZEOF(layer_sizes), layer_sizes, SIZEOF(activators), activators);///
   vec parameters(net->parameter_size, fill::randn);///

  int  netDataSize = net -> parameter_size;

 for(int i=0; i <netDataSize ; i++){
     net -> parameters[i] =0.01*(double)(rand() + 1)/(RAND_MAX);
 }

    mat ln = zeros<mat>(n_Atom,1); //(#Atoms, #Descripters)

    for(int i=0; i < n_Atom; i++)
        { 
           if(myTypes[i]== "Au"){ln(i) = 1;}
           else{ln(i) = 0;}

           }

    cube lnu = zeros<cube>(n_Atom,1,n_Atom);//(#Atoms,#descriptors,#Atoms)
for(int i=0; i < n_Atom; i++)
    { 
  cout << size(lnu) << endl;
     for(int j=0; j < n_Atom; j++)
         { 
        lnu(i,0,i) = 1; // sum((A.row(i)-A.row(j))%(A.row(i)-A.row(j)));
            }

       }
    mat xu = randu<mat>(n_Atom,sizeofstate);

    mat Xold = randu<mat>(n_Atom,sizeofstate) ; 
    mat Xdiff = ones<mat>(n_Atom,sizeofstate) ; 
    mat X = randu<mat>(n_Atom,sizeofstate);
    mat lu = zeros<mat>(n_Atom,1);
    lu = ln;
//     X.print("X");
//     Xold.print("Xold");
//     Xdiff.print("Xdiff");

for(int i=0; i < 100; i++){
//     X.print("X");
     cout << i << endl;
     Xold = X;
     X = hw(net, ln,lnu,X, lu,n_Atom,n_Atom);
     Xdiff = Xold - X;
     X.save("GNNX.dat");
     X.save("GNNXraw.dat",raw_ascii);
//     X.print("X cured");
//     Xold.print("Xold cured");
//     Xdiff.print("Xdiff cured");
}

     
     X.print("X");
     Xold.print("Xold");
     Xdiff.print("Xdiff");
	    
return 0;

}








