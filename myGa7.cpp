#include <iostream>
#include <string>
#include <armadillo>
#include <time.h>
extern "C" {
#include "network.h"
}
#include "myGa7.h"


using namespace std;
using namespace arma;


int main(int argc, char* argv[]){
     wall_clock timer;
     double myTime;
     int rerun = strtol(argv[1],NULL,10);
     int finalizeit = strtol(argv[2],NULL,10);
     int generations = strtol(argv[3],NULL,10);
     string saveBi = argv[4];
     string loadBi = argv[5];
     cout << finalizeit << endl;
     srand(time(NULL));
     Network *SampNet;
     double winningVal;
     vec winning;
     vec winningCV;

	mat X;
	X.load("loccmat_en_1_100000.dat");
  X.col(X.n_cols - 1) =100*( X.col(X.n_cols - 1) + 1739.856931542494976 + 1.16195386047558/2);
	mat Xtrain = X.rows(0,X.n_rows - 3001);
	mat Xcv = X.rows(X.n_rows - 3000,X.n_rows - 1);
     int sizes[4] = {X.n_cols - 1,3,100,1};

     cout << "Training data size: " << Xtrain.n_rows << "    ";
     cout << "CV data size: " << Xcv.n_rows << "      ";

     Network *net = network_mlp_create(sizes,4);
     cout << "NN size: " << net -> datasize << endl;
	    

    int netDataSize = net -> datasize;
			      
  double* dna = new double [netDataSize];
  mat DNAS;
vec vDiff = zeros<vec>(Xcv.n_rows);
vec vPred(Xcv.n_rows);
vec vExact(Xcv.n_rows);
vec tExact(Xtrain.n_rows);
vec tPred(Xtrain.n_rows);
vec tDiff(Xtrain.n_rows);

  if(rerun == 0){ DNAS = randu<mat>(200,netDataSize);}
  else{DNAS.load(loadBi);} 
  
  
     
     network_mlp_loaddata(net, dna);
												      
     network_mlp_compute_relu_lin(net);
     cout << net -> outputs[0] << endl; 


     for(int gen = 0; gen < generations; gen++){
//      winning=  getBest(DNAS, net, Xtrain);
      winningCV=  getBest(DNAS, net, Xcv);
     timer.tic();
//	      DNAS = ga_eval(DNAS, net, Xtrain ,0.01, 0.1, 0.005, 0.5,0.001,100.0, finalizeit, max(abs(vDiff)));
	     if(finalizeit == 0){ DNAS = ga_eval(DNAS, net, Xtrain ,0.01, 0.1, 0.001,1.0,0.0005,10.0,finalizeit,max(abs(vDiff)));}
	     else{DNAS = ga_eval(DNAS, net, Xtrain ,0.99, 0.001, 0.00001, 1.0,0.000000001,10, finalizeit,max(abs(vDiff)));}
      DNAS.save(saveBi);
//      cout << gen << " winning TR value diff is: " <<  getVal(winning, net, Xtrain.row(10)) <<endl;
      for(int i=0; i < Xcv.n_rows; i++){
       vDiff(i) =  getValDiff(winningCV, net,Xcv.row(i)); 
      }
      for(int i=0; i < Xcv.n_rows; i++){
       vPred(i) =  getVal(winningCV, net,Xcv.row(i)); 
      }
      vExact = Xcv.col(Xcv.n_cols - 1);

      for(int i=0; i < Xtrain.n_rows; i++){
       tDiff(i) =  getValDiff(winningCV, net,Xtrain.row(i)); 
      }
      for(int i=0; i < Xtrain.n_rows; i++){
       tPred(i) =  getVal(winningCV, net,Xtrain.row(i)); 
      }
      tExact = Xtrain.col(Xtrain.n_cols - 1);

      cout << gen << " Weight Sample " <<  DNAS(3,7) <<endl;
      cout << gen << " Largest Error " <<  max(abs(vDiff)) <<endl <<endl;
      cout << gen << " winning CV value is: " <<  getVal(winningCV, net, Xcv.row(1)) <<endl;
      cout << gen << " Real    CV value is: " <<  Xcv(1,Xcv.n_cols - 1) <<endl;
      cout << gen << " Diff       value is: " <<  getValDiff(winningCV, net, Xcv.row(1)) <<endl;
// SLOW!!      DNAS.save(saveBi + ".dat",raw_ascii);
     myTime = timer.toc();
     cout << "Time Full: " << myTime << " sec" << endl;
     vDiff.save("CVDiffSmall.dat",raw_ascii);
     vExact.save("CVExactSmall.dat",raw_ascii);
     vPred.save("CVPredSmall.dat",raw_ascii);
     tDiff.save("TrDiffSmall.dat",raw_ascii);
     tExact.save("TrExactSmall.dat",raw_ascii);
     tPred.save("TrPredSmall.dat",raw_ascii);
     }







     vec winner =  getBest(DNAS, net, Xtrain);

     vec Vals = getVals(DNAS, net, Xtrain.row(10));

     winner.print("Final Winner!");
     double winVal = getVal(winner, net, Xtrain.row(10));

     cout << "winval: " << winVal << endl;
      winner.save("bestDNAS.dat",raw_ascii);
														      
															      
return 0;

}







