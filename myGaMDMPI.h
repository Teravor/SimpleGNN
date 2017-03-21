#ifndef myGAMPI
#define myGA7MPI

#include <iostream>
#include <string>
#include <armadillo>
#include <time.h>
#include <mpi.h>
#include "network.h"
#include "armadilloMPI.h"


using namespace std;
using namespace arma;

//----------------------------------------------------------------------
vec getVals(mat DNAS, Network* net, rowvec inp){
	 vec vals(DNAS.n_rows);
  vec inputs = inp(span(0,inp.n_elem - 2)).t();
           for(int j=0;j < DNAS.n_rows; j++){
           for(int i=0;i < DNAS.n_cols; i++){ //
	              net->parameters[i] = DNAS(j,i);
	          }
       network_compute(net, inputs);
//           network_mlp_compute_relu_lin(net);
	            vals(j) = net -> outputs[0];
	         }
	   return vals;
       }


//----------------------------------------------------------------------
double getValDiff(vec v, Network* net, rowvec inp){
  vec inputs = inp(span(0,inp.n_elem - 2)).t();
           for(int i=0;i < v.n_elem; i++){
	      net->parameters[i] = v(i);
	   }
           network_compute(net,inputs);
	   return net -> outputs[0] - inp(inp.n_elem-1);
       }
//----------------------------------------------------------------------
double getVal(vec v, Network* net, rowvec inp){
  vec inputs = inp(span(0,inp.n_elem - 2)).t();
           for(int i=0;i < v.n_elem; i++){
	      net->parameters[i] = v(i);
	   }
           network_compute(net,inputs);
	   return net -> outputs[0];
       }
//----------------------------------------------------------------------
vec getBest(mat DNAS, Network* net, mat X, int world_rank, int world_size){
	vec accv = zeros<vec>(DNAS.n_rows);
	vec accvBuff = zeros<vec>(DNAS.n_rows);
	int XNrows = X.n_rows;
	int XNcols = X.n_cols;
	int DNASNcols = DNAS.n_cols;
	int DNASNrows = DNAS.n_rows;
	vec v(DNAS.n_rows);
	vec prod(X.n_rows);
  //------Paral -----------------------------------------------------
        for(int k=  XNrows / world_size * world_rank; k < XNrows*( world_rank + 1)/ world_size; k++){
        for(int j=0;j < DNASNrows; j++){
           for(int i=0;i < DNASNcols; i++){ 
	      net->parameters[i] = DNAS.at(j,i);
	   }
         network_compute(net,X(k,span(0,X.n_cols - 2)));
	  v(j) = net -> outputs[0]; 
	  accv(j) = accv(j) + abs(v(j) - X(k,XNcols-1)); 
	}
}


if(world_rank != 0){
	        MPI_SendVec( accv, 0, 123);
	        MPI_RecvVec( accv, 0, 234);
}

if(world_rank == 0){ 

	for(int i=1; i < world_size; i++) { MPI_RecvVec( accvBuff, i, 123); accv = accv + accvBuff; }


  //------Paral End -----------------------------------------------------

	accv = - accv;


        cout << " max average fitness CV per sample: " << accv.max()/X.n_rows << endl;

	for(int i=1; i < world_size; i++) { MPI_SendVec(accv, i, 234);};

    }


	return DNAS.row(index_max(accv)).t();

        }

//----------------------------------------------------------------------
vec getResults(mat DNAS, Network* net, mat X, int world_rank, int world_size){
    vec bestWeights = getBest(DNAS, net, X, world_rank, world_size);
	vec results(X.n_rows);
    for(int j=0;j < bestWeights.n_elem; j++){
        net->parameters[j] = bestWeights(j);
    }
	for(int i=0; i < X.n_rows; i++){
         network_compute(net,X(i,span(0,X.n_cols - 2)));
       results(i) = net -> outputs[0];
    }
    
	   return results;
       }
//----------------------------------------------------------------------
mat ga_eval(mat DNAS, Network* net, mat X ,  double mutRat, double mutAmmount, double bigMut,double bigMutAmmount,
                                                          double veryBigMut, double verBigMutAmmount, int
                                                          finalizeit, double regu, int world_rank, int world_size){

//MPI_Init(NULL,NULL);
//cout << "YO, rank iz: "<< world_rank <<endl;

  int Xrows = X.n_rows;
  int Xcols = X.n_cols;
  int DNAScols = DNAS.n_cols;
  int DNASrows = DNAS.n_rows;
	int XNrows = X.n_rows;
	int XNcols = X.n_cols;
	int DNASNcols = DNAS.n_cols;
	int DNASNrows = DNAS.n_rows;
	vec n = zeros<vec>(DNASrows);
	int decision[2*DNASrows];
	int myindex;
	double randBuff;
	rowvec baby(DNAScols);
        srand(100000*world_rank*time(NULL));
	mat orderedDNAS(DNASrows,DNAScols);
	mat newDNAS(DNASrows,DNAScols);
	double punishment = 0;

	vec v(DNASrows);
	vec accv = zeros<vec>(DNASrows);
	vec accvBuff = zeros<vec>(DNAS.n_rows);
	vec orderedV(DNASrows);


//------- PARAL ------------------------------
        for(int k=  XNrows / world_size * world_rank; k < XNrows*( world_rank + 1)/ world_size; k++){
        for(int j=0;j < DNASrows; j++){
           for(int i=0;i < DNAScols; i++){
	            net->parameters[i] = DNAS.at(j,i);
	   }
          network_compute(net, X(k, span(0,X.n_cols - 2)));
	  v.at(j) = net -> outputs[0]; //////// ----FIX HERE
	  accv.at(j) = accv.at(j) + abs(v.at(j) - X.at(k,Xcols-1)) + regu * sum(abs(DNAS.row(j))) ;
	}
	}
//------- RAL ------------------------------

if(world_rank != 0){
	   MPI_SendVec( accv, 0, 123);
	   MPI_RecvMat( newDNAS, 0, 234);
}

if(world_rank == 0){ 

	for(int i=1; i < world_size; i++) { MPI_RecvVec( accvBuff, i, 123); accv = accv + accvBuff; }


	accv=-accv;

  cout << " max average fitness Tr per sample  "  << accv.max()/Xrows <<endl;

	//accv.print("accv");

	if(accv.has_nan()){for(int i; i< accv.n_elem; i++) {if(not(accv(i) < 10000000)) accv(i) = -1e10;  }   }
	uvec q = sort_index(accv,"descend");

        for(int j=0;j < DNASrows; j++){
            orderedDNAS.row(j)= DNAS.row(q(j)); 
	    orderedV(j) = accv(q(j));
	   } 


        vec y = exp(10.0*orderedV)/sum(exp(10.0*orderedV));
        vec z = cumsum(y); 

        for(int j=0;j < 2*DNASrows; j++){
           myindex = index_min(abs(z - (double)(rand() + 1)/(RAND_MAX))  );
		if( finalizeit == 0 && myindex ==0 && (double)(rand() )/(RAND_MAX) < 0.40) myindex = index_min(abs(z - (double)(rand() + 1)/(RAND_MAX))  );
		if( finalizeit == 1 && myindex !=0 && (double)(rand() )/(RAND_MAX) < 1.0 ) myindex = index_min(abs(z - (double)(rand() + 1)/(RAND_MAX))  );
	   decision[j] = myindex;
	  }

      //BEGIN MATING RETURAL	
	for(int i=0; i < 2*DNAS.n_rows-1; i = i + 2){
		//cout << "THE I" << (i)/2 << endl;
           baby =  orderedDNAS.row(decision[i]);
	   //MIXING
	for(int j=0; j <DNAS.n_cols; j++){
         if((double)(rand() + 1)/(RAND_MAX) < 0.5) baby(j) = orderedDNAS.row(decision[i+1])(j);
	  }
	  //SMALL MUTATION
	for(int j=0; j <DNAS.n_cols; j++){
           randBuff = (double)(rand() + 1)/(RAND_MAX);
         if((double)(rand() + 1)/(RAND_MAX) < mutRat) baby(j) = baby(j) + baby(j) * mutAmmount*(2*randBuff-1);
	  }
	  //LARGE MUTATION
	for(int j=0; j <DNAS.n_cols; j++){
           randBuff = (double)(rand() + 1)/(RAND_MAX);
         if((double)(rand() + 1)/(RAND_MAX) < bigMut) baby(j) = baby(j) + baby(j) * bigMutAmmount*(2*randBuff-1);
	  }
	for(int j=0; j <DNAS.n_cols; j++){
           randBuff = (double)(rand() + 1)/(RAND_MAX);
         if((double)(rand() + 1)/(RAND_MAX) < veryBigMut) baby(j) = baby(j) + baby(j) * verBigMutAmmount*(2*randBuff-1);
	  }

	newDNAS.row(i/2) = baby; 
	}
	for(int i=1; i < world_size; i++) { MPI_SendMat(newDNAS, i, 234);};

    }


	return newDNAS;
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

//mat hw(Network* net, mat ln, cube lnu, mat xu, mat lu,int uSize,int nSize){
//  mat x = zeros<mat>(xu.n_rows,xu.n_cols);
//  vec buffxn1 = zeros<vec>(xu.n_cols);
//  vec buffxn2 = zeros<vec>(xu.n_cols);
//  vec justZeros = zeros<vec>(xu.n_cols);
//
//  double lnn  = ln.n_cols;
//  double lnun = lnu.slice(0).n_cols;
//  double xun  = xu.n_cols;
//  double lun  =lu.n_cols;
//
//  double lnnx = lnn + lnun;
//  double lnnxx = lnnx + xun;
//  double lnnxxx = lnnxx + lun;
////  cout << lnnxx << endl;
// 
//  for(int n=0;n<nSize;n++){
//    buffxn2 = justZeros;
//
//    for(int u=0;u<uSize;u++){
//        buffxn1 = justZeros;
//        if(n==u) continue;
//
//        for(int i=0;i<lnn;i++){ //??? FIX or maybe not
//           net -> inputs[i] = ln(n,i);
//        }
//
//        for(int i=lnn;i<lnnx;i++){
//         net -> inputs[i] = lnu(n,i-lnn,u);
//        }
//
//        for(int i=lnnx;i<lnnxx;i++){
//         net -> inputs[i] = xu(u,i-lnnx);
//        }
//
//        for(int i=lnnxx;i<lnnxxx;i++){
//         net -> inputs[i] = lu(u,i - lnnxx);
//        }
//           network_mlp_compute_relu_lin(net);
//
//        for(int i=0; i < xu.n_cols; i++){
//
//          buffxn1(i) = net -> outputs[i]; 
//
//          }
//        buffxn2 = buffxn2 + buffxn1;
//
//    }
//      x.row(n) = buffxn2.t();
//
//  }
//
//  return x;
//
//  }


#endif
