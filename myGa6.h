#ifndef myGA6
#define myGA6

#include <iostream>
#include <string>
#include <armadillo>
#include <time.h>
extern "C" {
#include "network.h"
}


using namespace std;
using namespace arma;

//----------------------------------------------------------------------
vec getVals(mat DNAS, Network* net, rowvec inp){
	 vec vals(DNAS.n_rows);
            for(int l=0;l < inp.n_elem - 1; l++){
             	  net -> inputs[l] = inp(l);}
           for(int j=0;j < DNAS.n_rows; j++){
           for(int i=0;i < DNAS.n_cols; i++){ //???
	              net->data[i] = DNAS(j,i);
	          }
           network_mlp_compute_relu_lin(net);
	            vals(j) = net -> outputs[0];
	         }
	   return vals;
       }


//----------------------------------------------------------------------
double getVal(vec v, Network* net, rowvec inp){
            for(int l=0;l < inp.n_elem - 1; l++){
	      net -> inputs[l] = inp(l);}
           for(int i=0;i < v.n_elem; i++){
	      net->data[i] = v(i);
	   }
           network_mlp_compute_relu_lin(net);
	   return net -> outputs[0] - inp(210);
       }
//----------------------------------------------------------------------
vec getBest(mat DNAS, Network* net, mat X){
	vec accv = zeros<vec>(DNAS.n_rows);
	vec v(DNAS.n_rows);
        for(int k=0;k < X.n_rows; k++){
            for(int l=0;l < X.n_cols - 1; l++){
	  net -> inputs[l] = X(k,l);}
        for(int j=0;j < DNAS.n_rows; j++){
           for(int i=0;i < DNAS.n_cols; i++){ //???
	      net->data[i] = DNAS(j,i);
	   }
          network_mlp_compute_relu_lin(net);
	  v(j) = net -> outputs[0]; //////// ----FIX HERE
	  accv(j) = accv(j) + abs(v(j) - X(k,X.n_cols-1));
	}
}

	accv = - accv;

        cout << "max fitness per sample: " << accv.max()/X.n_rows << endl;
	return DNAS.row(index_max(accv)).t();

        }

//----------------------------------------------------------------------
//mat ga_eval(mat DNAS, Network* net, mat X ,  double mutRat, double mutAmmount, double totalMutRat,double totalMutAmmount , int migration, int finalizeit, vec winning){
mat ga_eval(mat DNAS, Network* net, mat X ,  double mutRat, double mutAmmount, double bigMut,double bigMutAmmount ,
                                                          double veryBigMut, double verBigMutAmmount, int finalizeit, vec winning){

	vec n = zeros<vec>(DNAS.n_rows);
	int decision[2*DNAS.n_rows];
	int myindex;
	double randBuff;
	rowvec baby(DNAS.n_cols);
        srand(time(NULL)*time(NULL));
	mat orderedDNAS(DNAS.n_rows,DNAS.n_cols);
	mat newDNAS(DNAS.n_rows,DNAS.n_cols);
	double punishment = 0;

	vec v(DNAS.n_rows);
	vec accv = zeros<vec>(DNAS.n_rows);
	vec orderedV(DNAS.n_rows);


        for(int k=0;k < X.n_rows; k++){
            for(int l=0;l < X.n_cols - 1; l++){
	  net -> inputs[l] = X(k,l);}
        for(int j=0;j < DNAS.n_rows; j++){
           for(int i=0;i < DNAS.n_cols; i++){
	      net->data[i] = DNAS(j,i);
	   }
          network_mlp_compute_relu_lin(net);
	  v(j) = net -> outputs[0]; //////// ----FIX HERE
	  accv(j) = accv(j) + abs(v(j) - X(k,X.n_cols-1));
	}
	}

	accv=-accv;


	//accv.print("accv");

	if(accv.has_nan()){for(int i; i< accv.n_elem; i++) {if(not(accv(i) < 10000000)) accv(i) = -1e10;  }   }
	uvec q = sort_index(accv,"descend");

        for(int j=0;j < DNAS.n_rows; j++){
            orderedDNAS.row(j)= DNAS.row(q(j)); 
	    orderedV(j) = accv(q(j));
	   } 


        vec y = exp(10.0*orderedV)/sum(exp(10.0*orderedV));
        vec z = cumsum(y); 

        for(int j=0;j < 2*DNAS.n_rows; j++){
           myindex = index_min(abs(z - (double)(rand() + 1)/(RAND_MAX))  );
		if(myindex ==0 && (double)(rand() )/(RAND_MAX) < 0.40) myindex = index_min(abs(z - (double)(rand() + 1)/(RAND_MAX))  );
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

	if(finalizeit == 1){
           DNAS.row(DNAS.n_rows - 1) = winning.t(); 
	
	}

	return newDNAS;
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

mat hw(Network* net, mat ln, cube lnu, mat xu, mat lu,int uSize,int nSize){
  mat x = zeros<mat>(xu.n_rows,xu.n_cols);
  vec buffxn1 = zeros<vec>(xu.n_cols);
  vec buffxn2 = zeros<vec>(xu.n_cols);
  vec justZeros = zeros<vec>(xu.n_cols);

  double lnn  = ln.n_cols;
  double lnun = lnu.slice(0).n_cols;
  double xun  = xu.n_cols;
  double lun  =lu.n_cols;

  double lnnx = lnn + lnun;
  double lnnxx = lnnx + xun;
  double lnnxxx = lnnxx + lun;
//  cout << lnnxx << endl;
 
  for(int n=0;n<nSize;n++){
    buffxn2 = justZeros;

    for(int u=0;u<uSize;u++){
        buffxn1 = justZeros;
        if(n==u) continue;

        for(int i=0;i<lnn;i++){ //??? FIX or maybe not
           net -> inputs[i] = ln(n,i);
        }

        for(int i=lnn;i<lnnx;i++){
         net -> inputs[i] = lnu(n,i-lnn,u);
        }

        for(int i=lnnx;i<lnnxx;i++){
         net -> inputs[i] = xu(u,i-lnnx);
        }

        for(int i=lnnxx;i<lnnxxx;i++){
         net -> inputs[i] = lu(u,i - lnnxx);
        }
           network_mlp_compute_relu_lin(net);

        for(int i=0; i < xu.n_cols; i++){

          buffxn1(i) = net -> outputs[i]; 

          }
        buffxn2 = buffxn2 + buffxn1;

    }
      x.row(n) = buffxn2.t();

  }

  return x;

  }


#endif
