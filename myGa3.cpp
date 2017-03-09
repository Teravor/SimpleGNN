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
           for(int i=0;i < DNAS.n_cols; i++){
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
	   return net -> outputs[0];
       }

//----------------------------------------------------------------------
vec getBest(mat DNAS, Network* net, mat X){
	vec accv = zeros<vec>(DNAS.n_rows);
	vec v(DNAS.n_rows);
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

	accv = - accv;

//	v = -v%v;
        cout << "max fitness: " << accv.max() << endl;
	return DNAS.row(index_max(accv)).t();

        }

//----------------------------------------------------------------------
mat ga_eval(mat DNAS, Network* net, mat X ,  double mutRat, double mutAmmount, double totalMutRat,double totalMutAmmount , int migration, int finalizeit, vec winning){
	totalMutAmmount = totalMutAmmount*abs(DNAS).max();

	vec n = zeros<vec>(DNAS.n_rows);
	int decision[2*DNAS.n_rows];
	int myindex;
	double randBuff;
	rowvec baby(DNAS.n_cols);
        srand(time(NULL)*time(NULL));
	mat orderedDNAS(DNAS.n_rows,DNAS.n_cols);
	mat newDNAS(DNAS.n_rows,DNAS.n_cols);
	double punishment = 0;
//	double vSum;

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

//	orderedV.print("ORDEREDV");

        vec y = exp(10.0*orderedV)/sum(exp(10.0*orderedV));
//	y.print("y");
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
         if((double)(rand() + 1)/(RAND_MAX) < mutRat) baby(j) = baby(j) + mutAmmount*(2*randBuff-1);
	  }
	  //LARGE MUTATION
	for(int j=0; j <DNAS.n_cols; j++){
           randBuff = (double)(rand() + 1)/(RAND_MAX);
//	   cout << totalMutAmmount<< endl;
         if((double)(rand() + 1)/(RAND_MAX) < totalMutRat) baby(j) =  totalMutAmmount*(2*randBuff-1);
	  }
//	cout << size(baby) << size(newDNAS.row(i/2) ) << endl;
	newDNAS.row(i/2) = baby; 
	}

	for(int i=DNAS.n_rows - migration; i<DNAS.n_rows; i++){

             newDNAS.row(i) = randu<rowvec>(DNAS.n_cols); 

	}
	if(finalizeit == 1){
           DNAS.row(DNAS.n_rows-migration - 1) = winning.t(); 
	
	}

	return newDNAS;

}


//----------------------------------------------------------------------


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







