#include <iostream>
#include <string>
#include <armadillo>
#include <time.h>
extern "C" {
#include "network.h"
}


using namespace std;
using namespace arma;

mat ga_eval(mat DNAS, Network* net,  double mutRat, double mutAmmount, double totalMutRat,double totalMutAmmount , int migration){
	vec n = zeros<vec>(DNAS.n_rows);
//	int m = 0;
	int decision[2*DNAS.n_rows];
	int myindex;
	double randBuff;
	rowvec baby(DNAS.n_cols);
        srand(time(NULL));
	mat orderedDNAS(DNAS.n_rows,DNAS.n_cols);
	mat newDNAS(DNAS.n_rows,DNAS.n_cols);

	vec v(DNAS.n_rows);

//	DNAS.load("DNAS");
	
        for(int j=0;j < DNAS.n_rows; j++){
           for(int i=0;i < DNAS.n_cols; i++){
	      net->data[i] = DNAS(j,i);
	   }
          network_mlp_compute_relu_lin(net);
	  v(j) = net -> outputs[0]; //////// ----FIX HERE
//	cout <<"aaa" << net -> outputs[0] << endl;
	}

	uvec q = sort_index(v,"descend");

        for(int j=0;j < DNAS.n_rows; j++){
           for(int i=0;i < DNAS.n_cols; i++){
            orderedDNAS(j,i) = DNAS(q(j),i); 
	    net -> data[i] = DNAS(q(j),i);
	   } 
          network_mlp_compute_relu_lin(net);
	  v(j) = net -> outputs[0]; //////// ----FIX HERE
	//cout <<"bbb" << net -> outputs[0] << endl;
	}

        vec y = exp(1.1*v)/sum(exp(1.1*v));

//	y.print("y");
        vec z = cumsum(y); 
//	z.print("cumsum");
//ERACE //// TO GET STATISTICS
////	for(int i = 0; i < DNAS.n_rows; i++){

//	cout <<	index_min(abs(z - (double)rand()/(RAND_MAX))  ) << endl; //!!!!!!!MUST BE REPLACED SOMEDAY SOON!!!!

////        for(int j=0;j < DNAS.n_rows; j++){
        for(int j=0;j < 2*DNAS.n_rows; j++){
           myindex = index_min(abs(z - (double)(rand() + 1)/(RAND_MAX))  );
		if(myindex ==0 && (double)(rand() )/(RAND_MAX) < 0.40) myindex = index_min(abs(z - (double)(rand() + 1)/(RAND_MAX))  );
////             if(j == myindex) n(j)++;
	   //cout << "Ind: " <<  myindex << endl;
	   decision[j] = myindex;
	  }
////	}

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
         if((double)(rand() + 1)/(RAND_MAX) < totalMutRat) baby(j) =  totalMutAmmount*(2*randBuff-1);
	  }
//	cout << size(baby) << size(newDNAS.row(i/2) ) << endl;
	newDNAS.row(i/2) = baby; 
	}

	for(int i=DNAS.n_rows - migration; i<DNAS.n_rows; i++){

             newDNAS.row(i) = randu<rowvec>(DNAS.n_cols); 

	}

//        DNAS.print("DNAS");
//	orderedDNAS.print("ORDERED: ");
//	newDNAS.print("MYNEW: ");

        
////	cout << "n " <<(double) sum(n)/DNAS.n_rows/2 << endl;//Checking Distribution....;
////	cout << "n0 " <<(double) n(0)/DNAS.n_rows/2 << endl;//Checking Distribution....;
////	cout << "y0 " <<(double) y(0)<< endl;//Checking Distribution....;
////	cout << "n1 " <<(double) n(1)/DNAS.n_rows/2 << endl;// Checking Distribution....
////	cout << "y1 " <<(double) y(1)<< endl;//Checking Distribution....;
        n.save("myn",raw_ascii);
	//cout << "sum" << sum(y)<< endl;
	

	return newDNAS;

}


int main(){

     srand(time(NULL));
     Network *SampNet;
     int sizes[3] = {2,2,1};

	mat X;
	X.load("X.dat");

     Network *net = network_mlp_create(sizes,3);
	    

    int netDataSize = net -> datasize;
			      
  double* dna = new double [netDataSize];

  mat DNAS = randu<mat>(1100,netDataSize); 


					      
											      
     network_mlp_loaddata(net, dna);
												      
     network_mlp_compute_relu_lin(net);
     //cout << net -> outputs[0] << endl; 

     for(int i; i < 1000; i++){
       DNAS = ga_eval(DNAS, net,0.1, 0.1, 0.001, 5.0, 10);
     }
														      
      DNAS.save("DNAS");
      DNAS.save("DNAS.dat",raw_ascii);
															      
return 0;

}







