#ifndef myGAMPI
#define myGA7MPI

#include "GNN_interface.h"
#include <iostream>
#include <string>
#include <armadillo>
#include <time.h>
#include <mpi.h>
#include "network.h"
#include "armadilloMPI.h"

#define SIZEOF(_arr) (sizeof(_arr)/sizeof(_arr[0]))

using namespace std;
using namespace arma;

mat ga_eval(mat DNAS, double mutRat, double mutAmmount, double bigMut,double bigMutAmmount,
                                                          double veryBigMut, double verBigMutAmmount, int
                                                          finalizeit, double regu , double linContr, double squareContr, int world_rank, int world_size){
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  double box_size[3] = {7.0,7.0,7.0};
  Trajectory traj;
  traj.load("water_trajectories.dat");
  printf("Number of atoms: %d\n", traj.num_atoms);
  printf("Number of steps: %d\n", traj.num_steps);
 vec accv = zeros<vec>(DNAS.n_rows);
double Xrows = 1000;
double Xcols = 1000;

  /*Initialize the fw and gw networks!*/
  //fw
  int fw_input_size = calculateFwInputSize(NUM_EDGE_LABELS, NUM_NODE_LABELS, NODE_STATE_SIZE, traj.num_atoms-1, true);
  int fw_layer_sizes[] = {10, NODE_STATE_SIZE};
  ActivationFunction::Enum fw_activators[] = { 
        ActivationFunction::SIGMOID,
      ActivationFunction::SIGMOID};
  Network* fw = network_create(
    fw_input_size,
    SIZEOF(fw_layer_sizes), fw_layer_sizes,
    SIZEOF(fw_activators), fw_activators);
  //gw
  ActivationFunction::Enum gw_activators[] = { 
        ActivationFunction::SIGMOID, 
        ActivationFunction::SIGMOID};
    int gw_layer_sizes[] = {10,OUTPUT_SIZE};
  Network* gw = network_create(
    NODE_STATE_SIZE,
    SIZEOF(gw_layer_sizes), gw_layer_sizes,
    SIZEOF(gw_activators), gw_activators);
  //Load them with random parameters
  arma::vec fw_parameters(fw->parameter_size, arma::fill::randn);
  arma::vec gw_parameters(gw->parameter_size, arma::fill::randn);
  network_load_parameters(fw, fw_parameters);
  network_load_parameters(gw, gw_parameters);
  /*End of network initialization*/
  Frame frame;
  traj.getFrame(1, frame);
  frame.print();
  GNN* gnn = constructGNN(frame, fw, gw, box_size);
  arma::vec random_parameters(gnn->parameter_size(), arma::fill::randn);
  gnn->load_parameters(random_parameters);
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  for(int i = 0; i < traj.num_steps; ++i) {
    Frame frame;
    traj.getFrame(i, frame);
    GNN* gnn = constructGNN(frame, fw, gw, box_size);
    arma::vec random_state(gnn->getStateSize(), arma::fill::randn);
    gnn->setState(random_state.n_rows, random_state.memptr());
    for(int i = 0; i < 5; ++i) { //forloop till it converges
      double r = gnn->step();
      printf("%f\n", r); 
    }   
    arma::vec out(gnn->output_size());
    gnn->get_output(out);
    accv = abs(out - frame.forces);
  }
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  double newVal;
  int DNAScols = DNAS.n_cols;
  int DNASrows = DNAS.n_rows;
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
//	vec accv = zeros<vec>(DNASrows);
	vec accvBuff = zeros<vec>(DNAS.n_rows);
	vec orderedV(DNASrows);


//------- PARAL ------------------------------
//        for(int k=  XNrows / world_size * world_rank; k < XNrows*( world_rank + 1)/ world_size; k++){
//        for(int j=0;j < DNASrows; j++){
//           for(int i=0;i < DNAScols; i++){
//	            net->parameters[i] = DNAS.at(j,i);
//	   }
//          network_compute(net, X(k, span(0,X.n_cols - 2)));
//	  v.at(j) = net -> outputs[0]; //////// ----FIX HERE
//    newVal = abs(v.at(j) - X.at(k,Xcols-1));
//	  accv.at(j) = accv.at(j) + linContr*newVal + squareContr*pow(10*newVal,2)/10.0 + regu * sum(abs(DNAS.row(j))) ;
//	  accv.at(j) = accv.at(j) + linContr*newVal + regu * sum(abs(DNAS.row(j))) ;
//	}
//	}
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
      cout << "AAAAAAAAAAAAAAAAA" << j << endl;
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

    destroyGNN(gnn);
  network_destroy(fw);
  network_destroy(gw);

	return newDNAS;
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------


#endif
