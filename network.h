#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#ifndef NETWORK
#define NETWORK

#define true 1
#define false 0



///Data structure for a neural layer
typedef struct Layer {

  /// Number of neurons in the layer
  int nN;
  /// Number of inputs connected to the layer
  int nInputs;

  /// Pointer to the input values
  double *inputs;
  
  /// Neuron outputs
  double *outputs;

  double *weights;
  double *biases;
  
  /// Scale for parameters
  double pscale;
  
  
} Layer;

typedef struct Network {

	///number of inputs, outputs and hidden layers
	int nInputs;
	int nOutputs;
	int nHiddenLayers;

	int *layersize;
	
	/// Pointer to the inputs array
	double *inputs;
	double *outputs;

	/// Network weights, biases
	double *data;
	int datasize;
	int alloc_data;

	double *service;
	int nActivs;
	
	/// Number of layers
	int nLayers;
	/// List of layers
	Layer *layers;
  
} Network;

typedef struct RecurrentNetwork {
	
	int nInputs;
	int nOutputs;
	int nHidden;
	int nNeurons; //including in, out and hidden
	
	double *inputs, *outputs, *states, *buffer;
	
	double *data;
	double *weights, *biases;
	
	double tolerance, delta;
	
	int alloc_data;
	
} RecurrentNetwork;

RecurrentNetwork* rnn_create(int nin, int nout, int nhid);
void rnn_randomise(RecurrentNetwork *rnn);
void rnn_load(RecurrentNetwork *rnn, double *data);
void rnn_clean(RecurrentNetwork *rnn);
void rnn_reset(RecurrentNetwork *rnn);

void rnn_update_tanh_tanh(RecurrentNetwork *rnn);

void rnn_scf(RecurrentNetwork *rnn, int maxIter);



Layer* layer_add(Network*, int, int, double pscale);
void layer_randomise(Layer* layer);
void layer_compute(Layer*);

double network_relu(double arg);
double network_tanh(double arg);


Network* network_create(int inputs);
void network_compute(Network *);
void network_mlp_compute_gaus_lin(Network* net);
void network_mlp_compute_relu_lin(Network* net);
void network_mlp_compute_relu_tanh(Network* net);
void network_mlp_compute_tanh_tanh(Network* net);
void network_mlp_compute_tanh_lin(Network* net);


extern Network* network_mlp_create(int *layersize, int nlayers);
extern void network_mlp_randomise(Network *net, double pscale);
extern void network_mlp_loaddata(Network *net, double *data);
extern void network_mlp_compute(Network *net);
extern void network_mlp_clean(Network *net);




//MLPflex




#endif

