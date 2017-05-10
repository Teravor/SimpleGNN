#include "GNN_interface.h"
#include <iostream>

#define SIZEOF(_arr) (sizeof(_arr)/sizeof(_arr[0]))

int main() {
	double box_size[3] = {7.0,7.0,7.0};
	Trajectory traj;
	traj.load("water_trajectories.dat");
	printf("Number of atoms: %d\n", traj.num_atoms);
	printf("Number of steps: %d\n", traj.num_steps);
	//traj.print(0,0);

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
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	for(int i = 0; i < traj.num_steps; ++i) {
		Frame frame;
		traj.getFrame(i, frame);
		GNN* gnn = constructGNN(frame, fw, gw, box_size);
		arma::vec random_state(gnn->getStateSize(), arma::fill::randn);
		gnn->setState(random_state.n_rows, random_state.memptr());
		for(int i = 0; i < 5; ++i) {
			double r = gnn->step();
			printf("%f\n", r);
		}
		arma::vec out(gnn->output_size());
		gnn->get_output(out);
		arma::vec R = out - frame.forces;
		destroyGNN(gnn);
	}
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	Frame frame;
	traj.getFrame(1, frame);
	frame.print();
	GNN* gnn = constructGNN(frame, fw, gw, box_size);
	//gnn->graph.debug(std::cout, 38);
	//gnn->graph.debug(std::cout);
	arma::vec random_parameters(gnn->parameter_size(), arma::fill::randn);
	gnn->load_parameters(random_parameters);

	//Random initial size
	arma::vec random_state(gnn->getStateSize(), arma::fill::randn);
	gnn->setState(random_state.n_rows, random_state.memptr());
	for(int i = 0; i < 5; ++i) {
		double r = gnn->step();
		printf("%f\n", r);
	}
	arma::vec out(gnn->output_size());
	gnn->get_output(out);

	destroyGNN(gnn);
	network_destroy(fw);
	network_destroy(gw);
	
	return 0;
}
