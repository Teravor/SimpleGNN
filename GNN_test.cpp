#include "GNN.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <armadillo>
#include <assert.h>

#define SIZEOF(_arr) (sizeof(_arr)/sizeof(_arr[0]))

int helper_test() {
	GraphHelper helper(2,1);
	helper.addNode({0.5, 0.1});
	helper.addNode({0.2,0.2});
	helper.addNode({0.3, 0.6});
	helper.addEdge(0,1, {0.1});
	helper.addEdge(1,2, {0.4});
	//helper.addEdge(0,2, {0.7});
	GNNGraph graph(helper);
	graph.debug(std::cout);
	//GNN gnn(helper);
	bool complete_graph = helper.isCompleteGraph();
	int network_input_size = helper.getNetworkInputSize(1);
	printf("Graph is complete: %s. Network input size: %d\n",
		complete_graph ? "true" : "false", network_input_size);
	return 0;
}

void create_random_complete_graph(
	GraphHelper& _helper,
	int num_nodes, 
	int edge_label_size, 
	int node_label_size) 
{
	for(int i = 0; i < num_nodes; ++i) {
		arma::vec node_label(node_label_size, arma::fill::randn);
		_helper.addNode(node_label.n_rows, node_label.memptr());
		for(int j = 0; j < i; ++j) {
			arma::vec edge_label(edge_label_size, arma::fill::randn);
			_helper.addEdge(j,i,edge_label.n_rows, edge_label.memptr());
		}
	}
	assert(_helper.isCompleteGraph());
}

void initialize_networks(GraphHelper& helper, int node_state_size, int output_size,
	Network*& fw, Network*& gw) 
{
	int network_input_size = helper.getNetworkInputSize(node_state_size);
	//State transition network f_w
	ActivationFunction::Enum fw_activators[] = {
        ActivationFunction::SIGMOID, 
        ActivationFunction::SIGMOID, 
        ActivationFunction::SIGMOID };
	int fw_layer_sizes[] = {10,10, node_state_size};
	fw = network_create(
		network_input_size,
		SIZEOF(fw_layer_sizes), fw_layer_sizes,
		SIZEOF(fw_activators), fw_activators);
	//Output network g_w
	ActivationFunction::Enum gw_activators[] = {
        ActivationFunction::SIGMOID, 
        ActivationFunction::SIGMOID};
    int gw_layer_sizes[] = {10,output_size};
	gw = network_create(
		node_state_size,
		SIZEOF(gw_layer_sizes), gw_layer_sizes,
		SIZEOF(gw_activators), gw_activators);
	//Load the networks with random parameters
	arma::vec fw_parameters(fw->parameter_size, arma::fill::randn);
	arma::vec gw_parameters(gw->parameter_size, arma::fill::randn);
	network_load_parameters(fw, fw_parameters);
	network_load_parameters(gw, gw_parameters);
}

int simple_gnn_test() {
	/*Network dependent*/
	const int node_state_size = 1;
	const int output_size = 1;
	//Create the graph with some random labels for nodes and edges
	GraphHelper helper(2,1);
	helper.addNode({0.5, 0.1});
	helper.addNode({0.2,0.2});
	helper.addNode({0.3, 0.6});
	helper.addEdge(0,1, {0.1});
	helper.addEdge(1,2, {0.4});
	helper.addEdge(0,2, {0.7});
	int network_input_size = helper.getNetworkInputSize(node_state_size);
	//State transition network f_w
	ActivationFunction::Enum fw_activators[] = {
        ActivationFunction::SIGMOID, 
        ActivationFunction::SIGMOID, 
        ActivationFunction::SIGMOID };
	int fw_layer_sizes[] = {10,10, node_state_size};
	Network* fw = network_create(
		network_input_size,
		SIZEOF(fw_layer_sizes), fw_layer_sizes,
		SIZEOF(fw_activators), fw_activators);
	//Output network g_w
	ActivationFunction::Enum gw_activators[] = {
        ActivationFunction::SIGMOID, 
        ActivationFunction::SIGMOID};
    int gw_layer_sizes[] = {10,output_size};
	Network* gw = network_create(
		node_state_size,
		SIZEOF(gw_layer_sizes), gw_layer_sizes,
		SIZEOF(gw_activators), gw_activators);
	//Load the networks with random parameters
	arma::vec fw_parameters(fw->parameter_size, arma::fill::randn);
	arma::vec gw_parameters(gw->parameter_size, arma::fill::randn);
	network_load_parameters(fw, fw_parameters);
	network_load_parameters(gw, gw_parameters);

	GNN gnn(helper, node_state_size, fw, gw);
	//Load random initial state
	arma::vec random_state(gnn.getStateSize(), arma::fill::randn);
	gnn.setState(random_state.n_rows, random_state.memptr());
	//gnn.printState(std::cout);
	for(int i = 0; i < 5; ++i) {
		double err = gnn.step();
		printf("Iteration %d, Error: %.6f\n", i, err);
		//gnn.printState(std::cout);
	}
	network_destroy(fw);
	network_destroy(gw);
	return 0;
}

int complete_graph_gnn_test_nonpositional() {
	const int num_nodes = 10;
	const int edge_label_size = 2;
	const int node_label_size = 2;
	const int node_state_size = 2;
	const int output_size = 1;
	GraphHelper helper(node_label_size, edge_label_size);
	create_random_complete_graph(helper, num_nodes, edge_label_size, node_label_size);
	Network* fw = NULL;
	Network* gw = NULL;
	initialize_networks(helper, node_state_size, output_size, fw, gw);
	GNN gnn(helper, node_state_size, fw, gw);
	//Load random initial state
	arma::vec random_state(gnn.getStateSize(), arma::fill::randn);
	for(int i = 0; i < 10; ++i) {
		double err = gnn.step();
		printf("Iteration %d, Error: %.6f\n", i, err);
		gnn.printState(std::cout);
	}
	network_destroy(fw);
	network_destroy(gw);
	return 0;
}

int gnn_test_positional() {
	const int num_nodes = 10;
	const int edge_label_size = 2;
	const int node_label_size = 3;
	const int node_state_size = 2;
	const int output_size = 1;
	GraphHelper helper(node_label_size, edge_label_size, true);
	create_random_complete_graph(helper, num_nodes, edge_label_size, node_label_size);
	Network* fw = NULL;
	Network* gw = NULL;
	initialize_networks(helper, node_state_size, output_size, fw, gw);
	GNN gnn(helper, node_state_size, fw, gw);
	arma::vec random_state(gnn.getStateSize(), arma::fill::randn);
	for(int i = 0; i < 10; ++i) {
		double err = gnn.step();
		printf("Iteration %d, Error: %.6f\n", i, err);
		gnn.printState(std::cout);
	}
	network_destroy(fw);
	network_destroy(gw);
	return 0;
}

int main() {
	//simple_gnn_test();
	//complete_graph_gnn_test_nonpositional();
	gnn_test_positional();
}