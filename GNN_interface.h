#include <armadillo>
#include "GNN.h"

#define NODE_STATE_SIZE 3
#define NUM_EDGE_LABELS 1
#define NUM_NODE_LABELS 4
#define OUTPUT_SIZE 3

struct Frame {
	int num_atoms;
	int* atomic_numbers;
	arma::vec positions;
	arma::vec forces;
	void print();
};

struct Trajectory{
	Trajectory();
	~Trajectory();

	void load(const char* _path);
	
	void destroy();
	int numSteps() const;
	void print(int _atom_index, int _step);
	double distance(const double* ai, const double* aj) const;
	void getFrame(int _index, Frame& _frame);

	int num_steps;
	int num_atoms;
	int* atoms;
	double* positions;
	double* forces;
	double box_size[3];
	double r_box_size[3];
};

int calculateFwInputSize(int num_edge_labels, int num_node_labels, int node_state_size, int max_neighbors, bool positional);
void constructGraph(const Frame& _frame, GraphHelper& _helper);

/*These are the main functions*/
GNN* constructGNN(const Frame& _frame,  Network* _fw, Network* _gw, double* _box_size);
void destroyGNN(GNN* _gnn);