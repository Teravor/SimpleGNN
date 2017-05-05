#include <armadillo>
#include "GNN.h"

#define NODE_STATE_SIZE 3
#define NUM_EDGE_LABELS 2
#define NUM_NODE_LABELS 1
#define OUTPUT_SIZE 3

struct Trajectory{
	Trajectory();
	~Trajectory();

	void load(const char* _path, const double* _box_size);
	void destroy();
	int numSteps() const;
	void print(int _atom_index, int _step);
	double distance(const double* ai, const double* aj) const;

	int num_steps;
	int num_atoms;
	int* atoms;
	double* positions;
	double* forces;
	double box_size[3];
	double r_box_size[3];

	template<size_t N>
	void load(const char* _path, const double (&x)[N]) {
		static_assert(N == 3, "box is three dimensional orthorhombic box");
		load(_path, x);
	}
};

int calculateFwInputSize(int num_edge_labels, int num_node_labels, int node_state_size, int max_neighbors, bool positional);
void constructGraph(const Trajectory& _traj, int _index, GraphHelper& _helper);
GNN* constructGNN(const Trajectory& _traj, int _index,  Network* _fw, Network* _gw);
void destroyGNN(GNN* _gnn);