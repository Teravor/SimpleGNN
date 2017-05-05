#include "GNN_interface.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

int symbol2number(const char* _symbol) {
	switch(_symbol[0]) {
		case 'O':
		return 8;
		case 'H':
		return 1;
		default:
		fprintf(stderr, "Unspecified atomic symbol: %s\n", _symbol);
		abort();
	}
}

Trajectory::Trajectory()
	: atoms(NULL), positions(NULL), forces(NULL),
	num_atoms(0), num_steps(0)
	{}

Trajectory::~Trajectory() {
	destroy();
}

void Trajectory::load(const char* _path, const double* _box_size) {
	FILE * pFile;
	pFile = fopen (_path,"r");
	if(pFile == NULL) {
		fprintf(stderr, "Unable to open file %s\n", _path);
	}
	char buffer[2048];
	int ok;
	fgets(buffer, sizeof(buffer), pFile);
	printf("%s\n",buffer);
	fscanf(pFile, "%*s %*s %*s %d", &num_atoms);
	fscanf(pFile, "%*s %*s %*s %*s %d", &num_steps);
	assert(num_steps > 0);
	atoms = new int[num_atoms];
	positions = new double[num_atoms*num_steps*3];
	forces = new double[num_atoms*num_steps*3];

	fgets(buffer, sizeof(buffer), pFile);
	size_t pos = ftell(pFile);
	//Read atomic types
	fgets(buffer, sizeof(buffer), pFile);
	for(int i = 0; i < num_atoms; ++i) {
		fscanf(pFile, "%s %*f %*f %*f %*f %*f %*f", buffer);
		atoms[i] = symbol2number(buffer);
	}
	//Read in positions and forces at each step
	fseek(pFile, pos, SEEK_SET);
	double* a_pos = positions;
	double* a_force = forces;
	for(int i = 0; i < num_steps; ++i) {
		fgets(buffer, sizeof(buffer), pFile);
		for(int j = 0; j < num_atoms; ++j) {
			fscanf(pFile, "%*s %lf %lf %lf %lf %lf %lf",a_pos, a_pos+1, a_pos+2, a_force, a_force+1,a_force+2);
			a_pos += 3;
			a_force += 3;
		}
		fgets(buffer, sizeof(buffer), pFile);
	}
	fclose(pFile);
	memcpy(box_size, _box_size, 3*sizeof(double));
	for(int i = 0; i < 3; ++i)
		r_box_size[i] = 1.0/box_size[i];
}

double Trajectory::distance(const double* ai, const double* aj) const {
	arma::vec vector(3);
	for(int d = 0; d < 3; ++d) {
		double dd = abs(aj[d] - ai[d]);
		dd -= static_cast<int>(dd * r_box_size[d] +  0.5) * box_size[d];
		vector[d] = dd;
	}
	return arma::norm(vector);
}

void Trajectory::destroy() {
	if(atoms != NULL) {
		delete[] atoms;
		delete[] positions;
		delete[] forces;
	}
}


void Trajectory::print(int _atom_index, int _step) {
	assert(_atom_index < num_atoms);
	assert(_step < num_steps);
	printf("Z: %d\n", atoms[_atom_index]);
	double* pos = positions + 3*(_step*num_atoms + _atom_index);
	printf("Position: (%lf, %lf, %lf)\n", pos[0], pos[1], pos[2]);
	double* force = forces + 3*(_step*num_atoms + _atom_index);
	printf("Force: (%lf, %lf, %lf)\n", force[0], force[1], force[2]);
}


int calculateFwInputSize(int num_edge_labels, int num_node_labels, int node_state_size, int max_neighbors, bool positional) {
	if(positional) {
		return num_node_labels + max_neighbors*(num_edge_labels + num_node_labels + node_state_size + 3);
	}
	else {
		return  2*num_node_labels + num_edge_labels + node_state_size;
	}
}

/*
node label size is one (Z) and edge label size is two (1/r, Z_i*Z_j/r)
*/
void constructGraph(const Trajectory& _traj, int _index, GraphHelper& _helper) {
	assert(NUM_EDGE_LABELS == 2);
	assert(NUM_NODE_LABELS == 1);
	assert(_index < _traj.num_steps);
	double* pos = _traj.positions + 3*_index*_traj.num_atoms;
	for(int i = 0; i < _traj.num_atoms; ++i) {
		_helper.addNode({(double)_traj.atoms[i]});
		for(int j = 0; j < i; ++j) {
			double dist = _traj.distance(pos + 3*i, pos + 3*j);
			double rdist = 1.0/dist;
			double coulomb_rdist = _traj.atoms[i]*_traj.atoms[j] * rdist;
			_helper.addEdge(j,i, {dist, coulomb_rdist});
		}
	}
}

GNN* constructGNN(const Trajectory& _traj, int _index, Network* _fw, Network* _gw) {
	GraphHelper helper(1,2, false);
	constructGraph(_traj, _index, helper);
	GNN* gnn = new GNN(helper, NODE_STATE_SIZE, _fw, _gw);
	return gnn;
}
void destroyGNN(GNN* _gnn) {
	delete _gnn;
}