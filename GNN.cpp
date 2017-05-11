#include "GNN.h"
#include <assert.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <armadillo>

void Box::init(double x, double y, double z) {
    box_size[0] = x;
    box_size[1] = y;
    box_size[2] = z;
    r_box_size[0] = 1.0/x;
    r_box_size[1] = 1.0/y;
    r_box_size[2] = 1.0/z;
}

double Box::distance(const double* v1, const double* v2) const {
    double dist = 0.0;
    for(int d = 0; d < 3; ++d) {
        //double dd = std::abs(v1[d] - v2[d]);
        //dd -= ((double)static_cast<int>(dd * r_box_size[d] +  0.5)) * box_size[d];
        double dd = std::abs(v1[d] - v2[d])*r_box_size[d];
        dd -= ((double)static_cast<int>(dd +  0.5));
        dist += dd*dd;
    }
    return std::sqrt(dist);
}

double Box::relative_pos(double* relative, const double* origo, double* neighbor) {
    for(int d = 0; d < 3; ++d) {
        relative[d] = (neighbor - origo)*r_box_size[d];
        if(relative[d] < -0.5)
            relative[d] += 1.0;
        else if(relative[d] > 0.5)
            relative[d] -= 1.0;
        //relative[d] *= box_size[d];
    }
}



//Store edges to hash map with this index
union EdgeIndex {
    //Guaranteed invariant, node1 <= node2
    struct Index {
        int32_t node1 : 16;
        int32_t node2 : 16;
    };
    
    Index idx;
    int32_t as_int;

    EdgeIndex(int _node1, int _node2) {
        if(_node1 > _node2)
            std::swap(_node1, _node2);
        idx.node1 = _node1;
        idx.node2 = _node2;
    }
};

GraphHelper::GraphHelper(int _nodeLabelSize, int _edgeLabelSize, bool _has_position, double* _box_size)
    : nodeLabelSize(_nodeLabelSize), edgeLabelSize(_edgeLabelSize), 
    n_nodes(0), n_edges(0), has_position(_has_position)
{
    if(_has_position) {
        assert(nodeLabelSize >= 3);
        assert(_box_size != nullptr);
        box.init(_box_size[0], _box_size[1], _box_size[2]);
    }
}

int GraphHelper::addNode(int _size, const double* _nodeLabel) {
    assert(_size == nodeLabelSize);
    int node_index = n_nodes;
    for(int i = 0; i < _size; ++i)
        node_labels.push_back(_nodeLabel[i]);
    n_nodes++;
    return node_index;
}

void GraphHelper::addEdge(int node1, int node2, int _size, const double* _edgeLabel) {
    assert(_size == edgeLabelSize);
    assert(node1 < n_nodes);
    assert(node2 < n_nodes);
    EdgeIndex idx(node1, node2);
    auto it = edge_indices.find(idx.as_int);
    if(it != edge_indices.end()) {
        return; /*Warning here?, edge already exists*/
    }
    edge_indices[idx.as_int] = n_edges;
    for(int i = 0; i < _size; ++i)
        edge_labels.push_back(_edgeLabel[i]);
    n_edges++;
}

int GraphHelper::numNeighbors(int node) {
    int num_neighbors = 0;
    for(int i = 0; i < n_nodes; ++i) {
        EdgeIndex idx(node, i);
        if(edge_indices.find(idx.as_int) != edge_indices.end())
            num_neighbors++;
    }
    return num_neighbors;
}

bool GraphHelper::hasEdge(int node1, int node2) {
    EdgeIndex idx(node1, node2);
    return edge_indices.find(idx.as_int) != edge_indices.end();
}

int GraphHelper::getEdgeIndex(int node1, int node2) {
    EdgeIndex idx(node1, node2);
    //return edge_indices.at(idx.as_int);
    return edge_indices[idx.as_int];
}

bool GraphHelper::isCompleteGraph() {
    bool is_complete = true;
    for(int i = 0; i < n_nodes; ++i) {
        if(numNeighbors(i) != n_nodes-1) {
            is_complete = false;
            break;
        }
    }
    return is_complete;
}

int GraphHelper::maxNeighbors() {
    int max_neighbors = 0;
    for(int i = 0; i < n_nodes; ++i) {
        const int num = numNeighbors(i);
        max_neighbors = num > max_neighbors ? num : max_neighbors;
    }
    return max_neighbors;
}

int GraphHelper::getNetworkInputSize(int _node_state_size) {
    if(has_position) {
        const int max_n = maxNeighbors();
        const int max_input = nodeLabelSize + max_n*(_node_state_size + nodeLabelSize + edgeLabelSize);
        return max_input;
    }
    else {
        const int input_size = nodeLabelSize + nodeLabelSize + edgeLabelSize + _node_state_size;
        return input_size;
    }
}

GNNGraph::GNNGraph(GraphHelper& _helper) {
    create(_helper);
}
GNNGraph::~GNNGraph() {
    destroy();
}

void GNNGraph::create(GraphHelper& _helper) {
    n_nodes = _helper.n_nodes;
    n_edges = _helper.n_edges;
    n_neighbors = new int[n_nodes+1];
    int neighbor_size = 0;
    for(int i = 0; i < n_nodes; ++i) {
        int num_neighbors = _helper.numNeighbors(i);

        n_neighbors[i] = neighbor_size;
        neighbor_size += num_neighbors;
    }
    n_neighbors[n_nodes] = neighbor_size;
    neighbors = new int[neighbor_size];
    edges = new int[neighbor_size];
    
    for(int i = 0; i < n_nodes; ++i) {
        int* neighbor_ptr = neighbors + n_neighbors[i];
        int* edge_ptr = edges + n_neighbors[i];
        int n_counter = 0;
        for(int j = 0; j < n_nodes; ++j) {
            if(_helper.hasEdge(i,j)) {
                neighbor_ptr[n_counter] = j;
                edge_ptr[n_counter] = _helper.getEdgeIndex(i,j);
                n_counter++;
            }
        }
    }
    
    node_label_size = _helper.nodeLabelSize;
    edge_label_size = _helper.edgeLabelSize;
    node_labels = new double[_helper.node_labels.size()];
    edge_labels = new double[_helper.edge_labels.size()];
    memcpy(node_labels, _helper.node_labels.data(), 
        sizeof(double)*_helper.node_labels.size());
    memcpy(edge_labels, _helper.edge_labels.data(),
        sizeof(double)*_helper.edge_labels.size());
    if(_helper.has_position)
        sortPositions(_helper.box);
}

void GNNGraph::destroy() {
    if(n_neighbors != nullptr) {
        delete[] n_neighbors;
        delete[] neighbors;
        delete[] edges;
        delete[] node_labels;
        delete[] edge_labels;
    }
}

void GNNGraph::sortPositions(const Box& _box) {
    for(int i = 0; i < n_nodes; ++i) {
        const int num_neighbors = n_neighbors[i+1] - n_neighbors[i];
        const int n_index = n_neighbors[i];
        const double* label = node_labels + node_label_size*i;
        //Calculate distances
        std::vector<double> distances;
        arma::vec node_pos({label[0], label[1], label[2]});
        for(int j = 0; j < num_neighbors; ++j) {
            const double* n_label = node_labels + node_label_size*neighbors[n_index+j];
            arma::vec neighbor_pos({n_label[0], n_label[1], n_label[2]});
            distances.push_back(_box.distance(n_label, label));
        }
        /*
        std::cout << "Sorting for node " << i << std::endl;
        for(int j = 0; j < num_neighbors; ++j) {
            std::cout << neighbors[n_index + j] << " " << distances[j] << std::endl;
        }
        */
        //Sort according to distances
        std::sort(&neighbors[n_index], &neighbors[n_index] + num_neighbors,
            [&distances](size_t i1, size_t i2) {return distances[i1] < distances[i2];});
        /*
        for(int j = 0; j < num_neighbors; ++j) {
            std::cout << neighbors[n_index + j] << std::endl;
        }
        */
    }
}

int GNNGraph::maxNeighbors() {
    int max = 0;
    for(int i = 0; i < n_nodes; ++i) {
        const int num = n_neighbors[i+1] - n_neighbors[i];
        max = num > max ? num : max;
    }
    return max;
}

void GNNGraph::getNode(int _node_index, Node& _node) {
    _node.node = _node_index;
    _node.num_neighbors = n_neighbors[_node_index+1] - n_neighbors[_node_index];
    const int n_index = n_neighbors[_node_index];
    _node.neighbors = &neighbors[n_index];
    _node.edges = &edges[n_index];
}

void GNNGraph::debug(std::ostream& stream) {
    GNNGraph& graph = *this;
    stream << "n_nodes: " << graph.n_nodes << std::endl;
    stream << "n_edges: " << graph.n_edges << std::endl;
    stream << std::endl << "Node indices and labels:" << std::endl;
    
    for(int i = 0; i < graph.n_nodes; ++i) {
        stream << i << ' ';
        for(int j = 0; j < node_label_size; ++j)
            stream << node_labels[i*node_label_size + j] << ' ';
        stream << std::endl;
    }
    
    stream << std::endl << "Edge indices, nodes and labels:" << std::endl;
    for(int i = 0; i < graph.n_nodes; ++i) {
        const int n_index = graph.n_neighbors[i];
        int num_neighbors = graph.n_neighbors[i+1] - graph.n_neighbors[i];
        for(int j = 0; j < num_neighbors; ++j) {
            if(i < graph.neighbors[n_index + j]) {
                stream << graph.edges[n_index + j] << ' ';
                stream << '(' << i << ',' << graph.neighbors[n_index + j] << ") ";
                for(int k = 0; k < edge_label_size; ++k) {
                    stream << edge_labels[edges[n_index + j]*edge_label_size + k] << ' ';
                }
                stream << std::endl;
            }
        }
    }
    stream << std::endl << "n_neighbors array:" << std::endl;
    for(int i = 0; i < n_nodes; ++i) {
        stream << n_neighbors[i] << ' ';
    }
    stream << std::endl << std::endl;
    for(int i = 0; i < n_neighbors[n_nodes]; ++i) {
        stream << neighbors[i] << ' ';
    }
    stream << std::endl;
    stream << "Edge indices array" << std::endl;
    for(int i = 0; i < n_neighbors[n_nodes]; ++i) {
        stream << edges[i] << ' ';
    }
    stream << std::endl;
    /*
    stream << "Edge label array" << std::endl;
    for(int i = 0; i < n_nodes; ++i) {
        const double* labels = edge_labels[edges[n_neighbors[i]
        stream << edge_labels[i] << ' ';
    }
    stream << std::endl;
    */
}

void GNNGraph::debug(std::ostream& _stream, int _index) {
    assert(_index < n_nodes);
    _stream << "Debug for node: " << _index << std::endl;
    Node node;
    getNode(_index, node);
    _stream << "Node num neighbors: " << node.num_neighbors << std::endl;
    _stream << "Node neighbors:" << std::endl;
    for(int i = 0; i < node.num_neighbors; ++i) {
        _stream << node.neighbors[i] << " ";
    }
    _stream << std::endl;
    _stream << "Node edges:" << std::endl;
    for(int i = 0; i < node.num_neighbors; ++i) {
        _stream << i << ' ' << edges[i] << ' ';
        _stream << '(' << _index << ',' << node.neighbors[i] << ") ";
        for(int k = 0; k < edge_label_size; ++k) {
            _stream << edge_labels[edges[i]*edge_label_size + k] << ' ';
        }
        _stream << std::endl;
    }
}


GNN::GNN(GraphHelper& _helper, int _node_state_size, Network* _fw, Network* _gw)
    : graph(_helper), fw(_fw), gw(_gw), node_state_size(_node_state_size)
{
    has_position = _helper.has_position;
    if(has_position) {
        box = _helper.box;
    }
    node_state_size = _node_state_size;
    max_neighbors = graph.maxNeighbors();
    state_size = node_state_size*graph.n_nodes;
    state = new double[state_size];
    new_state = new double[state_size];
    if(has_position) {
        //Positional NN
        nn_input_size = graph.node_label_size
            + max_neighbors*(node_state_size + graph.edge_label_size + graph.node_label_size);
    }
    else {
        nn_input_size = graph.node_label_size + node_state_size + graph.edge_label_size + graph.node_label_size;
    }
    nn_input = new double[nn_input_size];
    //printf("%d, %d\n", fw->nInputs, nn_input_size);
    assert(fw->nInputs == nn_input_size);
    assert(fw->nOutputs == node_state_size);
    assert(gw->nInputs == node_state_size);
    ActivationFunction::Enum outlayer = gw->layers[gw->nLayers-1].activation_enum;
    null_value = getActivationFunction(outlayer).null_value;
}

GNN::~GNN() {
    delete[] state;
    delete[] new_state;
    delete[] nn_input;
}

int GNN::getStateSize() {
    return state_size;
}

void GNN::setState(int _size, const double* _state) {
    assert(_size == state_size);
    memcpy(state, _state, _size*sizeof(double));
}

double GNN::step() {
    if(has_position)
        stepPositional();
    else
        stepNonPositional();
    //Calculate difference
    double max_diff = 0.0;
    for(int i = 0; i < state_size; ++i) {
        double temp = std::abs(state[i] - new_state[i]);
        max_diff = temp > max_diff ? temp : max_diff;
    }
    std::swap(state, new_state);
    return max_diff;
}

/*
Input layout:
[Node label | Neighbor 0 edge label | Neighbor 0 node label | Neighbor 0 node state | Neighbor 1 edge label | ...]
*/
void GNN::stepPositional() {
    GNNGraph::Node node;
    for(int i = 0; i < graph.n_nodes; ++i ) {
        for(int j = 0; j < nn_input_size; ++j)
            nn_input[j] = null_value;

        graph.getNode(i, node);
        double* input_ptr = nn_input;
        //Fill out the input
        const int index = node.node;
        memcpy(input_ptr, graph.node_labels + graph.node_label_size*index, graph.node_label_size*sizeof(double));
        double* node_pos = graph.node_labels + graph.node_label_size*index;
        input_ptr += graph.node_label_size;
        for(int j = 0; j < node.num_neighbors; ++j) {
            const int n_index = node.neighbors[j];
            const int e_index = node.edges[j];
            memcpy(input_ptr, graph.edge_labels + graph.edge_label_size*e_index, graph.edge_label_size*sizeof(double));
            input_ptr += graph.edge_label_size;

            //Copy node label and calculate relative position
            memcpy(input_ptr, graph.node_labels + graph.node_label_size*n_index, graph.node_label_size*sizeof(double));
            double* neighbor_pos = graph.node_labels + graph.node_label_size*n_index;
            box.relative_pos(input_ptr, node_pos, neighbor_pos);
            input_ptr += graph.node_label_size;

            memcpy(input_ptr, state + node_state_size*n_index, node_state_size*sizeof(double));
            input_ptr += node_state_size;
        }
        //assert(input_ptr == nn_input + nn_input_size); //only true if same amount of neighbors
        transition(nn_input, new_state + node_state_size*index);
    }
}

void GNN::stepNonPositional() {
    GNNGraph::Node node;
    for(int i = 0; i < graph.n_nodes; ++i ) {
        graph.getNode(i, node);
        arma::vec node_state(node_state_size, arma::fill::zeros);
        const int index = node.node;
        memcpy(nn_input, graph.node_labels + graph.node_label_size*index, graph.node_label_size*sizeof(double));
        for(int j = 0; j < node.num_neighbors; ++j) {
            double* input_ptr = nn_input + graph.node_label_size;
            const int n_index = node.neighbors[i];
            const int e_index = node.edges[i];
            memcpy(input_ptr, graph.edge_labels + graph.edge_label_size*e_index, graph.edge_label_size*sizeof(double));
            input_ptr += graph.edge_label_size;
            memcpy(input_ptr, graph.node_labels + graph.node_label_size*e_index, graph.node_label_size*sizeof(double));
            input_ptr += graph.node_label_size;
            memcpy(input_ptr, state + node_state_size*n_index, node_state_size*sizeof(double));
            input_ptr += node_state_size;
            assert(input_ptr == nn_input + nn_input_size);
            network_compute(fw, nn_input_size, nn_input);
            int output_size;
            const double* output = network_output(fw, &output_size);
            assert(output_size == node_state_size);
            arma::vec new_state(output, output_size);
            node_state += new_state;
        }
        memcpy(new_state + node_state_size*index, node_state.memptr(), node_state.n_rows*sizeof(double));
    }
}

void GNN::transition(const double* __restrict _input, double* __restrict _new_state) {
    network_compute(fw, nn_input_size, _input);
    int output_size;
    const double* output = network_output(fw, &output_size);
    assert(output_size == node_state_size);
    memcpy(_new_state, output, node_state_size*sizeof(double));
}

void GNN::printState(std::ostream& stream) {
    std::cout << "Current GNN state: " << std::endl;
    for(int i = 0; i < state_size; ++i) {
        std::cout << state[i] << ' ';
    }
    std::cout << std::endl;
}

void GNN::get_parameters(int size, double* _parameters) {
    assert(size == fw->parameter_size + gw->parameter_size);
    memcpy(_parameters, fw->parameters, fw->parameter_size*sizeof(double));
    _parameters += fw->parameter_size;
    memcpy( _parameters, gw->parameters, gw->parameter_size*sizeof(double));
}

void GNN::set_parameters(int size, const double* _parameters) {
    assert(size == fw->parameter_size + gw->parameter_size);
    memcpy(fw->parameters, _parameters, fw->parameter_size*sizeof(double));
    _parameters += fw->parameter_size;
    memcpy(gw->parameters, _parameters, gw->parameter_size*sizeof(double));
}

int GNN::parameter_size() {
    return fw->parameter_size + gw->parameter_size;
}

int GNN::output_size() {
    return graph.n_nodes*gw->nOutputs;
}

void GNN::compute(double _tol, int size, double* _output) {
    assert(size == output_size());
    //Iterate to self consistency
    static const int MAX_ITERATIONS = 100;
    for(int i = 0; i < MAX_ITERATIONS; ++i) {
        double err = step();
        if(err < _tol)
            break;
    }
    //Get the output from gw network to _output
    for(int i = 0; i < graph.n_nodes; ++i) {
        network_compute(gw, node_state_size, &state[i*node_state_size]);
        const double* out = network_output(gw);
        memcpy(_output, out, node_state_size*sizeof(double));
        _output += node_state_size;
    }
}

void GNN::get_output(arma::vec& _output) {
    double* output = _output.memptr();
    //Get the output from gw network to _output
    for(int i = 0; i < graph.n_nodes; ++i) {
        network_compute(gw, node_state_size, &state[i*node_state_size]);
        const double* out = network_output(gw);
        memcpy(output, out, node_state_size*sizeof(double));
        output += node_state_size;
    }
}

void GNN::load_parameters(const arma::vec& _parameters) {
    set_parameters(_parameters.n_rows, _parameters.memptr());
}

void GNN::compute(double _tol, arma::vec& output) {
    compute(_tol, output.n_rows, output.memptr());
}