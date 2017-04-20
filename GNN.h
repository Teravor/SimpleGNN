#include <armadillo>
#include <unordered_map>
#include <vector>
#include <array>
#include <stdint.h>
#include "network.h"

/*
OVERVIEW:
The user creates the graph with GraphHelper by adding nodes and then edges between them
while supplying the necessary labels for them. Then the GraphHelper is used to create the GNN.
GNN takes the Networks fw and gw as input. They are the transition function and output function.
The GNN should be supplied with some reasonable initial state and then it can be iterated to convergence
with step() function which return the highest difference in a node in the state change.


DETAILS:
GraphHelper is just a a quick and dirty helper class and the data is laid out in more reasonable fashion
when GNN creates GNNGraph. GNNGraph contains all the graph data while GNN contains the state data and 
functions fw and gw. The basic flow in GNN for one step goes like:
Loop over nodes:
    Find out nodes neighbors
    Prepare input vector for fw (append neighbor states and labels into one vector)
    Calculate fw
Return biggest difference in state change.

Some points in the functionality
*If the graph is positional, then the first three labels in the node label are assumed to by the (x,y,z)
coordinates. This means that the neighboring nodes are always sorted in order from nearest to farthest.
Remember to set the has_position flag when using GraphHelper!
*The positions of neighbors provided by GNN will always be *relational* with respect to node i.e. 
node will have position (0,0,0) and neighbors will have positions. Remeber to input absolute positions, though.
*If it is not positional then no such ordering is done and the nodes enter the fw network in random order
(not exactly but this is a safe assumption)
*Remember that the size of fw network input must be adjusted to the size of maximum amount of neighbors!
The formula for input size is 
node_label_size + max_neighbors*(node_label_size + edge_label_size + node_state_size).
*If some nodes number of neighbors != max_neighbors in graph then the missing
neighbor vector values are padded with special null value which depends on the activation
function of the network fw (null value is such that it does not trigger activation).

*/
struct GraphHelper {
    int nodeLabelSize;
    int edgeLabelSize;
    int n_nodes;
    int n_edges;
    std::unordered_map<int32_t, int> edge_indices;
    std::vector<double> node_labels;
    std::vector<double> edge_labels;
    bool has_position;


    GraphHelper(int nodeLabelSize, int _edgeLabelSize, bool _has_position = false);
    //Input functions
    int addNode(int _size, const double* _nodeLabel);
    void addEdge(int node1, int node2, int _size, const double* _edgeLabel);

    //Mostly for internal use with GNNGraph
    int numNeighbors(int node);
    bool hasEdge(int node1, int node2);
    int getEdgeIndex(int node1, int node2);

    //Help with testing and checking
    bool isCompleteGraph();
    int maxNeighbors();
    int getNetworkInputSize(int node_state_size);


    //Helper functions for testing
    template<size_t N>
    int addNode(const double (&x)[N]) {
        return addNode(N, x);
    }

    template<size_t N>
    void addEdge(int node1, int node2, const double (&x)[N]) {
        addEdge(node1, node2, N, x);
    }

};

struct GNNGraph {
    struct Node {
        int node;
        int num_neighbors;
        int* neighbors;
        int* edges;
        Node(int _max_neighbors);
        ~Node();
    };
    int n_nodes;
    int n_edges;

    int neighbour_size;

    //n_neighbors stores index to neighbors and size as a difference of consecutive elements
    int* n_neighbors;
    int* neighbors;
    int* edges;

    int node_label_size;
    double* node_labels;
    int edge_label_size;
    double* edge_labels;

    GNNGraph(GraphHelper& _helper);
    ~GNNGraph();
    //Maximum amount neighbors any node can have in the graph
    int maxNeighbors();
    //Get nodes neighbor and edge indices
    void getNode(int _node_index, Node& _node);

    //Sort neighbors according to their position, for internal use
    void sortPositions();

    void debug(std::ostream& stream);
private:
    void create(GraphHelper& _helper);
    void destroy();
};

struct GNN {
    GNNGraph graph;
    Network* fw;
    Network* gw;
    //TransitionFunction* transition;
    int node_state_size;
    int state_size;
    double* state;
    double* new_state;
    int nn_input_size;
    double* nn_input;
    int max_neighbors;
    double null_value;
    bool has_position;

    const double* node_labels;
    const double* edge_labels;


    GNN(GraphHelper& _helper, int _node_state_size, Network* _fw, Network* _gw);
    ~GNN();
    int getStateSize();
    //Provide initial state
    void setState(int _size, const double* _state);
    //Do one iteration
    double step();
    //Mostly for internal use, use if you know what you are doing
    void stepPositional();
    void stepNonPositional();
    //For debug
    void printState(std::ostream& stream);

private:
    GNN(const GNN&); // no implementation
    GNN& operator=(const GNN&); // no implementation 

    //Internal function
    void transition(const double* __restrict _input, double* __restrict _new_state);
};