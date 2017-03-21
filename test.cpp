#include "network.h"
#include <stdio.h>
#include <stdlib.h>
#include <armadillo>

#define SIZEOF(_arr) (sizeof(_arr)/sizeof(_arr[0]))

int main() {
    ActivationFunction::Enum activators[] = {
        ActivationFunction::SIGMOID,
        ActivationFunction::SIGMOID };
    int sI = 1;
    int layer_sizes[] =  {20,1};
    Network* n = network_create(sI, 
        SIZEOF(layer_sizes), layer_sizes, 
        SIZEOF(activators), activators);
    arma::vec input(sI, arma::fill::ones);
    arma::vec parameters(n->parameter_size, arma::fill::ones);
    parameters = 0.01*parameters;
    network_load_parameters(n, parameters);
    network_compute(n, input);
    arma::colvec output;
    network_output(n, output); 
    printf("Arma writing to csv\n");
    printf("Output: %f\n", output[0]);
    output.save("output.csv",arma::raw_ascii);
    network_destroy(n);
}