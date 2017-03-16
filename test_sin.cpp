#include "network.h"
#include <stdio.h>
#include <stdlib.h>
#include <armadillo>
#include <assert.h>

#define SIZEOF(_arr) (sizeof(_arr)/sizeof(_arr[0]))

using namespace arma;
//Only for 1 input, one output NN
void gather_result(Network* n, vec& x, vec& result) {
    assert(n->nOutputs == 1);
    for(int i = 0; i < (int)x.n_rows; ++i) {
        network_compute(n, 1, &x[i]);
        result[i] = n->outputs[0];
    }
}

int main(int argc, char* argv[]) {
    int epochs = 4000;
    double rate = 0.1;
    printf("Doing %d epochs with rate %f\n", epochs, rate);
    int num = 100;
    arma::vec x = arma::linspace<arma::vec>(0,3,num);
    double normalization = 0.25;
    double shift = 1.0;
    arma::vec y = normalization*(arma::sin(x) + shift); 

    ActivationFunction::Enum activators[] = {
        ActivationFunction::SIGMOID, 
        ActivationFunction::SIGMOID, 
        ActivationFunction::SIGMOID};
    int layer_sizes[] =  {40,40,1};

    Network* n = network_create(1, 
        SIZEOF(layer_sizes), layer_sizes, 
        SIZEOF(activators), activators);
    arma::vec parameters(n->parameter_size, arma::fill::randn);

    network_load_parameters(n, parameters);

    
    arma::vec epoch_errors(epochs);
    for(int i = 0; i < epochs; ++i) {
        for(int j = 0; j < (int)x.n_rows; ++j) {
            network_backpropagate(n, &x[j], &y[j], rate);
        }
        vec result(num);
        gather_result(n, x, result);
        result = pow(result - y, 2);
        epoch_errors[i] = sum(result);
    }
    
    vec result(num);
    gather_result(n, x, result);
    mat out(x.n_rows, 3);
    out.col(0) = x;
    out.col(1) = result/normalization - shift;
    out.col(2) = y;
    out.save("sin.dat", arma::raw_ascii);
    epoch_errors.save("epoch_errors.dat", arma::raw_ascii);
    network_destroy(n);
    
}