#include "network.h"
#include <stdio.h>
#include <stdlib.h>
#include <armadillo>
#include <assert.h>
#include <float.h>

using namespace arma;

#define SIZEOF(_arr) (sizeof(_arr)/sizeof(_arr[0]))
/*
These functions work only for NN with one-to-one mapping i.e. f(x) = y
*/

void evaluate_on_vector(Network* n, const vec& x, vec& result) {
    assert(result.n_rows == x.n_rows);
    for(size_t i = 0; i < x.n_rows; ++i) {
        network_compute(n, 1, &x[i]);
        result[i] = network_output(n)[0];
    }
}

void evaluate_on_population(Network* n, mat& population, vec& x, vec& y, vec& result_fitness) {
    assert(x.n_rows == y.n_rows);
    assert(result_fitness.n_rows == population.n_cols);
    vec result(y.n_rows);
    for(size_t i = 0; i < population.n_cols; ++i) {
        network_load_parameters(n, population.col(i));
        evaluate_on_vector(n, x, result);
        result_fitness[i] = sum(pow(result - y,2));
    }
}

void mutate_population(mat& population, mat& mutations) {
    for(size_t i = 0; i < mutations.n_rows; ++i) {
        for(size_t j = 0; j < population.n_cols; ++j) {
            vec chance = randu<vec>(population.n_rows);
            for(size_t k = 0; k < population.n_rows; ++k)
                chance[k] = mutations.at(i,0) > chance(k) ? 1.0 : 0.0; 
            vec mut = mutations.at(i, 1)*randn<vec>(population.n_rows);
            population.col(j) += chance%mut%population.col(j);
        }
    }
}

void tournament_selection_and_mating(vec& fitness, mat& population, mat& new_population, int num_winners) {
    assert(population.n_cols % num_winners == 0);
    assert(population.n_cols % 2 == 0);
    int stride = population.n_cols/num_winners;
    for(int i = 0; i < num_winners; ++i) {
        int min_index = fitness.rows(i*stride, (i+1)*stride-1).index_min();
        new_population.col(i) = population.col(min_index);
    }
    for(int i = 0; i < num_winners; ++i) {
        for(int j = 0; j < stride; ++j) {
            new_population.col(j*num_winners + i) = new_population.col(i);
        }
    }
    /*
    stride = stride - 1;
    for(size_t i = 0; i < (population.n_cols - num_winners)/2; ++i) {
        vec parent1 = randi<vec>( population.n_rows, distr_param(0,1) );
        vec parent2 = 1.0 - parent1;
        new_population.col(2*i + num_winners) = parent1%new_population.col(i/stride) + parent2%new_population.col(i/stride + 1);
        new_population.col(2*i+1 + num_winners) = parent2%new_population.col(i/stride) + parent1%new_population.col(i/stride + 1);
    }
    */
}


int main() {
    int num = 100;
    int population_size = 100;
    int num_elites = 20;
    int generations = 300;
    mat mutations(2,2);
    /*Mutations are described such that
    column 0 describes the chance of that mutation (uniform),
    column 1 describes the variance of gaussian mutation*/
    mutations(0,0) = 0.25;
    mutations(0,1) = 0.25;
    mutations(1,0) = 0.05;
    mutations(1,1) = 0.5;
    vec x = linspace<vec>(0,3,num);
    vec y = sin(x);
    ActivationFunction::Enum activators[] = {
        ActivationFunction::RELU, 
        ActivationFunction::LINEAR, 
        ActivationFunction::LINEAR};
    int layer_sizes[] =  {40,40,1};
    Network* n = network_create(1, 
        SIZEOF(layer_sizes), layer_sizes, 
        SIZEOF(activators), activators);
    mat population = randu<mat>(n->parameter_size, population_size);

    vec fitness(population.n_cols);
    vec best_error(generations+1);
    for(int i = 0; i < generations; ++i) {
        if(i % 50 == 0) printf("Generation %d\n", i);
        evaluate_on_population(n, population, x, y, fitness);
        best_error(i) = fitness.min();
        mat new_population(n->parameter_size, population_size);
        tournament_selection_and_mating(fitness, population, new_population, num_elites);
        //mutate_population(new_population, mutations);
        mutate_population(new_population, mutations);
        population = new_population;
    }
    evaluate_on_population(n, population, x, y, fitness);
    best_error(generations) = fitness.min();
    int min_index = fitness.index_min();
    vec parameters = population.col(min_index);
    network_load_parameters(n, parameters);
    vec result(num);
    evaluate_on_vector(n, x, result);
    mat out(num, 2);
    out.col(0) = x;
    out.col(1) = result;
    out.save("ga_sin.dat", raw_ascii);
    best_error.save("ga_error.dat", raw_ascii);
}