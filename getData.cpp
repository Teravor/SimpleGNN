#include <iostream>
#include <string>
#include <armadillo>
#include <time.h>
extern "C" {
#include "network.h"
}


using namespace std;
using namespace arma;

int main(){

	vec v=linspace(-3,3,100);

	vec w = sin(v);

       mat	X = join_horiz(v,w);

	X.save("X.dat",raw_ascii);

return 0;

}







