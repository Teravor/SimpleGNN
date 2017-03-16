# SimpleGNN
Graph Neural Network currently with GA training.

## NN lib
Basic feedforward neural network implementation is now at network.h and network.cpp (interface is a bit different than the earlier version). Examples of usage are contained in test_*.cpp files. Backprogation is currently implemented only for testing purposes. The library depends on Armadillo C++ Linear Algebra library. Theoretically one can use any activation function currently provided for any layer, but only sigmoid has been tested properly.

See following files for examples:
* test.cpp (just initializing and running dummy network)
* test_sin.cpp (testing network with beckpropagation)
* test_ga.cpp (GA example, GA does not work but might give an idea how to plug in to similar system)
 
## CMake 

Build system is controlled by CMake. There are lots of tutorials around, but not much is needed to understand this particular system. Some notes about the CMake if you are unfamiliar with it. Of course you can use your own build system. Only files needed for the network is network.h and network.cp

SET flags are just compiler directives. Currently they work only for g++. Cmake command tries to find CMakelists.txt in given location, which describes how the build system (in Linux environment Make) should be configured. After running CMake you can just build the targets (libraries/executables) specified in the CMakeLists.txt. 

* include directories adds external third party includes.
* add_library <target name> <source files> creates library file (default is static so linker can do all the optimization shenanigans) from given source files.
* target_link_libraries <target> <libs> links given libraries to the target.
* add_executable <target name> creates an executable from given sources.

* find_package tries to find Armadillo C++ library. If you have installed it through package manager or in standard locations then it should just work. If not, you need give CMake the path where it can find it. To do this just prepend your cmake command with prefix path "CMAKE_PREFIX_PATH=/path/to/armadillo/installation cmake ... (CMake arguments)".

* CMake makes it easy to do out-of-source builds so that your precious source directory is not cluttered with mess. I usually do in source directory "mkdir _build", "cd _build" and "cmake .." to create the build system (also makes it nice to work with git, just never include any files in _build directory).

* With CMake and make (usually Linux environment) you must specify what is your desired build. If one wants debug version then use argument "cmake <location of CMakeLists.txt> -DCMAKE_BUILD_TYPE=Debug" while release version is "cmake <location of CMakeLists.txt> -DCMAKE_BUILD_TYPE=Release". Note that release version usually compiles out any assertions in the code.

* One of the targets you can try to build is test_network with "make test_network"

* If you need to know what arguments Make is giving to the compiler (some library missing and can't understand why), you can use "make <target> VERBOSE=1" and it will output the compiler and linker arguments.
