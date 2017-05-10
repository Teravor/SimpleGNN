#If this fails use the one below
cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
#cmake_minimum_required(VERSION 2.8.0 FATAL_ERROR)

set(CMAKE_CXX_COMPILER "mpic++" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
#set (CMAKE_CXX_STANDARD 11)

#set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

#NN base library
add_library(network network.cpp)
target_link_libraries(network ${ARMADILLO_LIBRARIES})

#GNN library
add_library(GNN GNN.cpp GNN_interface.cpp)
target_link_libraries(GNN network ${ARMADILLO_LIBRARIES})

#Test cases
add_executable(test_ga myGaMDMPI.cpp )
target_link_libraries(test_ga network )
#add_executable(testGNN testGNN.cpp)
#target_link_libraries(testGNN network)
add_executable(test_sin test_sin.cpp)
target_link_libraries(test_sin network)
#add_executable(ga test_ga.cpp)
#target_link_libraries(ga network)

#GNN test case
add_executable(test_GNN GNN_test.cpp)
target_link_libraries(test_GNN GNN network ${ARMADILLO_LIBRARIES})
add_executable(test_interface GNN_interface_test.cpp)
target_link_libraries(test_interface GNN network ${ARMADILLO_LIBRARIES})

