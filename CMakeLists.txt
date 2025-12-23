cmake_minimum_required(VERSION 3.12.0)
project(Wave LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_executable(Wave_Newmark
        src/Newmark/Wave.cpp
        src/Newmark/main.cpp)
deal_ii_setup_target(Wave_Newmark)

add_executable(Wave_Serial
        src/Theta_Method/serial/main.cpp
        src/Theta_Method/serial/Wave.cpp)
deal_ii_setup_target(Wave_Serial)

if(DEAL_II_WITH_MPI AND DEAL_II_WITH_TRILINOS)
    add_executable(Wave_Parallel
            src/Theta_Method/parallel/main.cpp
            src/Theta_Method/parallel/WaveParallel.cpp)
    deal_ii_setup_target(Wave_Parallel)
endif()
