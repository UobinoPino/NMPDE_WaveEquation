cmake_minimum_required(VERSION 3.12.0)
project(Wave LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_executable(Wave_Newmark
        Newmark/Wave_Newmark.cpp
        Newmark/Wave.cpp)
deal_ii_setup_target(Wave_Newmark)

add_executable(Wave_Serial
        Theta_Method/serial/main.cpp
        Theta_Method/serial/Wave.cpp)
deal_ii_setup_target(Wave_Serial)

if(DEAL_II_WITH_MPI AND DEAL_II_WITH_TRILINOS)
    add_executable(Wave_Parallel
            Theta_Method/parallel/main.cpp
            Theta_Method/parallel/WaveParallel.cpp)
    deal_ii_setup_target(Wave_Parallel)
endif()
