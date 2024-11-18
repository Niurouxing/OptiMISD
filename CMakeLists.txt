cmake_minimum_required(VERSION 3.10)
project(OptiMISD VERSION 1.0)


add_subdirectory(extern/eigen)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# -DEIGEN_STACK_ALLOCATION_LIMIT=1000000 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_STACK_ALLOCATION_LIMIT=1048576")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


add_executable(demo main.cpp)

target_link_libraries(demo Eigen3::Eigen)