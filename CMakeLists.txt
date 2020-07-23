cmake_minimum_required(VERSION 3.14)
project(NumericalOptimization)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS -pthread)
set(GLOB FUNCTIONS functions/*.h function.h)
set(GLOB LINE_SEARCHES line_searches/*.h line_searches.h)
set(GLOB METHODS methods/*/*.h methods.h)


add_executable(NumericalOptimization main.cpp)
add_library(numopt_clib SHARED library.h library.cpp ${FUNCTIONS} ${LINE_SEARCHES} ${METHODS})
find_package(Python2 COMPONENTS Development NumPy)
target_include_directories(NumericalOptimization PRIVATE ${Python2_INCLUDE_DIRS} ${Python2_NumPy_INCLUDE_DIRS})
target_link_libraries(NumericalOptimization Python2::Python Python2::NumPy)