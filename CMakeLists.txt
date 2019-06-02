cmake_minimum_required(VERSION 3.14)
project(NumericalOptimization)

set(CMAKE_CXX_STANDARD 17)

set(GLOB FUNCTIONS functions/*.h function.h)
set(GLOB LINE_SEARCHES line_searches/*.h line_searches.h)
set(GLOB METHODS methods/*.h methods.h)

add_executable(NumericalOptimization main.cpp ${FUNCTIONS} ${LINE_SEARCHES} ${METHODS} functions/diagonal1.h functions/generalized_psc1.h functions/extended_qp1.h functions/fletchcr.h functions/ap_quad.h)
add_library(numopt_clib SHARED library.h library.cpp ${FUNCTIONS} ${LINE_SEARCHES} ${METHODS})