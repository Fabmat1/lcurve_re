cmake_minimum_required(VERSION 3.30)
project(lcurve_re)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp -funroll-loops -ftree-vectorize -fopenmp-simd -fopt-info-vec-optimized")

# Add the source files from the "utilities" directory
set(SOURCES
        src/model.cpp
        src/model.h
        src/new_subs.h
        src/ldc.h
        src/roche.cpp
        src/roche.h
        src/lcurve.cpp
        src/lcurve.h
        src/new_subs.cpp
        src/new_helpers.cpp
        src/new_helpers.h
        src/lcurve_base/array1d.h
        src/lcurve_base/light_curve_comp.cpp
        src/lcurve_base/lcurve.h
        src/lcurve_base/constants.h
        src/lcurve_base/buffer2d.h
)

add_library(utilities STATIC ${SOURCES})

add_executable(lcurve_re main.cpp)

target_link_libraries(lcurve_re PRIVATE utilities)

# Find and link nlohmann_json (header-only, no linking needed, just include path)
find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(lcurve_re PRIVATE nlohmann_json::nlohmann_json)
