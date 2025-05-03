cmake_minimum_required(VERSION 3.28.3)
project(lcurve_re)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp -funroll-loops -ftree-vectorize -fopenmp-simd -fopt-info-vec-optimized")


# --- Find gnuplot-iostream header (header-only) ---
find_path(GNUPLOT_IOSTREAM_INCLUDE_DIR
        NAMES gnuplot-iostream.h
        HINTS /usr/include /usr/local/include /usr/local/include/gnuplot-iostream
)

if(NOT GNUPLOT_IOSTREAM_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find gnuplot-iostream.h. Please install gnuplot-iostream or set GNUPLOT_IOSTREAM_INCLUDE_DIR manually.")
endif()

include_directories(${GNUPLOT_IOSTREAM_INCLUDE_DIR})

# --- Find Boost libraries for Filesystem and Iostreams ---
find_package(Boost REQUIRED COMPONENTS filesystem iostreams)
if(Boost_FOUND)
    message(STATUS "Boost found: ${Boost_VERSION}")
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Boost not found!")
endif()

# Add the source files from the "utilities" directory
set(SOURCES
        src/lcurve_base/model.cpp
        src/model.h
        src/new_subs.h
        src/lcurve_base/ldc.h
        src/lroche_base/roche.cpp
        src/lroche_base/roche.h
        src/lcurve_base/lcurve.cpp
        src/new_subs.cpp
        src/new_helpers.cpp
        src/new_helpers.h
        src/lcurve_base/array1d.h
        src/lcurve_base/light_curve_comp.cpp
        src/lcurve_base/lcurve.h
        src/lcurve_base/constants.h
        src/lcurve_base/buffer2d.h
        src/lcurve_base/comp_light.cpp
        src/lcurve_base/set_star_grid.cpp
        src/lcurve_base/set_star_continuum.cpp
        src/lcurve_base/set_disc_grid.cpp
        src/lcurve_base/disc_eclipse.cpp
        src/lcurve_base/set_bright_spot_grid.cpp
        src/lcurve_base/re_scale.cpp
        src/lcurve_base/comp_gravity.cpp
        src/lcurve_base/comp_radius.cpp
        src/lcurve_base/set_disc_continuum.cpp
        src/lcurve_base/star_eclipse.cpp
        src/lroche_base/sphere_eclipse.cpp
        src/lroche_base/face.cpp
        src/svdfit.cpp
        src/byte_swap.cpp
        src/lroche_base/ingress_egress.cpp
        src/lroche_base/ref_sphere.cpp
        src/lcurve_base/numface.cpp
        src/lroche_base/strining.cpp
        src/lroche_base/stradv.cpp
        src/lroche_base/drpot.cpp
        src/lroche_base/rpot.cpp
        src/lroche_base/gsint.cpp
        src/bsstep.cpp
        src/mmid.cpp
        src/pzextr.cpp
        src/stoerm.cpp
        src/lroche_base/set_earth.cpp
        src/lroche_base/xl1.cpp
        src/lroche_base/pot_min.cpp
        src/dbrent.cpp
        src/lroche_base/rpot_val.cpp
        src/lroche_base/rpot_grad.cpp
        src/lroche_base/rpot_val_grad.cpp
        src/lroche_base/rocacc.cpp
        src/lroche_base/fblink.cpp
)

add_library(utilities STATIC ${SOURCES})

add_executable(lcurve_re main.cpp)
add_executable(lcurve_mcmc ./src/new_scripts/mcmc_solver.cpp)

target_link_libraries(lcurve_re PRIVATE utilities)
target_link_libraries(lcurve_mcmc PRIVATE utilities)

# Find and link nlohmann_json (header-only, no linking needed, just include path)
find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(lcurve_re PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(lcurve_mcmc PRIVATE nlohmann_json::nlohmann_json)

target_link_libraries(lcurve_re PRIVATE Boost::filesystem Boost::iostreams)
target_link_libraries(lcurve_mcmc PRIVATE Boost::filesystem Boost::iostreams)
