cmake_minimum_required(VERSION 3.22.3)
project(lcurve_re)

set(CMAKE_CXX_STANDARD 20)

cmake_policy(SET CMP0146 NEW)  # Use modern CUDA support
cmake_policy(SET CMP0167 NEW)  # Use modern Boost support

# Check for CUDA using the modern approach
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DHAVE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_70")
    set(CUDA_FOUND TRUE)
    
    # Find CUDA toolkit for libraries
    find_package(CUDAToolkit REQUIRED)
endif()

# --- Find gnuplot-iostream header (header-only) ---
find_path(GNUPLOT_IOSTREAM_INCLUDE_DIR
        NAMES gnuplot-iostream.h
        HINTS /usr/include /usr/local/include /usr/local/include/gnuplot-iostream
)

if(NOT GNUPLOT_IOSTREAM_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find gnuplot-iostream.h. Please install gnuplot-iostream or set GNUPLOT_IOSTREAM_INCLUDE_DIR manually.")
endif()

include_directories(${GNUPLOT_IOSTREAM_INCLUDE_DIR})

# --- Find Boost libraries using modern approach ---
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
        src/amoeba.cpp
        src/mass_ratio_pdf.cpp
        src/mass_ratio_pdf.h
)

add_library(utilities STATIC ${SOURCES})

find_package(OpenMP REQUIRED)          # looks for the compiler's OpenMP support
target_link_libraries(utilities PRIVATE OpenMP::OpenMP_CXX)

# Link CUDA runtime if found
if(CMAKE_CUDA_COMPILER)
    target_link_libraries(utilities PRIVATE CUDA::cudart)
    set_property(TARGET utilities PROPERTY CUDA_ARCHITECTURES 70)
endif()

add_executable(lcurve_re main.cpp)
add_executable(lcurve_mcmc new_scripts/mcmc_solver.cpp)
add_executable(lcurve_simplex new_scripts/simplex.cpp)
add_executable(lcurve_visualize new_scripts/visualize.cpp)
add_executable(test new_scripts/test.cpp)

target_link_libraries(lcurve_re PRIVATE utilities)
target_link_libraries(lcurve_mcmc PRIVATE utilities)
target_link_libraries(lcurve_simplex PRIVATE utilities)
target_link_libraries(test PRIVATE utilities)
target_link_libraries(lcurve_visualize  PRIVATE utilities)

# Find and link nlohmann_json (header-only, no linking needed, just include path)
find_package(nlohmann_json CONFIG REQUIRED)
message(STATUS "nlohmann_json version: ${nlohmann_json_VERSION}")
target_link_libraries(lcurve_re PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(lcurve_mcmc PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(lcurve_simplex PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test PRIVATE nlohmann_json::nlohmann_json)

target_link_libraries(lcurve_re PRIVATE Boost::filesystem Boost::iostreams)
target_link_libraries(lcurve_mcmc PRIVATE Boost::filesystem Boost::iostreams)
target_link_libraries(lcurve_simplex PRIVATE Boost::filesystem Boost::iostreams)
target_link_libraries(test PRIVATE Boost::filesystem Boost::iostreams)
target_link_libraries(lcurve_visualize PRIVATE Boost::filesystem Boost::iostreams)

# Installation configuration
include(GNUInstallDirs)

# Install the main executables to bin directory
install(TARGETS lcurve_re lcurve_mcmc lcurve_simplex
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Runtime)
install(TARGETS lcurve_visualize
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Runtime)

# Set default install prefix if not specified
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Default install prefix" FORCE)
endif()

# ─── OPTIONS ───────────────────────────────────────────────────────
option(LCURVE_STRICT          "Fail the build on any warning"  OFF)
option(LCURVE_ENABLE_ASAN     "Build with address/UB sanitiser" OFF)

# ─── BUILD TYPE DEFAULT ───────────────────────────────────────────
# Only set a default when the user has not supplied -DCMAKE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Debug/Release/RelWithDebInfo/MinSizeRel" FORCE)
endif()

# ─── COMPILER-SPECIFIC WARNING FLAGS  ─────────────────────────────
set(_COMMON_WARNINGS "")
set(_WERROR "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    list(APPEND _COMMON_WARNINGS -Wall -Wextra -Wpedantic)
elseif(MSVC)
    list(APPEND _COMMON_WARNINGS /W4 /permissive-)
endif()

if(LCURVE_STRICT)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        list(APPEND _WERROR -Werror)
    elseif(MSVC)
        list(APPEND _WERROR /WX)
    endif()
endif()

# ─── SANITISER FLAGS (only meaningful for Clang/GCC & no CUDA) ───
set(_SAN_FLAGS "")
if(LCURVE_ENABLE_ASAN AND NOT CMAKE_CUDA_COMPILER
   AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    list(APPEND _SAN_FLAGS -fsanitize=address,undefined
                           -fno-omit-frame-pointer)
endif()


# ─── HELPER FUNCTION TO APPLY EVERYTHING TO A TARGET ──────────────
function(lcurve_setup_target TGT)
    # Core warnings
    target_compile_options(${TGT} PRIVATE ${_COMMON_WARNINGS} ${_WERROR})

    # Sanitiser flags (compile + link)
    if(_SAN_FLAGS)
        target_compile_options(${TGT} PRIVATE
            $<$<CONFIG:Debug>:${_SAN_FLAGS}>)
        target_link_options(${TGT}    PRIVATE
            $<$<CONFIG:Debug>:${_SAN_FLAGS}>)
    endif()

    # Release optimisation flags
    if(_RELEASE_FLAGS)
        target_compile_options(${TGT} PRIVATE
            $<$<CONFIG:Release>:${_RELEASE_FLAGS}>)
        target_link_options(${TGT} PRIVATE
            $<$<CONFIG:Release>:${_RELEASE_FLAGS}>)
    endif()
endfunction()

# ─── APPLY THE RULES TO ALL PROJECT TARGETS ───────────────────────
# (add every target you create here)
set(_LCURVE_ALL_TARGETS
    utilities
    lcurve_re
    lcurve_mcmc
    lcurve_simplex
    lcurve_visualize
    test)

foreach(t ${_LCURVE_ALL_TARGETS})
    if(TARGET ${t})
        lcurve_setup_target(${t})
    endif()
endforeach()