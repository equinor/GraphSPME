cmake_minimum_required(VERSION 3.14)
project(graphSPME LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH "${graphSPME_SOURCE_DIR}/cmake/modules")

find_package(LibR REQUIRED)
# Find Rcpp.h and RcppEigen.h
# TODO: Consolidate with LibR.
find_package(R REQUIRED)

file(GLOB_RECURSE R_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/R-package/src/*.cpp)

add_library(graphSPME-r SHARED ${R_SOURCES})
target_compile_definitions(graphSPME-r PUBLIC)
target_include_directories(graphSPME-r PRIVATE
    ${LIBR_INCLUDE_DIRS}
    ${R_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include/graph_spme)
target_link_libraries(graphSPME-r PUBLIC ${LIBR_CORE_LIBRARY})

set_target_properties(
  graphSPME-r PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
  SUFFIX ".so"
  PREFIX ""
  OUTPUT_NAME "GraphSPME")

include(cmake/RPackageInstallTargetSetup.cmake)
setup_rpackage_install_target(graphSPME-r "${CMAKE_CURRENT_BINARY_DIR}/R-package-install")
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/dummy_inst")
