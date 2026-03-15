cmake_minimum_required(VERSION 3.15)
project(NeuralNetworkOCR VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Find tinyML library
find_package(tinyML REQUIRED)

# Main executable
add_executable(ocr_trainer
    NeuralNetworkComputation/Source.cpp
    NeuralNetworkComputation/GenAlg.cpp
)

# Include directories
target_include_directories(ocr_trainer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworkComputation
    ${TINYML_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(ocr_trainer PRIVATE
    tinyML::tinyML
    X11
)

# Platform-specific settings
if(APPLE)
    find_package(X11 REQUIRED)
    target_link_libraries(ocr_trainer PRIVATE ${X11_LIBRARIES})
elseif(UNIX)
    find_package(X11 REQUIRED)
    target_link_libraries(ocr_trainer PRIVATE ${X11_LIBRARIES})
endif()

# Installation
install(TARGETS ocr_trainer DESTINATION bin)
