cmake_minimum_required(VERSION 3.15)
project(NeuralNetworkOCR VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Platform-specific configuration
if(APPLE)
    # macOS support
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")
    find_package(X11 QUIET)
    if(NOT X11_FOUND)
        message(STATUS "X11 not found on macOS, building without X11 support")
    endif()
else()
    # Linux requires X11
    find_package(X11 REQUIRED)
endif()

# Fetch tinyML v1.0.0
include(FetchContent)
FetchContent_Declare(
    tinyML
    GIT_REPOSITORY https://github.com/godofecht/tinyML.git
    GIT_TAG v1.0.0
)

FetchContent_GetProperties(tinyML)
if(NOT tinyml_POPULATED)
    FetchContent_Populate(tinyML)
    add_subdirectory(${tinyml_SOURCE_DIR} ${tinyml_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Main executable
add_executable(ocr_trainer
    NeuralNetworkComputation/Source.cpp
    NeuralNetworkComputation/GenAlg.cpp
)

# Include directories
target_include_directories(ocr_trainer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworkComputation
)

# Platform-specific compilation flags
if(APPLE)
    # macOS: Disable CImg display since X11 is not required
    target_compile_definitions(ocr_trainer PRIVATE cimg_display=0)
else()
    # Linux: Enable X11 display if available
    if(X11_FOUND)
        target_include_directories(ocr_trainer PRIVATE ${X11_INCLUDE_DIRS})
    endif()
endif()

# Add tinyML includes and linking
target_include_directories(ocr_trainer PRIVATE ${tinyml_SOURCE_DIR}/include)
target_link_libraries(ocr_trainer PRIVATE TinyML)

# Link X11 if available (Linux)
if(X11_FOUND)
    target_link_libraries(ocr_trainer PRIVATE ${X11_LIBRARIES})
endif()

# Installation
install(TARGETS ocr_trainer DESTINATION bin)
