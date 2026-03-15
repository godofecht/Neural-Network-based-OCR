cmake_minimum_required(VERSION 3.15)
project(NeuralNetworkOCR VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find X11
find_package(X11 REQUIRED)

# Try to find tinyML with CMake config first
find_package(tinyML CONFIG QUIET)

# If not found, manually set include path
if(NOT tinyML_FOUND)
    message(STATUS "tinyML CMake config not found, using manual include path")
    if(DEFINED CMAKE_PREFIX_PATH)
        message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
        find_path(TINYML_INCLUDE_DIR
            NAMES Network.h NN.h
            PATHS ${CMAKE_PREFIX_PATH}/include
            NO_DEFAULT_PATH
        )
        if(TINYML_INCLUDE_DIR)
            message(STATUS "Found tinyML headers at: ${TINYML_INCLUDE_DIR}")
        else()
            message(WARNING "tinyML headers not found, trying system paths")
            find_path(TINYML_INCLUDE_DIR NAMES Network.h NN.h)
        endif()
    endif()
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

# Add tinyML includes
if(tinyML_FOUND)
    target_include_directories(ocr_trainer PRIVATE ${tinyML_INCLUDE_DIRS})
    target_link_libraries(ocr_trainer PRIVATE ${tinyML_LIBRARIES})
elseif(TINYML_INCLUDE_DIR)
    target_include_directories(ocr_trainer PRIVATE ${TINYML_INCLUDE_DIR})
else()
    # Fallback to system search
    target_include_directories(ocr_trainer PRIVATE /usr/local/include /usr/include)
endif()

# Link X11
target_link_libraries(ocr_trainer PRIVATE ${X11_LIBRARIES})

# Installation
install(TARGETS ocr_trainer DESTINATION bin)
