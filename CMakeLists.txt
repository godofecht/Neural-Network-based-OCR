cmake_minimum_required(VERSION 3.15)
project(NeuralNetworkOCR VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Include FetchContent module
include(FetchContent)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 QUIET)

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

# Fetch ImGui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.1
)

FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
endif()

# Fetch GLFW if not found
if(NOT glfw3_FOUND)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
    )
    FetchContent_GetProperties(glfw)
    if(NOT glfw_POPULATED)
        FetchContent_Populate(glfw)
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        add_subdirectory(${glfw_SOURCE_DIR} ${glfw_BINARY_DIR})
    endif()
    set(GLFW_LIB glfw)
else()
    set(GLFW_LIB glfw)
endif()

# Fetch tinyML v1.0.0
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

# Main CLI executable (keep for backward compatibility)
add_executable(ocr_trainer
    NeuralNetworkComputation/Source.cpp
    NeuralNetworkComputation/GenAlg.cpp
)

# ImGui GUI executable
set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

add_executable(ocr_gui
    NeuralNetworkComputation/TrainerGUI.cpp
    NeuralNetworkComputation/GenAlg.cpp
    ${IMGUI_SOURCES}
)

# Include directories for CLI
target_include_directories(ocr_trainer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworkComputation
)

# Include directories for GUI
target_include_directories(ocr_gui PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworkComputation
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
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

# Add tinyML includes and linking for CLI
target_include_directories(ocr_trainer PRIVATE ${tinyml_SOURCE_DIR}/include)
target_link_libraries(ocr_trainer PRIVATE TinyML)

# Add tinyML includes and linking for GUI
target_include_directories(ocr_gui PRIVATE ${tinyml_SOURCE_DIR}/include)
target_link_libraries(ocr_gui PRIVATE TinyML ${GLFW_LIB})

# Link OpenGL on macOS
if(APPLE)
    target_link_libraries(ocr_gui PRIVATE "-framework OpenGL")
else()
    target_link_libraries(ocr_gui PRIVATE GL)
endif()

# Link X11 if available (Linux)
if(X11_FOUND)
    target_link_libraries(ocr_trainer PRIVATE ${X11_LIBRARIES})
endif()

# Installation
install(TARGETS ocr_trainer ocr_gui DESTINATION bin)
