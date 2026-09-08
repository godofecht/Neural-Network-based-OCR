cmake_minimum_required(VERSION 3.20)
project(NeuralNetworkOCR VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 QUIET)

if(UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
endif()

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.1
)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
endif()

if(NOT glfw3_FOUND)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    if(UNIX AND NOT APPLE)
        set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
        set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
    endif()

    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
    )
    FetchContent_GetProperties(glfw)
    if(NOT glfw_POPULATED)
        FetchContent_Populate(glfw)
        add_subdirectory(${glfw_SOURCE_DIR} ${glfw_BINARY_DIR})
    endif()
    set(GLFW_LIB glfw)
else()
    set(GLFW_LIB glfw)
endif()

# OCR only uses TinyML's core Network API. Pin the modern core-only build so
# unrelated extended backends do not become transitive build requirements.
set(TINYML_BUILD_EXTENDED OFF CACHE BOOL "" FORCE)
set(TINYML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TINYML_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(TINYML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TINYML_BUILD_PLAYGROUND OFF CACHE BOOL "" FORCE)
set(TINYML_FETCH_DEPENDENCIES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    tinyML
    GIT_REPOSITORY https://github.com/godofecht/tinyML.git
    GIT_TAG 44b188494b65f6f723d2f702b189c1b0f8cab287
)
FetchContent_MakeAvailable(tinyML)

add_executable(ocr_trainer
    NeuralNetworkComputation/Source.cpp
    NeuralNetworkComputation/GenAlg.cpp
)

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

target_include_directories(ocr_trainer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworkComputation
    ${tinyml_SOURCE_DIR}/include
)

target_include_directories(ocr_gui PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworkComputation
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${tinyml_SOURCE_DIR}/include
)

target_link_libraries(ocr_trainer PRIVATE TinyMLCore)
target_link_libraries(ocr_gui PRIVATE TinyMLCore ${GLFW_LIB} OpenGL::GL)

if(MSVC)
    target_compile_options(ocr_trainer PRIVATE /O2)
    target_compile_options(ocr_gui PRIVATE /O2)
else()
    target_compile_options(ocr_trainer PRIVATE -O3)
    target_compile_options(ocr_gui PRIVATE -O3)
endif()

if(APPLE)
    target_compile_definitions(ocr_trainer PRIVATE cimg_display=0)
elseif(UNIX)
    target_include_directories(ocr_trainer PRIVATE ${X11_INCLUDE_DIRS})
    target_link_libraries(ocr_trainer PRIVATE ${X11_LIBRARIES})
endif()

install(TARGETS ocr_trainer ocr_gui DESTINATION bin)
