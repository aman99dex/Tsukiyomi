cmake_minimum_required(VERSION 3.20)

project(cNeRF VERSION 1.0
              DESCRIPTION "Neural radiance fields with libtorch"
              LANGUAGES CXX)

# Configure CUDA
find_package(CUDA QUIET)
if(CUDA_FOUND)
    find_library(LIBNVTOOLSEXT
        NAMES nvToolsExt
        HINTS
            ENV CUDA_TOOLKIT_ROOT_DIR
            ENV CONDA_PREFIX
        PATH_SUFFIXES lib64 lib
    )
endif()

# Configure libtorch and OpenCV
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Configure ImGui and GLFW
include(FetchContent)
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui
  GIT_TAG        docking
)
FetchContent_MakeAvailable(imgui)

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

set(IMGUI_DIR ${imgui_SOURCE_DIR})
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(cNeRF src/main.cpp src/model.cpp src/renderer.cpp src/utils.cpp src/trainer.cpp ${IMGUI_SOURCES})
target_include_directories(cNeRF PUBLIC include ${IMGUI_DIR} ${IMGUI_DIR}/backends)

if(CUDA_FOUND)
    target_link_libraries(cNeRF ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${CUDA_LIBRARIES} glfw OpenGL::GL)
else()
    target_link_libraries(cNeRF ${TORCH_LIBRARIES} ${OpenCV_LIBS} glfw OpenGL::GL)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cNeRF OpenMP::OpenMP_CXX)
endif()
