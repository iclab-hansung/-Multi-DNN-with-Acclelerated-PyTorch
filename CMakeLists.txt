cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(fifoq_switch)


set(CMAKE_PREFIX_PATH /home/kmsjames/very-big-storage/hojin/pytorch/libtorch)
find_package(Torch REQUIRED)

set(COMMON_DIR /home/kmsjames/very-big-storage/inmo/FIFOQ_switch/include)

file(GLOB SOURCES "src/*")
# set(SOURCES src/joo_model_prac.cpp) #manual add

include_directories(${COMMON_DIR})

add_executable(fifoq_switch ${SOURCES})
target_link_libraries(fifoq_switch "${TORCH_LIBRARIES}")
set_property(TARGET fifoq_switch PROPERTY CXX_STANDARD 14)