cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(fifoq-test)

# set(CMAKE_PREFIX_PATH /home/kmsjames/very-big-storage/hojin/pytorch/newlibtorch/libtorch)
set(CMAKE_PREFIX_PATH /home/kmsjames/very-big-storage/hojin/pytorch/perfisol/libtorch)
find_package(Torch REQUIRED)

set(COMMON_DIR /home/kmsjames/very-big-storage/JSH/HGD_C++/git_pri/priority/pytorch/include)

file(GLOB SOURCES "src/*")
#set(SOURCES src/multithreadtest.cpp) #manual add

include_directories(${COMMON_DIR})

add_executable(fifoq-test ${SOURCES})
target_link_libraries(fifoq-test "${TORCH_LIBRARIES}")
set_property(TARGET fifoq-test PROPERTY CXX_STANDARD 14)
