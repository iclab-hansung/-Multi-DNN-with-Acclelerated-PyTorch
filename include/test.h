#ifndef TEST_H
#define TEST_H

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAMultiStreamGuard.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <pthread.h>
#include "thpool.h"
#include <utility>

/* FIFOQ 0 is PrioQ */
/* FIFOQ=0 -> Priority Q*/
/* FIFOQ=1 -> FIFOQ Q*/
#ifndef FIFOQ
#define FIFOQ 0 
#endif

// pytorch streampool 32개
#define n_streamPerPool 32
#define threshold 

extern threadpool thpool; 
extern pthread_cond_t *cond_t;
extern pthread_mutex_t *mutex_t;
extern int *cond_i;
extern int stream_priority;
extern std::vector<std::vector <at::cuda::CUDAStream>> streams;
extern cudaEvent_t event_A;

#endif