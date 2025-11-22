
#ifndef _DRAM_ALLOCATION_H_
#define _DRAM_ALLOCATION_H_

#include <atomic>
#include <iostream>
#include <cuda_runtime.h>
#include "FAAQueueAdd.h"

class DRAMAlloc
{
public:
    // size should be correlated to the size of the batch
    void alloc(size_t num, size_t size)
    {
        // each element needs to hold the parallel_item and the batch_num?
        //  TODO: is num the number of batches, and size the batch size (in floats)?
        // size_t total_element = size + 2*sizeof (int);

        // 🔧 FIX: 在子进程中重新初始化 CUDA 上下文
        // 这对于 multiprocessing fork 模式是必需的
        cudaError_t init_err = cudaSetDevice(0);
        if (init_err != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed: %s (error code %d)\n", 
                    cudaGetErrorString(init_err), init_err);
            fprintf(stderr, "Attempting to continue anyway...\n");
        }
        
        // 确保 CUDA 已完全初始化
        cudaFree(0);

        cudaError_t err = cudaMallocHost((void **)&array, num * size * 4);
        // assert(err == cudaSuccess);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMallocHost failed: %s (error code %d)\n", cudaGetErrorString(err), err);
            exit(EXIT_FAILURE);
        }
    }

    // insert all free elements to the queue
    void initialize(size_t num, size_t size)
    {
        // size_t element_size = size + 2*sizeof (int);
        for (int i = 0; i < num; i++)
        {
            free_add.enqueue(array + (i * size), 0);
        }
    }

    // get free address
    float *get_add(size_t id)
    {
        while (true)
        {
            float *add = free_add.dequeue(id);
            if (add != nullptr)
                return add;
        }
    }

    // put free address
    void put_add(float *add, size_t id)
    {
        free_add.enqueue(add, 0);
    }

private:
    FAAArrayQueueAdd<float *> free_add;
    float *array;
};

#endif
