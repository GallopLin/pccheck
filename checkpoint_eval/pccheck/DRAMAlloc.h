
#ifndef _DRAM_ALLOCATION_H_
#define _DRAM_ALLOCATION_H_

#include <atomic>
#include <iostream>
#include <cuda_runtime.h>
#include "FAAQueueAdd.h"

class DRAMAlloc
{
public:
    void alloc(size_t num, size_t size)
    {
        cudaError_t init_err = cudaSetDevice(0);
        if (init_err != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed: %s (error code %d)\n", 
                    cudaGetErrorString(init_err), init_err);
            fprintf(stderr, "Attempting to continue anyway...\n");
        }
        
        cudaFree(0);

        cudaError_t err = cudaMallocHost((void **)&array, num * size * 4);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMallocHost failed: %s (error code %d)\n", cudaGetErrorString(err), err);
            exit(EXIT_FAILURE);
        }
    }

    void initialize(size_t num, size_t size)
    {
        int tid = thread_tid();
        for (size_t i = 0; i < num; i++)
        {
            free_add.enqueue(array + (i * size), tid);
        }
    }

    float *get_add(size_t /* id — kept for API compat */)
    {
        int tid = thread_tid();
        while (true)
        {
            float *add = free_add.dequeue(tid);
            if (add != nullptr)
                return add;
        }
    }

    void put_add(float *add, size_t /* id */)
    {
        int tid = thread_tid();
        free_add.enqueue(add, tid);
    }

private:
    FAAArrayQueueAdd<float *> free_add;
    float *array;

    static int thread_tid()
    {
        static std::atomic<int> next_tid{0};
        thread_local int tid = next_tid.fetch_add(1);
        return tid;
    }
};

#endif
