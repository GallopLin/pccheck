
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

        // Keep one 4 KiB page of slack per logical chunk.  The Python D2H
        // path selects a pointer inside this slack whose page residue matches
        // the destination file offset.  Consequently the aligned middle of a
        // chunk can be submitted to O_DIRECT without an intermediate bounce
        // buffer, even when a stream itself does not start on a page boundary.
        logical_size = size;
        allocation_stride = size + direct_io_slack_floats;
        cudaError_t err = cudaMallocHost(
            (void **)&array, num * allocation_stride * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMallocHost failed: %s (error code %d)\n", cudaGetErrorString(err), err);
            exit(EXIT_FAILURE);
        }
    }

    void initialize(size_t num, size_t size)
    {
        if (size != logical_size) {
            fprintf(stderr, "DRAMAlloc initialize size mismatch: %zu != %zu\n",
                    size, logical_size);
            exit(EXIT_FAILURE);
        }
        int tid = thread_tid();
        for (size_t i = 0; i < num; i++)
        {
            free_add.enqueue(array + (i * allocation_stride), tid);
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
    static constexpr size_t direct_io_slack_floats = 4096 / sizeof(float);
    FAAArrayQueueAdd<float *> free_add;
    float *array = nullptr;
    size_t logical_size = 0;
    size_t allocation_stride = 0;

    static int thread_tid()
    {
        static std::atomic<int> next_tid{0};
        thread_local int tid = next_tid.fetch_add(1);
        return tid;
    }
};

#endif
