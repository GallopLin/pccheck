
#ifndef _DRAM_ALLOCATION_H_
#define _DRAM_ALLOCATION_H_

#include <atomic>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "FAAQueueAdd.h"

class DRAMAlloc
{
public:
    void alloc(size_t num, size_t size)
    {
        // 尝试使用 CUDA pinned memory（性能最优）
        // 如果 CUDA 不可用（例如在 fork 后的子进程中），回退到普通对齐内存
        bool cuda_ok = false;
        cudaError_t init_err = cudaSetDevice(0);
        if (init_err == cudaSuccess) {
            cudaFree(0);
            cudaError_t err = cudaMallocHost((void **)&array, num * size * 4);
            if (err == cudaSuccess) {
                cuda_ok = true;
                using_pinned = true;
            }
        }
        
        if (!cuda_ok) {
            fprintf(stderr, "[DRAMAlloc] CUDA pinned memory unavailable, falling back to posix_memalign\n");
            void *ptr = nullptr;
            int ret = posix_memalign(&ptr, 4096, num * size * 4);
            if (ret != 0 || ptr == nullptr) {
                fprintf(stderr, "[DRAMAlloc] posix_memalign failed (ret=%d)\n", ret);
                exit(EXIT_FAILURE);
            }
            array = (float *)ptr;
            using_pinned = false;
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
    bool using_pinned = false;

    static int thread_tid()
    {
        static std::atomic<int> next_tid{0};
        thread_local int tid = next_tid.fetch_add(1);
        return tid;
    }
};

#endif
