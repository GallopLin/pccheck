#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <atomic>
#include <cstring>
#include <unistd.h>
#include <fstream>
#include <pthread.h>
#include <sys/time.h>
#include <chrono>
#include <assert.h>
#include <stdio.h>
#include <linux/mman.h>
#include <sys/types.h>
#include <fcntl.h> /* Definition of AT_* constants */
#include <sys/stat.h>
#include <sys/mman.h>
#include <libpmem.h>
#include <sys/stat.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>
#include <stdint.h>
#include <queue>
#include <condition_variable>
#include <functional>
#include <unordered_map>
#include "FAAQueue.h"
#include "DRAMAlloc.h"
#include "socket_work.h"


using namespace std;
#define CACHELINES_IN_1G 16777216
#define BYTES_IN_1G 1073741824
#define CACHELINE_SIZE 64
#define OFFSET_SIZE 4096
#define FLOAT_IN_CACHE 16
#define REGION_SIZE 113406487152ULL
#define PR_FILE_NAME "/mnt/pmem0/file_1_1"
#define PR_DATA_FILE_NAME "/mnt/pmem0/file_1_2"
#define MAX_ITERATIONS 8

// In NMV: checkpoint * ; checkpoint init ; checkpoint area 1 ; ... ; checkpint area MAX_ITERATIONS;
// Then all the tensors data begin. TODO: check if better to put checkpoint near data
// First, there is a checkpoint pointer. It points to the metadata of the current active chekpoint.
// Then, there is the metadata of the initial checkpoint. Used only once for initialization.
// Afterwards, there are MAX_ITERATIONS checkpoint metadatas that are updated one at a time
// when a new checkpoint is registered.
// After all the metadata, the real tensors data begin. The metadata points to its relevant data.
//=================== || ==================== || ========== ... ========== || =======================
//                    ||                      ||                           ||
//    Checkpoint *    || Checkpoint Metadata  ||    Checkpoint Metadata    ||   Checkpoint Metadata
//                    ||        init          || 0, ... MAX_ITERATIONS - 2 ||    MAX_ITERATIONS - 1
//                    ||                      ||                           ||
//=================== || ==================== || ========== ... ========== || =======================
//=================== || ==================== || ========== ... ========== || =======================
//                    ||                      ||                           ||
//     Checkpoint 0   ||     Checkpoint 1     ||      Checkpoint 2, ...    ||        Checkpoint
//                    ||                      ||     MAX_ITERATIONS - 2    ||      MAX_ITERATIONS - 1
//                    ||                      ||                           ||
//=================== || ==================== || ========================= || =======================

static int curr_running = 0;

static int *PR_ADDR;
static int *PEER_CHECK_ADDR;
static int PADDING[64];
static int *PR_ADDR_DATA;

static uint8_t Cores[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

struct thread_data
{
    uint32_t id;
    float *arr;
    float *pr_arr;
    uint32_t size;
} __attribute__((aligned(64)));

struct checkpoint
{
    long area;
    long counter;
} __attribute__((aligned(64)));

// NUM_THREADS = # of parallel threads that works on a single checkpoint
// ASYNC_CHECK = # of maximal parallel checkpoints
// SIZE = writing size of a single write
// TEST_TYPE = use flushes (FLUSH_FENCE) or non-temporal stores (default)
// curr_parall_iter = # of current  parallel checkpoints
// counter = upgraded within each checkpoint. Tracks the newest one
static int NUM_THREADS = 16;
static int ASYNC_CHECK = 1;
static int SIZE = 512;
static string TEST_TYPE = "";
static atomic<int> curr_parall_iter __attribute__((aligned(64)));
static atomic<long> counter __attribute__((aligned(64)));
// static MSQueue<int> free_space;
static FAAArrayQueue<int> free_space;

int fd;
DRAMAlloc dram;

// for distributed
bool is_distributed;
int my_rank;
int world_size;
struct sockaddr_in socket_address;
int comm_fd;
int comm_socket;
vector<int> client_sockets;
int port = 1235;

//===================================================================

/* Allocate one core per thread */
static inline void set_cpu(int cpu)
{
    assert(cpu > -1);
    int n_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu < n_cpus)
    {
        int cpu_use = Cores[cpu];
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu_use, &mask);
        pthread_t thread = pthread_self();
        if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &mask) != 0)
        {
            fprintf(stderr, "Error setting thread affinity\n");
        }
    }
}

//===================================================================

static void printHelp()
{
    cout << "  -T     test type" << endl;
    cout << "  -N     thread num" << endl;
    cout << "  -C     asynchronous checkpoints number" << endl;
    cout << "  -S     writing size" << endl;
}

//===================================================================

static bool parseArgs(int argc, char **argv)
{
    int arg;
    while ((arg = getopt(argc, argv, "T:N:C:S:H")) != -1)
    {
        switch (arg)
        {
        case 'T':
            TEST_TYPE = string(optarg);
            break;
        case 'N':
            NUM_THREADS = atoi(optarg);
            break;
        case 'C':
            ASYNC_CHECK = atoi(optarg);
            break;
        case 'S':
            SIZE = atoi(optarg);
            break;
        case 'H':
            printHelp();
            return false;
        default:
            return false;
        }
    }
    return true;
}

//====================================================================

static void mapPersistentRegion(const char *filename, int *regionAddr, const uint64_t regionSize, bool data, int fd)
{

    size_t mapped_len;
    int is_pmem;
    /*if (data) {
        if ((PR_ADDR_DATA = (int*)pmem_map_file(filename, regionSize, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmem)) == NULL) {
            perror("pmem_map_file");
            exit(1);
        }
    } else {
        if ((PR_ADDR = (int*)pmem_map_file(filename, regionSize, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmem)) == NULL) {
            perror("pmem_map_file");
            exit(1);
        }
    }
    assert (is_pmem > 0);*/
    if (data)
    {
        if ((PR_ADDR_DATA = (int *)mmap(NULL, regionSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) == MAP_FAILED)
        {
            perror("mmap_file");
            exit(1);
        }
    }
    else
    {
        if ((PR_ADDR = (int *)mmap(NULL, regionSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) == MAP_FAILED)
        {
            perror("mmap_file");
            exit(1);
        }
    }
}

//====================================================================

static void FLUSH(void *p)
{
    asm volatile("clwb (%0)" ::"r"(p));
}

static void SFENCE()
{
    asm volatile("sfence" ::: "memory");
}

static void BARRIER(void *p)
{
    FLUSH(p);
    SFENCE();
}

//====================================================================

static void initialize(const char *filename, int max_async, size_t batch_size_floats, size_t num_batches, bool dist, int dist_rank, int wsize)
{

    struct stat buffer;
    bool newfile = (stat(filename, &buffer) == -1);

    fd = open(filename, O_CREAT | O_RDWR | O_TRUNC, (mode_t)0666);
    ftruncate(fd, REGION_SIZE);

    //mapPersistentRegion(filename, PR_ADDR_DATA, REGION_SIZE, true, fd);
    // 现在和filename无关（与fd有关），PR_ADDR PR_ADDR_DATA都是mmap出来的，与第二个参数无关（有关逻辑被注释了）
    mapPersistentRegion(filename, PR_ADDR, REGION_SIZE, false, fd);
    PEER_CHECK_ADDR = PR_ADDR + OFFSET_SIZE;
    PR_ADDR_DATA = PR_ADDR + (max_async+3)*OFFSET_SIZE;

    printf("PR_ADDR_DATA is %p, PR_ADDR is %p, PEER_CHECK_ADDR is %p\n", PR_ADDR_DATA, PR_ADDR, PEER_CHECK_ADDR);

    curr_parall_iter.store(0);
    counter.store(1);

    // write init checkpoint on NVMM - locted right next to the checkpoint *
    void *next_addr = PR_ADDR + 2*OFFSET_SIZE; // sizeof(checkpoint*) == CACHELINE_SIZE
    struct checkpoint check = {0, 0};
    // pmem_memcpy_persist(next_addr, &check, sizeof(struct checkpoint));
    // pmem_memcpy_persist(PR_ADDR, &next_addr, sizeof(struct checkpoint*));

    memcpy(next_addr, &check, sizeof(struct checkpoint));
    int res = msync((void *)PR_ADDR, 2*OFFSET_SIZE + sizeof(struct checkpoint), MS_SYNC);
    if (res == -1)
    {
        perror("msync - init, next_addr ");
        exit(1);
    }

    memcpy(PR_ADDR, &next_addr, sizeof(struct checkpoint *));
    res = msync((void *)PR_ADDR, sizeof(struct checkpoint *), MS_SYNC);
    if (res == -1)
    {
        perror("msync - init, PR_ADDR");
        exit(1);
    }

    memcpy(PEER_CHECK_ADDR, &next_addr, sizeof(struct checkpoint *));
    res = msync((void *)PEER_CHECK_ADDR, sizeof(struct checkpoint *), MS_SYNC);
    if (res == -1)
    {
        perror("msync - init, PEER_CHECK_ADDR");
        exit(1);
    }

    // insert the current free data slots in the file
    for (int i = 0; i <= max_async; i++)
    {
        printf("--------------- init enqueue: %d\n", i);
        free_space.enqueue(i, 0);
    }

    printf("Call dram.alloc, num_batches is %lu, batch_size_floats is %lu\n", num_batches, batch_size_floats);
    dram.alloc(num_batches, batch_size_floats);
    dram.initialize(num_batches, batch_size_floats);

    is_distributed = dist;
    my_rank = dist_rank;
    world_size = wsize;

    printf("------------------------- is_distributed is %d, rank is %d\n", is_distributed, my_rank);
    if (is_distributed) {

        char const* tmp = getenv("PCCHECK_COORDINATOR");
        std::string server_ip(tmp);

        printf("My rank is %d\n", my_rank);
        if (my_rank==0) {
            setup_rank0_socket(port, &comm_fd, &socket_address, world_size-1, client_sockets);
        }
        else {
            setup_other_socket(&comm_socket, &socket_address, server_ip, port);
        }
    }


}

//====================================================================

/* Provides ways to write data to a dedicated address within PR_ADDR_DATA.
 * savenvm_thread_flush and savenvm_thread_nd writes and persists the
 * data to a dedicated address. These methods are called by every parallel
 * thread that writes within a single checkpoint (out of NUM_THREADS).
 * savenvm synchronizes the entire checkpoint. Called every time there is
 * a new checkpoint to be written. */

// 多流写入支持的结构体
struct StreamWriter {
    int stream_id;                    // 流ID (0: param, 1: grad, 2: exp_avg, 3: exp_avg_sq)
    size_t stream_offset;             // 在文件中的偏移
    size_t stream_size;               // 流的总大小（元素个数）
    float* cpu_buffer;                // CPU缓冲区
    bool is_writing;                  // 是否正在写入
    
    StreamWriter() : stream_id(-1), stream_offset(0), stream_size(0), 
                    cpu_buffer(nullptr), is_writing(false) {}
};

// 线程池实现（用于write_stream_chunk，减少线程创建/销毁开销）
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
    int num_threads;
    
public:
    ThreadPool(int num_workers) : stop(false), num_threads(num_workers) {
        for (int i = 0; i < num_workers; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { 
                            return this->stop || !this->tasks.empty(); 
                        });
                        
                        if (this->stop && this->tasks.empty()) {
                            return;
                        }
                        
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
        printf("ThreadPool initialized with %d workers\n", num_workers);
    }
    
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) {
                return;
            }
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }
    
    void wait_all() {
        // 等待所有任务完成
        while (true) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (tasks.empty()) {
                break;
            }
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) {
            worker.join();
        }
        printf("ThreadPool destroyed\n");
    }
};

class NVM_write
{
private:
    std::vector<std::unique_ptr<StreamWriter>> streams;
    std::atomic<int> active_streams;
    bool multistream_mode;
    size_t total_checkpoint_size;     // 单个检查点的总大小（以float为单位）
    size_t checkpoint_padding_bytes;  // 单个checkpoint的4KB对齐padding字节数
    size_t checkpoint_stride_floats;  // 单个checkpoint的stride（以float为单位，包含padding）
    size_t chunk_threshold_bytes;     // 动态线程切分阈值（字节），根据检查点大小自动调整
    std::unique_ptr<ThreadPool> thread_pool;  // 线程池（用于write_stream_chunk）
    int pool_num_threads;             // 线程池大小
    std::unordered_map<int, float*> slot_buffers;  // 槽位→pinned buffer映射（用于DRAM复用）
    std::mutex slot_mutex;             // 保护slot_buffers的互斥锁
    
    // ✅ 对齐开销彻底固定：复用checkpoint基地址计算，避免每次重复计算offset
    float* checkpoint_base_ptr(int parall_iter) {
        if (checkpoint_stride_floats == 0) {
            fprintf(stderr, "[ERROR] checkpoint_stride_floats not initialized! Call init_streams first.\n");
            exit(1);
        }
        return (float *)PR_ADDR_DATA + (parall_iter * checkpoint_stride_floats);
    }
    
public:
    NVM_write() : active_streams(0), multistream_mode(false), total_checkpoint_size(0), 
                  checkpoint_padding_bytes(0), checkpoint_stride_floats(0), 
                  chunk_threshold_bytes(64 * 1024 * 1024), pool_num_threads(16) {
        // 默认创建12个工作线程的线程池
        // 默认阈值64MB（适合2-10GB检查点）
        thread_pool = std::make_unique<ThreadPool>(16);
    }
    
    // 初始化多个数据流
    // ✅ 对齐开销彻底固定：提前计算checkpoint_padding_bytes和checkpoint_stride_floats
    int init_streams(int num_streams, size_t* stream_sizes) {
        printf("Initializing %d streams for multi-stream checkpoint\n", num_streams);
        streams.clear();
        streams.reserve(num_streams);
        multistream_mode = true;
        
        size_t cumulative_offset = 0;
        total_checkpoint_size = 0;
        
        for (int i = 0; i < num_streams; i++) {
            auto stream = std::make_unique<StreamWriter>();
            stream->stream_id = i;
            stream->stream_offset = cumulative_offset;
            stream->stream_size = stream_sizes[i];
            stream->is_writing = false;
            
            // 不在C++端分配缓冲区，Python端已经有pinned memory
            // 这样可以避免大内存分配失败的问题
            stream->cpu_buffer = nullptr;
            
            cumulative_offset += stream_sizes[i];
            total_checkpoint_size += stream_sizes[i];
            
            printf("  Stream %d: offset=%lu, size=%lu floats (%.2f GB)\n", 
                   i, stream->stream_offset, stream->stream_size,
                   (stream_sizes[i] * sizeof(float)) / 1e9);
            
            streams.push_back(std::move(stream));
        }
        
        // ✅ 对齐开销彻底固定：提前计算4KB对齐的padding和stride
        size_t checkpoint_bytes = total_checkpoint_size * sizeof(float);
        size_t page_size = 4096;
        
        // 计算需要多少padding才能对齐到4KB
        checkpoint_padding_bytes = (page_size - (checkpoint_bytes % page_size)) % page_size;
        // 避免"整除却仍补4KB"的额外浪费：如果已经对齐，padding为0
        
        // checkpoint_stride_floats = 实际数据 + padding（转换为float单位）
        checkpoint_stride_floats = total_checkpoint_size + (checkpoint_padding_bytes / sizeof(float));
        
        // ✅ 动态线程切分阈值：根据检查点大小自动调整
        // 策略：
        // - 小检查点(<1GB): 32MB阈值
        // - 中检查点(1-5GB): 64MB阈值
        // - 大检查点(>5GB): 128MB阈值 (避免多流模式下线程竞争)
        if (checkpoint_bytes < 1ULL * 1024 * 1024 * 1024) {
            chunk_threshold_bytes = 64 * 1024 * 1024;
        } else if (checkpoint_bytes < 5ULL * 1024 * 1024 * 1024) {
            chunk_threshold_bytes = 128 * 1024 * 1024;
        } else {
            chunk_threshold_bytes = 256 * 1024 * 1024;
        }
        
        printf("Total checkpoint size: %lu floats (%.2f GB)\n", 
               total_checkpoint_size, checkpoint_bytes / 1e9);
        printf("Checkpoint padding: %lu bytes (%.2f KB), stride: %lu floats\n",
               checkpoint_padding_bytes, checkpoint_padding_bytes / 1024.0, checkpoint_stride_floats);
        printf("Dynamic chunk threshold: %.2f MB (auto-adjusted for checkpoint size)\n",
               chunk_threshold_bytes / (1024.0 * 1024.0));
        
        return 0;
    }
    
    // 异步写入单个流的数据块（使用线程池，减少线程创建/销毁开销）
    // ✅ 动态线程切分：按4MB粒度估算线程数，小块退化为单线程，大块多线程，64B对齐
    void write_stream_chunk(int stream_id, float* data, size_t offset_in_stream, 
                           size_t chunk_size, int parall_iter, int num_threads) {
        if (stream_id < 0 || stream_id >= (int)streams.size()) {
            fprintf(stderr, "[ERROR] Invalid stream_id: %d\n", stream_id);
            return;
        }
        
        if (!thread_pool) {
            fprintf(stderr, "[ERROR] Thread pool not initialized\n");
            return;
        }
        
        if (checkpoint_stride_floats == 0) {
            fprintf(stderr, "[ERROR] checkpoint_stride_floats not initialized! Call init_streams first.\n");
            exit(1);
        }
        
        StreamWriter* stream = streams[stream_id].get();
        stream->is_writing = true;
        
        // ✅ 对齐开销彻底固定：使用checkpoint_base_ptr复用地址计算
        float* checkpoint_base = checkpoint_base_ptr(parall_iter);
        float* target_addr = checkpoint_base + stream->stream_offset + offset_in_stream;
        
        size_t chunk_bytes = chunk_size * sizeof(float);
        const size_t CACHE_LINE_SIZE = 64;  // 64B对齐
        
        // ✅ 动态线程切分：使用动态阈值（根据检查点大小自动调整）
        // 阈值已在init_streams中根据检查点大小设置（16MB/32MB/64MB）
        int actual_threads = 1;
        if (chunk_bytes >= chunk_threshold_bytes) {
            // 大块数据：按动态阈值粒度估算线程数，最多不超过线程池规模
            actual_threads = std::min((int)(chunk_bytes / chunk_threshold_bytes) + 1, pool_num_threads);
            actual_threads = std::min(actual_threads, (int)num_threads);
            // 确保至少2个线程才使用多线程（避免单线程路径的开销）
            if (actual_threads < 2) {
                actual_threads = 1;
            }
        }
        // 小块数据退化为单线程memcpy
        
        if (actual_threads == 1) {
            // 单线程路径：直接memcpy，无需同步开销
            memcpy(target_addr, data, chunk_bytes);
        } else {
            // ✅ 同步路径瘦身：多线程路径按byte范围拷贝，复用atomic + condition_variable
            std::atomic<int> completed_tasks(0);
            std::mutex completion_mutex;
            std::condition_variable completion_cv;
            
            size_t bytes_per_thread = chunk_bytes / actual_threads;
            // ✅ 64B对齐：每个任务对齐到cache line边界
            bytes_per_thread = (bytes_per_thread / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;
            
            for (int i = 0; i < actual_threads; i++) {
                size_t byte_start = i * bytes_per_thread;
                size_t byte_end = (i == actual_threads - 1) ? chunk_bytes : (i + 1) * bytes_per_thread;
                size_t copy_bytes = byte_end - byte_start;
                
                // 确保对齐到64B边界
                if (copy_bytes > 0) {
                    thread_pool->enqueue([=, &completed_tasks, &completion_cv, &completion_mutex]() {
                        // 按byte范围拷贝（使用char*指针）
                        memcpy((char*)target_addr + byte_start, (char*)data + byte_start, copy_bytes);
                        
                        int remaining = completed_tasks.fetch_add(1) + 1;
                        if (remaining == actual_threads) {
                            std::unique_lock<std::mutex> lock(completion_mutex);
                            completion_cv.notify_one();
                        }
                    });
                }
            }
            
            // 等待所有任务完成
            {
                std::unique_lock<std::mutex> lock(completion_mutex);
                completion_cv.wait(lock, [&completed_tasks, actual_threads] {
                    return completed_tasks.load() == actual_threads;
                });
            }
        }
        
        stream->is_writing = false;
        // ✅ 同步路径瘦身：去掉热路径printf，减少syslog开销
    }
    
    // 同步单个流到磁盘
    // ✅ 同步/flush逻辑复用：使用checkpoint_base_ptr确保和写入路径一致
    void sync_stream(int stream_id, int parall_iter) {
        if (stream_id < 0 || stream_id >= (int)streams.size()) {
            fprintf(stderr, "[ERROR] Invalid stream_id for sync: %d\n", stream_id);
            return;
        }
        
        if (checkpoint_stride_floats == 0) {
            fprintf(stderr, "[ERROR] checkpoint_stride_floats not initialized! Call init_streams first.\n");
            exit(1);
        }
        
        StreamWriter* stream = streams[stream_id].get();
        
        // ✅ 同步/flush逻辑复用：使用checkpoint_base_ptr，确保和写入路径一致
        float* checkpoint_base = checkpoint_base_ptr(parall_iter);
        float* target_addr = checkpoint_base + stream->stream_offset;
        size_t sync_bytes = stream->stream_size * sizeof(float);
        
        // msync 需要页对齐的地址和大小
        size_t page_size = 4096;
        uintptr_t addr_int = (uintptr_t)target_addr;
        uintptr_t page_start = (addr_int / page_size) * page_size;  // 向下对齐到页边界
        size_t offset_in_page = addr_int - page_start;
        size_t aligned_size = ((offset_in_page + sync_bytes + page_size - 1) / page_size) * page_size;
        
        void* aligned_addr = (void*)page_start;
        
        auto t1 = std::chrono::high_resolution_clock::now();
        int res = msync(aligned_addr, aligned_size, MS_SYNC);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        
        if (res == -1) {
            printf("[ERROR] Stream %d msync failed at addr %p (aligned from %p), size %lu (original %lu)\n", 
                   stream_id, aligned_addr, target_addr, aligned_size, sync_bytes);
            perror("msync");
            exit(1);
        }
        
        printf("[Stream %d] Synced %.2f GB to disk in %.2f ms\n",
               stream_id, sync_bytes / 1e9, ms_double.count());
    }
    
    // 同步所有流到磁盘，并更新检查点元数据
    // ✅ 优化：合并4个stream的msync为1次，减少系统调用开销和竞争
    // ✅ 同步/flush逻辑复用：使用checkpoint_base_ptr确保和写入路径一致
    void sync_all_streams(int parall_iter) {
        if (checkpoint_stride_floats == 0) {
            fprintf(stderr, "[ERROR] checkpoint_stride_floats not initialized! Call init_streams first.\n");
            exit(1);
        }
        
        printf("Syncing all %lu streams to parall_iter %d (merged msync)...\n", streams.size(), parall_iter);
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // ✅ 同步/flush逻辑复用：使用checkpoint_base_ptr，确保和写入路径一致
        float* checkpoint_base = checkpoint_base_ptr(parall_iter);
        
        // 计算整个检查点区域的大小（所有stream的总和，不包含padding）
        size_t total_sync_bytes = total_checkpoint_size * sizeof(float);
        
        // ✅ 优化：单次msync整个检查点区域，而不是4次分别msync
        // 这样可以：
        // 1. 减少系统调用次数（4次 → 1次）
        // 2. 避免内核锁竞争
        // 3. 让内核优化磁盘写入顺序
        
        // msync需要页对齐的地址和大小
        size_t page_size = 4096;
        uintptr_t addr_int = (uintptr_t)checkpoint_base;
        uintptr_t page_start = (addr_int / page_size) * page_size;  // 向下对齐到页边界
        size_t offset_in_page = addr_int - page_start;
        size_t aligned_size = ((offset_in_page + total_sync_bytes + page_size - 1) / page_size) * page_size;
        
        void* aligned_addr = (void*)page_start;
        
        int msync_res = msync(aligned_addr, aligned_size, MS_SYNC);
        if (msync_res == -1) {
            fprintf(stderr, "[ERROR] Merged msync failed at addr %p (aligned from %p), size %lu (original %lu)\n", 
                   aligned_addr, checkpoint_base, aligned_size, total_sync_bytes);
            perror("msync");
            exit(1);
        }
        
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        printf("All streams synced (merged) %.2f GB in %.2f ms (%.2f GB/s)\n",
               total_sync_bytes / 1e9, ms_double.count(), 
               (total_sync_bytes / 1e9) / (ms_double.count() / 1000.0));
        
        // 更新检查点元数据（沿用原pccheck的逻辑）
        struct checkpoint *curr_checkpoint = (struct checkpoint *)(PR_ADDR + OFFSET_SIZE * (parall_iter + 3));
        struct checkpoint *volatile last_check = *(struct checkpoint *volatile *)PR_ADDR;
        long curr_counter = curr_checkpoint->counter;
        
        // 同步元数据
        int metadata_res = msync(curr_checkpoint, sizeof(struct checkpoint), MS_SYNC);
        if (metadata_res == -1) {
            printf("[ERROR] Proc %d msync during checkpoint sync, with addr %p\n", parall_iter, curr_checkpoint);
            perror("msync");
            exit(1);
        }
        
        // CAS更新检查点指针
        while (true) {
            bool cas_res = __sync_bool_compare_and_swap((checkpoint **)PR_ADDR, last_check, curr_checkpoint);
            struct checkpoint *volatile check = *(struct checkpoint *volatile *)PR_ADDR;
            if (cas_res) {
                printf("CAS was successful! new counter is %ld\n", check->counter);
                BARRIER(PR_ADDR);
                
                // 释放旧的槽位
                int free = (((int *)last_check - PR_ADDR) / OFFSET_SIZE) - 3;
                if (free == -1)
                    return;
                free_space.enqueue(free, free);
                printf("Freed slot %d back to free_space\n", free);
                return;
            }
            else if (check->counter < curr_counter) {
                last_check = check;
                continue;
            }
            else {
                BARRIER(PR_ADDR);
                free_space.enqueue(parall_iter, parall_iter);
                printf("Freed slot %d (current) back to free_space\n", parall_iter);
                return;
            }
        }
    }
    
    // ✅ DRAM槽位管理：借用和归还pinned内存（用于多流pccheck内存复用）
    // 从DRAMAlloc借用一块pinned buffer
    float* borrow_chunk() {
        // 从DRAMAlloc借用一块pinned buffer
        // 使用0作为tid，因为DRAMAlloc内部其实忽略了这个参数(或者用于调试)
        float* buffer = dram.get_add(0);
        return buffer;
    }
    
    // 归还pinned buffer到DRAMAlloc
    void return_chunk(float* buffer) {
        if (buffer == nullptr) return;
        // 归还到DRAMAlloc
        dram.put_add(buffer, 0);
    }
    
    // 兼容旧接口（如果还需要的话，或者直接废弃）
    float* borrow_cpu_slot(int parall_iter) {
        return borrow_chunk();
    }
    
    void return_cpu_slot(int parall_iter) {
        // 旧接口无法正确归还，因为没有传入buffer指针
        // 这里什么都不做，或者打印错误
        fprintf(stderr, "[WARNING] return_cpu_slot(int) is deprecated and does nothing. Use return_chunk(float*) instead.\n");
    }
    
    // 清理资源
    ~NVM_write() {
        // cpu_buffer由Python端管理，这里不需要释放
        // 线程池会在析构时自动清理
        thread_pool.reset();
        streams.clear();
    }
    
    static void *take_cpu_address(size_t tid)
    {
        // TODO: what should the id be?
        void *ret = (void *)(dram.get_add(tid));
        return ret;
    }

    static void savenvm_thread_flush(thread_data *data)
    {
        int id = data->id;
        float *arr = data->arr;
        float *add = data->pr_arr;
        int sz = data->size;
        // set_cpu(id);
        for (int i = 0; i < sz;)
        {
            float *add_to_flush = add;
            for (int j = 0; j < FLOAT_IN_CACHE; j++)
            {
                *add = arr[i];
                i++;
                if (i == sz)
                    break;
                add++;
            }
            FLUSH(add_to_flush);
        }
        SFENCE();
    }

    static void savenvm_thread_nd(thread_data *data)
    {
        int id = data->id;
        float *arr = data->arr;
        float *add = (float *)data->pr_arr;
        size_t sz = data->size;

        set_cpu(id);
        // printf("At savenvm_thread_nd id is %d, sz is %lu!\n", id, sz);
        for (size_t i = 0; i < sz;)
        {

            // pmem_memcpy_nodrain((void*)add, (void*)arr, SIZE);
            memcpy((void *)add, (void *)arr, SIZE);
            arr += SIZE / sizeof(float);
            add += SIZE / sizeof(float);
            i += SIZE / sizeof(float);
        }
        // pmem_drain();
    }

    static int registerCheck()
    {
        int parall_iter = 0;
        // get a new counter for the current checkpoint attempt
        long curr_counter = atomic_fetch_add(&counter, (long)1);
        // find free space to update the new checkpoint

        while (true)
        {
            parall_iter = free_space.dequeue(parall_iter);
            if (parall_iter == INT_MIN)
                continue;
            else
                break;
        }

        // get the metadata address of the new slot
        struct checkpoint *curr_checkpoint = (struct checkpoint *)(PR_ADDR + OFFSET_SIZE * (parall_iter + 3));
        struct checkpoint curr_check = {parall_iter, curr_counter};
        printf("Parall_iter %d, Write new metadata at address %p\n", parall_iter, curr_checkpoint);
        memcpy(curr_checkpoint, &curr_check, sizeof(struct checkpoint));
        return parall_iter;
    }

    static void savenvmNew(size_t tid, float *arr, size_t total_size, int num_threads, int parall_iter, int batch_num, size_t batch_size, bool last_batch)
    {

        printf("------------------------- savenvmNew, is_distributed is %d\n", is_distributed);

        // check the last updated checkpoint. Tries to change this value only in the last batch
        struct checkpoint *checkp_info_new = (struct checkpoint *)(PR_ADDR + OFFSET_SIZE * (parall_iter + 3));

        // int parallel_iteration = checkp_info_new->area;
        int counter_num = checkp_info_new->counter;
        struct checkpoint *volatile last_check = *(struct checkpoint *volatile *)PR_ADDR;
        long curr_counter = counter_num;
        // int parall_iter = parallel_iteration;

        printf("At savenvmNew, tid is %d, parall_iter is %d, num_threads is %d, last counter is %d, curr_counter is %d\n", tid, parall_iter, num_threads, last_check->counter, curr_counter);
        if (last_check->counter > curr_counter)
        { // Room for optimization
            if (last_batch)
            {
                BARRIER(PR_ADDR);
                printf("Return!\n");
                free_space.enqueue(parall_iter, parall_iter);
            }
            return;
        }

        float *curr_arr = arr; // address of the current batch
        size_t size_for_thread = batch_size / num_threads;
        size_t reminder = batch_size % num_threads;

        // get the metadata address of the new slot - already filled in the register function
        struct checkpoint *curr_checkpoint = checkp_info_new;
        // get the data address of the new slot - batches start from 1

        // make sure the start address is aligned at 4KB
        size_t offset = 4096 - (total_size * sizeof(float)) % 4096;
        float *start_pr_arr = NULL;
        // printf("offset is %ld\n", offset);
        start_pr_arr = (float *)PR_ADDR_DATA + (parall_iter * total_size) + (parall_iter * offset) / 4;
        float *curr_pr_arr = start_pr_arr + (batch_size * (batch_num - 1));

        thread *threads[num_threads];
        thread_data allThreadsData[num_threads];

        size_t num_floats_SIZE = SIZE / sizeof(float);
        size_t rem_floats_SIZE = size_for_thread % num_floats_SIZE;
        size_t curr_sz = 0;

        size_t total_batches = total_size / batch_size;
        int thread_offset = (num_threads * (batch_num - 1)) + (parall_iter * total_batches * num_threads);

        printf("Save checkpoint - create threads!\n");

        for (int i = 0; i < num_threads; i++)
        {
            size_t size_for_thread_i = size_for_thread;
            // all should be multiple of SIZE
            size_for_thread_i += num_floats_SIZE - rem_floats_SIZE;
            size_for_thread_i = std::min(size_for_thread_i, batch_size - curr_sz);

            thread_data &data = allThreadsData[i];

            // take into a consideration all the running threads in the system
            // data.id =(i + 1 + thread_offset) % 46; //TODO: fix this - need to actually verify how many threads are currently running
            data.id = parall_iter * num_threads + i + 1;
            // the address to copy from
            data.arr = curr_arr;
            // the address to copy to
            data.pr_arr = curr_pr_arr + 4096;
            data.size = size_for_thread_i;
            threads[i] = new thread(&savenvm_thread_nd, &data);
            curr_arr += size_for_thread_i;
            curr_pr_arr += size_for_thread_i;
            curr_sz += size_for_thread_i;
        }

        for (int j = 0; j < num_threads; j++)
        {
            threads[j]->join();
        }

        //printf("AFTER ALL JOINED, tid is %d, parall_iter is %d, num_threads is %d\n", tid, parall_iter, num_threads);

        // free mem copy
        // TODO: what should id be?
        dram.put_add(arr, tid);
        //printf("AFTER PUT_ADD, tid is %d, parall_iter is %d, num_threads is %d\n", tid, parall_iter, num_threads);

        if (last_batch)
        {
            // do a total msync here
            auto t1 = std::chrono::high_resolution_clock::now();
            int res = msync((void *)(start_pr_arr), total_size * sizeof(float), MS_SYNC);
            if (res == -1)
            {
                printf("[ERROR] Proc %d msync during model persisting addr %p, size %lu\n", parall_iter, start_pr_arr, total_size * sizeof(float));
                exit(1);
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ms_double = t2 - t1;
            printf("MSYNC TOOK %f ms\n", ms_double.count());
        }

        //printf("AFTER MSYNC, tid is %d, parall_iter is %d, num_threads is %d\n", tid, parall_iter, num_threads);

        // cout << "--------------FINISH SAVE NVM-------------" << endl;

        if (last_batch)
        {
            int res = msync(curr_checkpoint, sizeof(struct checkpoint), MS_SYNC);
            if (res == -1)
            {
                printf("[ERROR] Proc %d msync during checkpoint sync, with addr %p\n", parall_iter, curr_checkpoint);
                exit(1);
            }
            while (true)
            {
                bool res = __sync_bool_compare_and_swap((checkpoint **)PR_ADDR, last_check, curr_checkpoint);
                struct checkpoint *volatile check = *(struct checkpoint *volatile *)PR_ADDR;
                if (res)
                {
                    printf("CAS was successful! new counter is %ld, is_distributed is %d\n", check->counter, is_distributed);
                    BARRIER(PR_ADDR);
                    if (is_distributed) {
                        if (my_rank==0) {
                            wait_to_receive(client_sockets, world_size-1);
                        }
                        else {
                            send_and_wait(&comm_socket, curr_counter);
                        }

                        memcpy(PEER_CHECK_ADDR, curr_checkpoint, sizeof(struct checkpoint *));
                        res = msync((void *)PEER_CHECK_ADDR, sizeof(struct checkpoint *), MS_SYNC);
                        if (res == -1) {
                            perror("msync - init, PEER_CHECK_ADDR");
                            exit(1);
                        }
                    }

                    int free = (((int *)last_check - PR_ADDR) / OFFSET_SIZE) - 3;
                    if (free == -1)
                        return;
                    free_space.enqueue(free, free);
                }
                else if (check->counter < curr_counter)
                {
                    last_check = check;
                    continue;
                }
                else
                {
                    BARRIER(PR_ADDR);
                    free_space.enqueue(parall_iter, parall_iter);
                }
                return;
            }
        }
        return;
    }

    float *readfromnvm(float *ar, int size)
    {
        float *w = (float *)PR_ADDR_DATA;
        for (int i = 0; i < size; i++)
        {
            ar[i] = *w;
            w++;
        }
        return nullptr;
    }
};

//====================================================================

extern "C" // 生成的.so文件暴露出来的接口
{

    NVM_write *writer(const char *filename, int max_async, size_t batch_size_floats, size_t num_batches, bool dist, int dist_rank, int world_size)
    {
        NVM_write *nvmobj = new NVM_write();
        // printf("%s\n", filename);
        // ✅ 内存优化：对于多流pccheck，num_batches应该至少为max_async + 1
        // 这样可以确保DRAMAlloc有足够的pinned buffer槽位供并行检查点复用
        // 如果传入的num_batches小于max_async + 1，则使用max_async + 1
        // size_t actual_num_batches = (num_batches < (size_t)(max_async + 1)) ? (max_async + 1) : num_batches;
        initialize(filename, max_async, batch_size_floats, num_batches, dist, dist_rank, world_size);
        return nvmobj;
    }

    float *readfromnvm(NVM_write *t, float *ar, int sz)
    {
        return t->readfromnvm(ar, sz);
    }

    int registerCheck(NVM_write *t)
    {
        return t->registerCheck();
    }

    void *take_cpu_address(NVM_write *t, size_t tid)
    {
        return t->take_cpu_address(tid);
    }

    // registerCheck() {}
    void savenvm_new(NVM_write *t, size_t tid, float *arr, size_t total_size, int num_threads, int parall_iter, int batch_num, size_t batch_size, bool last_batch)
    {

        t->savenvmNew(tid, arr, total_size, num_threads, parall_iter, batch_num, batch_size, last_batch);
    }
    
    // ========== 多流并行写入接口 ==========
    
    // 初始化多个数据流
    int init_streams(NVM_write *t, int num_streams, size_t *stream_sizes) {
        return t->init_streams(num_streams, stream_sizes);
    }
    
    // 写入单个流的数据块
    void write_stream_chunk(NVM_write *t, int stream_id, float *data, 
                           size_t offset_in_stream, size_t chunk_size, 
                           int parall_iter, int num_threads) {
        t->write_stream_chunk(stream_id, data, offset_in_stream, chunk_size, 
                             parall_iter, num_threads);
    }
    
    // 同步单个流到磁盘
    void sync_stream(NVM_write *t, int stream_id, int parall_iter) {
        t->sync_stream(stream_id, parall_iter);
    }
    
    // 同步所有流到磁盘（沿用原pccheck的槽位管理和元数据更新）
    void sync_all_streams(NVM_write *t, int parall_iter) {
        t->sync_all_streams(parall_iter);
    }
    
    // ✅ DRAM槽位管理接口（用于多流pccheck内存复用）
    // 借用一块pinned buffer
    float* borrow_chunk(NVM_write *t) {
        return t->borrow_chunk();
    }
    
    // 归还pinned buffer
    void return_chunk(NVM_write *t, float* buffer) {
        t->return_chunk(buffer);
    }

    // 兼容旧接口
    float* borrow_cpu_slot(NVM_write *t, int parall_iter) {
        return t->borrow_cpu_slot(parall_iter);
    }
    
    void return_cpu_slot(NVM_write *t, int parall_iter) {
        t->return_cpu_slot(parall_iter);
    }
}

int main(int argc, char **argv)
{
    return 0;
}
