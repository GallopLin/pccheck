#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <atomic>
#include <cstring>
#include <cerrno>
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
#include <libaio.h>
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
#include <deque>
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
static uint64_t MAPPED_SIZE = 0;

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
    int64_t area;
    int64_t counter;
} __attribute__((aligned(64)));

// Relocatable publication record.  The original format stores an absolute
// mmap pointer at byte 0, which is only valid inside the writer process.  New
// writers also persist this record in unused header space; readers fall back
// to scanning metadata counters for clean legacy files.
struct published_checkpoint_record
{
    uint64_t magic;
    int64_t area;
    int64_t counter;
    uint64_t checksum;
};

static constexpr uint64_t PUBLISHED_RECORD_MAGIC = 0x43534E4150534C54ULL;
static constexpr size_t PUBLISHED_RECORD_BYTE_OFFSET = 64;

static uint64_t published_record_checksum(int64_t area, int64_t counter)
{
    return PUBLISHED_RECORD_MAGIC ^ (uint64_t)area ^
           ((uint64_t)counter * 0x9E3779B185EBCA87ULL);
}

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
static atomic<int64_t> counter __attribute__((aligned(64)));
// static MSQueue<int> free_space;
static FAAArrayQueue<int> free_space;

int fd = -1;
int fd_direct = -1;
DRAMAlloc dram;

static constexpr size_t DIRECT_IO_ALIGNMENT = 4096;

static bool data_direct_requested()
{
    const char *e = getenv("PCCHECK_DATA_DIRECT");
    return e == nullptr || strcmp(e, "0") != 0;
}

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

static uint64_t getFileSize(const char *filename, int fd)
{
    struct stat st;
    if (fstat(fd, &st) != 0)
    {
        perror("fstat");
        exit(1);
    }
    if (st.st_size == 0)
    {
        fprintf(stderr, "[ERROR] file size is zero for %s\n", filename);
        exit(1);
    }
    return (uint64_t)st.st_size;
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
    if (fd < 0) {
        perror("open checkpoint file");
        exit(1);
    }
    if (ftruncate(fd, REGION_SIZE) != 0) {
        perror("ftruncate");
        exit(1);
    }

    // The metadata mapping and small unaligned boundaries retain a normal
    // descriptor.  Bulk checkpoint data uses a second descriptor for the same
    // inode with O_DIRECT, so the already-pinned CUDA host buffer is submitted
    // to storage without creating a second checkpoint-sized page-cache copy.
    if (data_direct_requested()) {
        fd_direct = open(filename, O_RDWR | O_DIRECT, (mode_t)0666);
        if (fd_direct < 0) {
            fprintf(stderr,
                    "[WARN] O_DIRECT unavailable for %s (%s); using buffered pwrite\n",
                    filename, strerror(errno));
        } else {
            printf("Checkpoint data path: O_DIRECT enabled (alignment=%zu bytes)\n",
                   DIRECT_IO_ALIGNMENT);
        }
    } else {
        printf("Checkpoint data path: buffered pwrite (PCCHECK_DATA_DIRECT=0)\n");
    }

    MAPPED_SIZE = REGION_SIZE;

    //mapPersistentRegion(filename, PR_ADDR_DATA, REGION_SIZE, true, fd);
    // 现在和filename无关（与fd有关），PR_ADDR PR_ADDR_DATA都是mmap出来的，与第二个参数无关（有关逻辑被注释了）
    mapPersistentRegion(filename, PR_ADDR, REGION_SIZE, false, fd);
    PEER_CHECK_ADDR = PR_ADDR + OFFSET_SIZE;
    // 头部占用：checkpoint* | PEER | init | metadata[parall_iter 0..max_async]
    // = (max_async+4) 个 OFFSET_SIZE。旧值 (max_async+3) 使 parall_iter==max_async
    // 的 metadata 槽（位于 OFFSET_SIZE*(max_async+3)）与 slot0 数据区前 64 字节
    // 重叠——每次发布最后一个槽位都会覆写 slot0 param 流开头（byte-diff 实锤）。
    PR_ADDR_DATA = PR_ADDR + (max_async+4)*OFFSET_SIZE;

    published_checkpoint_record published = {
        PUBLISHED_RECORD_MAGIC, -1, 0, published_record_checksum(-1, 0)};
    memcpy((char *)PR_ADDR + PUBLISHED_RECORD_BYTE_OFFSET,
           &published, sizeof(published));
    if (msync((void *)PR_ADDR, 4096, MS_SYNC) == -1) {
        perror("msync - initialize publication record");
        exit(1);
    }

    printf("PR_ADDR_DATA is %p, PR_ADDR is %p, PEER_CHECK_ADDR is %p\n", PR_ADDR_DATA, PR_ADDR, PEER_CHECK_ADDR);

    curr_parall_iter.store(0);
    counter.store(1);
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

        // 并行多作业隔离：coordinator 端口可由环境变量覆盖（默认 1235）
        char const* port_env = getenv("PCCHECK_COORD_PORT");
        if (port_env != nullptr && port_env[0] != '\0') {
            int p = atoi(port_env);
            if (p > 0 && p < 65536) port = p;
        }

        char const* tmp = getenv("PCCHECK_COORDINATOR");
        std::string server_ip;
        if (tmp != nullptr && tmp[0] != '\0') {
            server_ip = tmp;
        } else {
            // 单机多卡默认走本机回环地址，避免 std::string(nullptr) 崩溃
            server_ip = "127.0.0.1";
            fprintf(stderr,
                    "[WARN] PCCHECK_COORDINATOR is not set, fallback to %s\n",
                    server_ip.c_str());
        }

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

static void initialize_for_read(const char *filename, int max_async)
{
    fd_direct = -1;
    fd = open(filename, O_RDWR, (mode_t)0666);
    if (fd < 0)
    {
        perror("open");
        exit(1);
    }

    uint64_t region_size = getFileSize(filename, fd);
    MAPPED_SIZE = region_size;
    mapPersistentRegion(filename, PR_ADDR, region_size, false, fd);

    PEER_CHECK_ADDR = PR_ADDR + OFFSET_SIZE;
    PR_ADDR_DATA = PR_ADDR + (max_async + 4) * OFFSET_SIZE;  // 与 initialize() 保持一致

    printf("[Reader] PR_ADDR_DATA is %p, PR_ADDR is %p, PEER_CHECK_ADDR is %p\n", PR_ADDR_DATA, PR_ADDR, PEER_CHECK_ADDR);

    is_distributed = false;
    my_rank = 0;
    world_size = 1;
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
    struct DirectAioBatch;

    struct DirectAioRequest {
        struct iocb control{};
        std::mutex mutex;
        std::condition_variable ready;
        bool complete = false;
        long result = 0;
        long result2 = 0;
        size_t expected_bytes = 0;
        DirectAioBatch *batch = nullptr;
    };

    struct DirectAioBatch {
        std::vector<std::unique_ptr<DirectAioRequest>> requests;
        std::mutex mutex;
        std::condition_variable ready;
        size_t remaining = 0;
        size_t completed_bytes = 0;
        bool complete = false;
        bool success = true;
        bool submission_finished = false;
        bool buffer_return_scheduled = false;
        bool buffer_returned = false;
        float *base_buffer = nullptr;
        const char *full_src = nullptr;
        char *mmap_target = nullptr;
        size_t full_bytes = 0;
        off64_t full_offset = 0;
        size_t direct_bytes = 0;
        int parall_iter = -1;
    };

    struct PendingReclaimSegment {
        off64_t start;
        off64_t len;
        int parall_iter;
    };

    std::vector<std::unique_ptr<StreamWriter>> streams;
    std::atomic<int> active_streams;
    bool multistream_mode;
    size_t total_checkpoint_size;     // 单个检查点的总大小（以float为单位）
    size_t checkpoint_padding_bytes;  // 单个checkpoint的4KB对齐padding字节数
    size_t checkpoint_stride_floats;  // 单个checkpoint的stride（以float为单位，包含padding）
    // ✅ 移除了不再需要的成员变量：
    // - chunk_threshold_bytes: 多线程切分阈值（已移除多线程 memcpy）
    // - thread_pool: C++ 线程池（由 Python ssd_thread_pool 替代）
    // - pool_num_threads: 线程池大小
    std::unordered_map<int, float*> slot_buffers;  // 槽位→pinned buffer映射（用于DRAM复用）
    std::mutex slot_mutex;             // 保护slot_buffers的互斥锁
    std::mutex publish_mutex;
    int metadata_max_async = -1;
    // Process-wide reclaim accounting for this writer.  The former
    // thread_local windows silently multiplied the dirty-page allowance by
    // the Python SSD worker count.  A shared FIFO makes the configured budget
    // independent of that worker count.
    std::mutex reclaim_mutex;
    std::deque<PendingReclaimSegment> reclaim_pending;
    size_t reclaim_pending_bytes = 0;

    // Cumulative counters are intentionally lock-free: Python SSD workers
    // issue disjoint chunks concurrently.  They make direct-I/O coverage and
    // any unexpected buffered fallback observable in production profiles.
    std::atomic<uint64_t> direct_bytes{0};
    std::atomic<uint64_t> buffered_boundary_bytes{0};
    std::atomic<uint64_t> buffered_fallback_bytes{0};
    std::atomic<uint64_t> direct_calls{0};
    std::atomic<uint64_t> fallback_calls{0};
    std::atomic<uint64_t> direct_errors{0};
    std::atomic<uint64_t> durable_sync_calls{0};
    std::atomic<uint64_t> durable_sync_ns{0};
    std::atomic<uint64_t> direct_aio_subrequests{0};
    std::once_flag direct_aio_once;
    io_context_t direct_aio_context = 0;
    std::atomic<bool> direct_aio_ready{false};
    std::atomic<bool> direct_aio_stop{false};
    std::atomic<uint64_t> direct_aio_outstanding{0};
    std::thread direct_aio_completion_thread;
    std::mutex direct_async_batches_mutex;
    std::vector<std::shared_ptr<DirectAioBatch>> direct_async_batches;

    static bool direct_aio_requested() {
        const char *e = getenv("PCCHECK_DIRECT_AIO");
        return e == nullptr || strcmp(e, "0") != 0;
    }

    static size_t direct_aio_request_bytes() {
        static const size_t bytes = [] {
            const char *e = getenv("PCCHECK_DIRECT_AIO_REQUEST_MB");
            long mb = (e != nullptr && e[0] != '\0') ? atol(e) : 64;
            if (mb < 1) mb = 1;
            if (mb > 64) mb = 64;
            size_t value = (size_t)mb * 1024 * 1024;
            return (value / DIRECT_IO_ALIGNMENT) * DIRECT_IO_ALIGNMENT;
        }();
        return bytes;
    }

    void direct_aio_completion_loop() {
        constexpr long max_events = 64;
        struct io_event events[max_events];
        while (!direct_aio_stop.load(std::memory_order_acquire) ||
               direct_aio_outstanding.load(std::memory_order_acquire) != 0) {
            struct timespec timeout{0, 100000000};  // 100 ms
            int count = io_getevents(direct_aio_context, 1, max_events,
                                     events, &timeout);
            if (count < 0) {
                if (count == -EINTR) continue;
                static std::atomic<bool> warned{false};
                if (!warned.exchange(true)) {
                    errno = -count;
                    perror("[WARN] io_getevents");
                }
                continue;
            }
            for (int index = 0; index < count; ++index) {
                auto *request = static_cast<DirectAioRequest *>(
                    events[index].data);
                if (request->batch != nullptr) {
                    DirectAioBatch *batch = request->batch;
                    bool return_buffer = false;
                    bool batch_completed = false;
                    {
                        std::lock_guard<std::mutex> lock(batch->mutex);
                        request->result = events[index].res;
                        request->result2 = events[index].res2;
                        request->complete = true;
                        if (request->result > 0) {
                            batch->completed_bytes += (size_t)request->result;
                        }
                        if (request->result != (long)request->expected_bytes ||
                            request->result2 != 0) {
                            batch->success = false;
                        }
                        if (--batch->remaining == 0 &&
                            batch->submission_finished) {
                            batch->complete = true;
                            batch_completed = true;
                            return_buffer = batch->success &&
                                            !batch->buffer_return_scheduled;
                            if (return_buffer) {
                                batch->buffer_return_scheduled = true;
                            }
                        }
                    }
                    if (return_buffer) {
                        direct_bytes.fetch_add(batch->direct_bytes,
                                               std::memory_order_relaxed);
                        dram.put_add(batch->base_buffer, 0);
                        {
                            std::lock_guard<std::mutex> lock(batch->mutex);
                            batch->buffer_returned = true;
                        }
                    }
                    direct_aio_outstanding.fetch_sub(
                        1, std::memory_order_release);
                    if (batch_completed) batch->ready.notify_all();
                    continue;
                }
                {
                    std::lock_guard<std::mutex> lock(request->mutex);
                    request->result = events[index].res;
                    request->result2 = events[index].res2;
                    request->complete = true;
                }
                direct_aio_outstanding.fetch_sub(1,
                                                  std::memory_order_release);
                request->ready.notify_one();
            }
        }
    }

    bool ensure_direct_aio() {
        if (!direct_aio_requested()) return false;
        std::call_once(direct_aio_once, [this] {
            int result = io_setup(256, &direct_aio_context);
            if (result < 0) {
                errno = -result;
                perror("[WARN] io_setup for O_DIRECT; using synchronous pwrite");
                direct_aio_context = 0;
                return;
            }
            direct_aio_ready.store(true, std::memory_order_release);
            direct_aio_completion_thread = std::thread(
                [this] { direct_aio_completion_loop(); });
        });
        return direct_aio_ready.load(std::memory_order_acquire);
    }

    bool direct_aio_write(const char *src, size_t bytes, off64_t offset,
                          size_t *completed) {
        if (!ensure_direct_aio()) return false;
        const size_t request_bytes = direct_aio_request_bytes();
        const size_t request_count =
            (bytes + request_bytes - 1) / request_bytes;
        std::vector<std::unique_ptr<DirectAioRequest>> owned;
        std::vector<struct iocb *> controls;
        owned.reserve(request_count);
        controls.reserve(request_count);

        size_t prepared = 0;
        for (size_t index = 0; index < request_count; ++index) {
            size_t length = std::min(request_bytes, bytes - prepared);
            auto request = std::make_unique<DirectAioRequest>();
            io_prep_pwrite(&request->control, fd_direct,
                           const_cast<char *>(src + prepared), length,
                           offset + (off64_t)prepared);
            request->control.data = request.get();
            controls.push_back(&request->control);
            owned.push_back(std::move(request));
            prepared += length;
        }

        size_t submitted_total = 0;
        int transient_retries = 0;
        bool submission_ok = true;
        while (submitted_total < request_count) {
            int remaining = (int)(request_count - submitted_total);
            direct_aio_outstanding.fetch_add((uint64_t)remaining,
                                              std::memory_order_release);
            int submitted = io_submit(direct_aio_context, remaining,
                                      controls.data() + submitted_total);
            if (submitted <= 0) {
                direct_aio_outstanding.fetch_sub((uint64_t)remaining,
                                                  std::memory_order_release);
                if ((submitted == -EAGAIN || submitted == 0) &&
                    transient_retries++ < 10000) {
                    std::this_thread::yield();
                    continue;
                }
                if (submitted < 0) errno = -submitted;
                submission_ok = false;
                break;
            }
            transient_retries = 0;
            if (submitted < remaining) {
                direct_aio_outstanding.fetch_sub(
                    (uint64_t)(remaining - submitted),
                    std::memory_order_release);
            }
            submitted_total += (size_t)submitted;
            direct_aio_subrequests.fetch_add((uint64_t)submitted,
                                              std::memory_order_relaxed);
        }

        size_t total_completed = 0;
        bool completions_ok = true;
        for (size_t index = 0; index < submitted_total; ++index) {
            DirectAioRequest &request = *owned[index];
            std::unique_lock<std::mutex> lock(request.mutex);
            request.ready.wait(lock, [&request] { return request.complete; });
            if (request.result > 0) total_completed += (size_t)request.result;
            if (request.result != (long)request.control.u.c.nbytes ||
                request.result2 != 0) {
                completions_ok = false;
                if (request.result < 0) errno = (int)-request.result;
                else if (request.result2 < 0) errno = (int)-request.result2;
            }
        }
        if (completed != nullptr) *completed = total_completed;
        return submission_ok && submitted_total == request_count &&
               completions_ok && total_completed == bytes;
    }

    bool direct_aio_submit_async(const char *src, size_t bytes, off64_t offset,
                                 float *base_buffer, const char *full_src,
                                 char *mmap_target, size_t full_bytes,
                                 off64_t full_offset, int parall_iter) {
        if (!ensure_direct_aio()) return false;
        auto batch = std::make_shared<DirectAioBatch>();
        batch->base_buffer = base_buffer;
        batch->full_src = full_src;
        batch->mmap_target = mmap_target;
        batch->full_bytes = full_bytes;
        batch->full_offset = full_offset;
        batch->direct_bytes = bytes;
        batch->parall_iter = parall_iter;

        const size_t request_bytes = direct_aio_request_bytes();
        const size_t request_count =
            (bytes + request_bytes - 1) / request_bytes;
        batch->requests.reserve(request_count);
        size_t prepared = 0;
        for (size_t index = 0; index < request_count; ++index) {
            size_t length = std::min(request_bytes, bytes - prepared);
            auto request = std::make_unique<DirectAioRequest>();
            request->expected_bytes = length;
            request->batch = batch.get();
            io_prep_pwrite(&request->control, fd_direct,
                           const_cast<char *>(src + prepared), length,
                           offset + (off64_t)prepared);
            request->control.data = request.get();
            batch->requests.push_back(std::move(request));
            prepared += length;
        }

        bool submission_ok = true;
        for (auto &owned_request : batch->requests) {
            struct iocb *control = &owned_request->control;
            int retries = 0;
            while (true) {
                {
                    std::lock_guard<std::mutex> lock(batch->mutex);
                    ++batch->remaining;
                }
                direct_aio_outstanding.fetch_add(1,
                                                  std::memory_order_release);
                int submitted = io_submit(direct_aio_context, 1, &control);
                if (submitted == 1) {
                    direct_aio_subrequests.fetch_add(
                        1, std::memory_order_relaxed);
                    break;
                }
                direct_aio_outstanding.fetch_sub(1,
                                                  std::memory_order_release);
                {
                    std::lock_guard<std::mutex> lock(batch->mutex);
                    --batch->remaining;
                }
                if ((submitted == -EAGAIN || submitted == 0) &&
                    retries++ < 10000) {
                    std::this_thread::yield();
                    continue;
                }
                if (submitted < 0) errno = -submitted;
                submission_ok = false;
                break;
            }
            if (!submission_ok) break;
        }

        if (submission_ok) {
            std::lock_guard<std::mutex> lock(direct_async_batches_mutex);
            direct_async_batches.push_back(batch);
        }

        bool return_buffer = false;
        bool batch_completed = false;
        {
            std::lock_guard<std::mutex> lock(batch->mutex);
            batch->submission_finished = true;
            if (!submission_ok) batch->success = false;
            if (batch->remaining == 0) {
                batch->complete = true;
                batch_completed = true;
                return_buffer = batch->success &&
                                !batch->buffer_return_scheduled;
                if (return_buffer) batch->buffer_return_scheduled = true;
            }
        }
        if (return_buffer) {
            direct_bytes.fetch_add(batch->direct_bytes,
                                   std::memory_order_relaxed);
            dram.put_add(batch->base_buffer, 0);
            {
                std::lock_guard<std::mutex> lock(batch->mutex);
                batch->buffer_returned = true;
            }
        }
        if (batch_completed) batch->ready.notify_all();

        if (!submission_ok) {
            std::unique_lock<std::mutex> lock(batch->mutex);
            batch->ready.wait(lock, [&batch] { return batch->complete; });
            size_t partial_bytes = batch->completed_bytes;
            lock.unlock();
            if (partial_bytes > 0) fdatasync(fd_direct);
            return false;
        }
        return true;
    }

    void wait_async_direct(int parall_iter) {
        std::vector<std::shared_ptr<DirectAioBatch>> selected;
        {
            std::lock_guard<std::mutex> lock(direct_async_batches_mutex);
            auto it = direct_async_batches.begin();
            while (it != direct_async_batches.end()) {
                if (parall_iter < 0 || (*it)->parall_iter == parall_iter) {
                    selected.push_back(*it);
                    it = direct_async_batches.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (auto &batch : selected) {
            std::unique_lock<std::mutex> lock(batch->mutex);
            batch->ready.wait(lock, [&batch] {
                return batch->complete &&
                       (!batch->success || batch->buffer_returned);
            });
            bool success = batch->success;
            size_t partial_bytes = batch->completed_bytes;
            lock.unlock();
            if (success) continue;

            direct_errors.fetch_add(1, std::memory_order_relaxed);
            if (partial_bytes > 0) fdatasync(fd_direct);
            fallback_calls.fetch_add(1, std::memory_order_relaxed);
            buffered_fallback_bytes.fetch_add(batch->full_bytes,
                                              std::memory_order_relaxed);
            if (!write_buffered(batch->full_src, batch->full_bytes,
                                batch->full_offset)) {
                memcpy(batch->mmap_target, batch->full_src,
                       batch->full_bytes);
            }
            dram.put_add(batch->base_buffer, 0);
            {
                std::lock_guard<std::mutex> relock(batch->mutex);
                batch->buffer_returned = true;
            }
        }
    }

    void persist_published_record(int area, int64_t published_counter) {
        published_checkpoint_record record = {
            PUBLISHED_RECORD_MAGIC,
            area,
            published_counter,
            published_record_checksum(area, published_counter),
        };
        memcpy((char *)PR_ADDR + PUBLISHED_RECORD_BYTE_OFFSET,
               &record, sizeof(record));
        if (msync((void *)PR_ADDR, 4096, MS_SYNC) == -1) {
            perror("msync - published checkpoint record");
            exit(1);
        }
    }

    static bool pwrite_all(int out_fd, const char *src, size_t bytes,
                           off64_t offset, size_t *completed = nullptr) {
        size_t done = 0;
        while (done < bytes) {
            ssize_t written = pwrite(out_fd, src + done, bytes - done,
                                     offset + (off64_t)done);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) break;
            done += (size_t)written;
        }
        if (completed != nullptr) *completed = done;
        return done == bytes;
    }

    bool write_buffered(const char *src, size_t bytes, off64_t offset) {
        if (bytes == 0) return true;
        if (pwrite_all(fd, src, bytes, offset)) return true;
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            perror("[WARN] buffered pwrite data region");
        }
        return false;
    }

    // Write one chunk without a CPU bounce buffer.  The unaligned head/tail
    // are each <4 KiB; the checkpoint-sized middle goes straight from the
    // cudaMallocHost buffer through O_DIRECT.  Python arranges the source
    // pointer residue to match the file-offset residue.
    bool write_direct_chunk(const char *src, size_t bytes, off64_t offset) {
        if (fd_direct < 0 || bytes == 0) return false;

        const size_t file_residue = (size_t)offset % DIRECT_IO_ALIGNMENT;
        size_t head = file_residue == 0 ? 0 : DIRECT_IO_ALIGNMENT - file_residue;
        if (head > bytes) head = bytes;
        const size_t remaining = bytes - head;
        const size_t middle = (remaining / DIRECT_IO_ALIGNMENT) * DIRECT_IO_ALIGNMENT;
        const size_t tail = remaining - middle;

        // No full direct-I/O block, or the caller did not provide the matching
        // aligned view.  This is expected for the tiny extra-state stream and
        // remains a bounded buffered fallback.
        if (middle == 0 ||
            ((uintptr_t)(src + head) % DIRECT_IO_ALIGNMENT) != 0 ||
            (((size_t)offset + head) % DIRECT_IO_ALIGNMENT) != 0) {
            return false;
        }

        direct_calls.fetch_add(1, std::memory_order_relaxed);
        if (!write_buffered(src, head, offset)) return false;

        size_t direct_done = 0;
        bool direct_ok;
        if (ensure_direct_aio()) {
            direct_ok = direct_aio_write(src + head, middle,
                                         offset + (off64_t)head,
                                         &direct_done);
            // Submission-layer failure is not a reason to abandon direct I/O.
            // If no byte reached storage, retry through synchronous O_DIRECT.
            if (!direct_ok && direct_done == 0) {
                direct_ok = pwrite_all(fd_direct, src + head, middle,
                                       offset + (off64_t)head, &direct_done);
            }
        } else {
            direct_ok = pwrite_all(fd_direct, src + head, middle,
                                   offset + (off64_t)head, &direct_done);
        }
        if (!direct_ok) {
            direct_errors.fetch_add(1, std::memory_order_relaxed);
            // If the kernel ever reports a partial direct write, order it
            // before the complete buffered rewrite performed by the caller.
            if (direct_done > 0) fdatasync(fd_direct);
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true)) {
                perror("[WARN] O_DIRECT pwrite, falling back to buffered path");
            }
            return false;
        }

        if (!write_buffered(src + head + middle, tail,
                            offset + (off64_t)(head + middle))) {
            direct_errors.fetch_add(1, std::memory_order_relaxed);
            fdatasync(fd_direct);
            return false;
        }

        direct_bytes.fetch_add(middle, std::memory_order_relaxed);
        buffered_boundary_bytes.fetch_add(head + tail,
                                          std::memory_order_relaxed);
        return true;
    }

    bool write_direct_chunk_async(const char *src, float *base_buffer,
                                  char *mmap_target, size_t bytes,
                                  off64_t offset, int parall_iter) {
        if (fd_direct < 0 || bytes == 0 || base_buffer == nullptr) return false;

        const size_t file_residue = (size_t)offset % DIRECT_IO_ALIGNMENT;
        size_t head = file_residue == 0 ? 0 : DIRECT_IO_ALIGNMENT - file_residue;
        if (head > bytes) head = bytes;
        const size_t remaining = bytes - head;
        const size_t middle =
            (remaining / DIRECT_IO_ALIGNMENT) * DIRECT_IO_ALIGNMENT;
        const size_t tail = remaining - middle;
        if (middle == 0 ||
            ((uintptr_t)(src + head) % DIRECT_IO_ALIGNMENT) != 0 ||
            (((size_t)offset + head) % DIRECT_IO_ALIGNMENT) != 0) {
            return false;
        }

        if (!write_buffered(src, head, offset) ||
            !write_buffered(src + head + middle, tail,
                            offset + (off64_t)(head + middle))) {
            return false;
        }
        direct_calls.fetch_add(1, std::memory_order_relaxed);
        if (!direct_aio_submit_async(
                src + head, middle, offset + (off64_t)head,
                base_buffer, src, mmap_target, bytes, offset, parall_iter)) {
            direct_errors.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        buffered_boundary_bytes.fetch_add(head + tail,
                                          std::memory_order_relaxed);
        return true;
    }

    void durable_sync_data() {
        const int sync_fd = fd_direct >= 0 ? fd_direct : fd;
        auto start = std::chrono::steady_clock::now();
        int res;
        do {
            res = fdatasync(sync_fd);
        } while (res < 0 && errno == EINTR);
        auto end = std::chrono::steady_clock::now();
        durable_sync_calls.fetch_add(1, std::memory_order_relaxed);
        durable_sync_ns.fetch_add(
            (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count(),
            std::memory_order_relaxed);
        if (res < 0) {
            perror("fdatasync checkpoint data");
            exit(1);
        }
    }

    static bool group_reclaim_enabled() {
        static const bool enabled = [] {
            const char *e = getenv("PCCHECK_GROUP_RECLAIM");
            return e != nullptr && strcmp(e, "1") == 0;
        }();
        return enabled;
    }

    static size_t process_reclaim_budget_bytes() {
        static const size_t budget = [] {
            const char *e = getenv("PCCHECK_RECLAIM_BUDGET_MB");
            long mb = (e != nullptr && e[0] != '\0') ? atol(e) : 4096;
            if (mb < 256) mb = 256;
            return (size_t)mb * 1024 * 1024;
        }();
        return budget;
    }

    void reclaim_segment(const PendingReclaimSegment &seg) {
        if (sync_file_range(fd, seg.start, seg.len,
                            SYNC_FILE_RANGE_WAIT_BEFORE |
                            SYNC_FILE_RANGE_WRITE |
                            SYNC_FILE_RANGE_WAIT_AFTER) == 0) {
            int advise_res = posix_fadvise(fd, seg.start, seg.len,
                                           POSIX_FADV_DONTNEED);
            if (advise_res != 0) {
                static std::atomic<bool> fadv_warned{false};
                if (!fadv_warned.exchange(true)) {
                    errno = advise_res;
                    perror("[WARN] posix_fadvise(DONTNEED)");
                }
            }
        } else {
            static std::atomic<bool> wait_warned{false};
            if (!wait_warned.exchange(true)) {
                perror("[WARN] sync_file_range(WAIT_AFTER)");
            }
        }
    }

    void queue_reclaim_segment(off64_t start, off64_t len, int parall_iter) {
        std::lock_guard<std::mutex> lock(reclaim_mutex);
        reclaim_pending.push_back({start, len, parall_iter});
        reclaim_pending_bytes += (size_t)len;
        const size_t budget = process_reclaim_budget_bytes();
        while (!reclaim_pending.empty() && reclaim_pending_bytes > budget) {
            PendingReclaimSegment oldest = reclaim_pending.front();
            reclaim_pending.pop_front();
            reclaim_segment(oldest);
            reclaim_pending_bytes -= (size_t)oldest.len;
        }
    }

    void retire_reclaim_segments(int parall_iter) {
        if (!group_reclaim_enabled()) return;
        std::lock_guard<std::mutex> lock(reclaim_mutex);
        auto it = reclaim_pending.begin();
        while (it != reclaim_pending.end()) {
            if (it->parall_iter != parall_iter) {
                ++it;
                continue;
            }
            // The merged MS_SYNC for this slot has already completed, so only
            // cache eviction is needed here; no additional writeback wait.
            int advise_res = posix_fadvise(fd, it->start, it->len,
                                           POSIX_FADV_DONTNEED);
            if (advise_res != 0) {
                static std::atomic<bool> retire_warned{false};
                if (!retire_warned.exchange(true)) {
                    errno = advise_res;
                    perror("[WARN] finalize posix_fadvise(DONTNEED)");
                }
            }
            reclaim_pending_bytes -= (size_t)it->len;
            it = reclaim_pending.erase(it);
        }
    }
    
    // ✅ 复用checkpoint基地址计算，避免每次重复计算offset
    float* checkpoint_base_ptr(int parall_iter) {
        if (checkpoint_stride_floats == 0) {
            fprintf(stderr, "[ERROR] checkpoint_stride_floats not initialized! Call init_streams first.\n");
            exit(1);
        }
        return (float *)PR_ADDR_DATA + (parall_iter * checkpoint_stride_floats);
    }
    
public:
    NVM_write() : active_streams(0), multistream_mode(false), total_checkpoint_size(0), 
                  checkpoint_padding_bytes(0), checkpoint_stride_floats(0) {
        // ✅ 简化：移除 C++ 线程池，由 Python ssd_thread_pool 管理并行
    }

    void set_metadata_max_async(int max_async) {
        metadata_max_async = max_async;
    }
    
    // 初始化多个数据流
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
        
        printf("Total checkpoint size: %lu floats (%.2f GB)\n", 
               total_checkpoint_size, checkpoint_bytes / 1e9);
        printf("Checkpoint padding: %lu bytes (%.2f KB), stride: %lu floats\n",
               checkpoint_padding_bytes, checkpoint_padding_bytes / 1024.0, checkpoint_stride_floats);
        
        return 0;
    }
    
    // 写入单个流的数据块到 mmap 区域
    // ✅ 简化设计：使用单线程 memcpy，原因：
    // 1. mmap 写入的瓶颈是磁盘 IO 带宽，不是 CPU
    // 2. Python 端已经有 ssd_thread_pool 实现并行（4个流并行）
    // 3. 多线程 memcpy 对 mmap 几乎没有收益，反而增加同步开销
    void write_stream_chunk(int stream_id, float* data, size_t offset_in_stream, 
                           size_t chunk_size, int parall_iter, int num_threads) {
        if (stream_id < 0 || stream_id >= (int)streams.size()) {
            fprintf(stderr, "[ERROR] Invalid stream_id: %d\n", stream_id);
            return;
        }
        
        if (checkpoint_stride_floats == 0) {
            fprintf(stderr, "[ERROR] checkpoint_stride_floats not initialized! Call init_streams first.\n");
            exit(1);
        }
        
        StreamWriter* stream = streams[stream_id].get();
        
        // 计算目标地址
        float* checkpoint_base = checkpoint_base_ptr(parall_iter);
        float* target_addr = checkpoint_base + stream->stream_offset + offset_in_stream;
        
        size_t chunk_bytes = chunk_size * sizeof(float);

        // Bulk data defaults to O_DIRECT.  Unlike buffered pwrite, it does not
        // duplicate the pinned checkpoint in cgroup-accounted page cache.  A
        // normal descriptor remains for sub-page boundaries and compatibility.
        static const bool data_pwrite = [] {
            const char *e = getenv("PCCHECK_DATA_PWRITE");
            return e == nullptr || strcmp(e, "0") != 0;
        }();
        const off64_t file_offset =
            (off64_t)((char *)target_addr - (char *)PR_ADDR);
        bool wrote_via_direct = write_direct_chunk(
            (const char *)data, chunk_bytes, file_offset);
        bool wrote_via_fd = wrote_via_direct;
        if (!wrote_via_direct) {
            fallback_calls.fetch_add(1, std::memory_order_relaxed);
            buffered_fallback_bytes.fetch_add(chunk_bytes,
                                              std::memory_order_relaxed);
            if (data_pwrite) {
                wrote_via_fd = write_buffered(
                    (const char *)data, chunk_bytes, file_offset);
            }
            if (!wrote_via_fd) {
                memcpy(target_addr, data, chunk_bytes);
            }
        }

        // ✅ 逐组增量刷盘（group-granular durability 第一阶段）：
        // 旧行为把全部落盘推迟到 finalize 的一次性 msync（实测占 checkpoint
        // 时长 ~70%）。这里在每个 chunk memcpy 后立即用
        // sync_file_range(SYNC_FILE_RANGE_WRITE) 异步启动该区间的内核回写，
        // 让设备写入与后续组的更新/拷贝重叠；finalize 的 msync(MS_SYNC)
        // 仍是唯一耐久点与发布屏障（一致性语义不变），但届时大部分脏页
        // 已回写完成。开关：PCCHECK_GROUP_FLUSH=0 关闭（默认开启）。
        static const bool group_flush = [] {
            const char *e = getenv("PCCHECK_GROUP_FLUSH");
            return e == nullptr || strcmp(e, "0") != 0;
        }();
        // ✅ 逐组页回收（第二阶段，PCCHECK_GROUP_RECLAIM=1 开启）——窗口化回写：
        // 背景：mmap 脏页计入 cgroup memory，慢盘上多 rank 并发大 checkpoint
        // （BLOOM PP4 每次 ~84GB）会顶爆容器 memory.max → cgroup OOM 杀 rank。
        // 第一版实现逐 chunk 同步 WAIT_AFTER，实测把盘打成队列深度=1 的单请求
        // 模式（内核失去合并/重排/深队列），checkpoint span 301→446s 反而劣化。
        // 现改窗口化：每 chunk 只发异步 SYNC_FILE_RANGE_WRITE（内核保持批量
        // 回写），进程共享 FIFO 超过 PCCHECK_RECLAIM_BUDGET_MB（默认 4096）
        // 才对最老段做 WAIT_AFTER + fadvise(DONTNEED) 回收。预算不再随 Python
        // SSD worker 数隐式倍增；finalize 的 msync 仍兜底全量并清理该 slot。
        if (group_flush && !wrote_via_direct) {
            const size_t fpage = 4096;
            size_t file_off = (size_t)((char *)target_addr - (char *)PR_ADDR);
            size_t fstart = (file_off / fpage) * fpage;
            size_t fend = ((file_off + chunk_bytes + fpage - 1) / fpage) * fpage;
            if (sync_file_range(fd, (off64_t)fstart, (off64_t)(fend - fstart),
                                SYNC_FILE_RANGE_WRITE) == -1) {
                // 回写失败不致命（最终 msync 仍保证耐久），仅告警一次
                static std::atomic<bool> sfr_warned{false};
                if (!sfr_warned.exchange(true)) {
                    perror("[WARN] sync_file_range");
                }
            } else if (group_reclaim_enabled()) {
                queue_reclaim_segment((off64_t)fstart,
                                      (off64_t)(fend - fstart), parall_iter);
            }
        }
    }

    bool write_stream_chunk_async(int stream_id, float *data,
                                  float *base_buffer,
                                  size_t offset_in_stream,
                                  size_t chunk_size, int parall_iter) {
        if (stream_id < 0 || stream_id >= (int)streams.size() ||
            checkpoint_stride_floats == 0) {
            return false;
        }
        StreamWriter *stream = streams[stream_id].get();
        float *checkpoint_base = checkpoint_base_ptr(parall_iter);
        float *target_addr = checkpoint_base + stream->stream_offset +
                             offset_in_stream;
        size_t chunk_bytes = chunk_size * sizeof(float);
        off64_t file_offset =
            (off64_t)((char *)target_addr - (char *)PR_ADDR);
        return write_direct_chunk_async(
            (const char *)data, base_buffer, (char *)target_addr,
            chunk_bytes, file_offset, parall_iter);
    }

    // 读取单个流的数据块到用户缓冲区
    void read_stream_chunk(int stream_id, float* out, size_t offset_in_stream,
                           size_t chunk_size, int parall_iter) {
        if (stream_id < 0 || stream_id >= (int)streams.size()) {
            fprintf(stderr, "[ERROR] Invalid stream_id for read: %d\n", stream_id);
            return;
        }

        if (checkpoint_stride_floats == 0) {
            fprintf(stderr, "[ERROR] checkpoint_stride_floats not initialized! Call init_streams first.\n");
            exit(1);
        }

        StreamWriter* stream = streams[stream_id].get();
        float* checkpoint_base = checkpoint_base_ptr(parall_iter);
        float* source_addr = checkpoint_base + stream->stream_offset + offset_in_stream;

        size_t chunk_bytes = chunk_size * sizeof(float);
        memcpy(out, source_addr, chunk_bytes);
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
        int res = 0;
        if (fd_direct >= 0) {
            durable_sync_data();
        } else {
            res = msync(aligned_addr, aligned_size, MS_SYNC);
        }
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

        // Python workers only submit O_DIRECT requests.  The slot durability
        // barrier owns completion waiting, so hundreds of pinned chunks may
        // remain queued while training/copy work continues.
        wait_async_direct(parall_iter);
        
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
        
        int msync_res = 0;
        if (fd_direct >= 0) {
            // O_DIRECT completion does not imply power-loss durability.  One
            // fdatasync preserves the old publish contract and also flushes
            // the tiny buffered boundary/extra-state writes on this inode,
            // without walking a checkpoint-sized page-cache mapping.
            durable_sync_data();
        } else {
            msync_res = msync(aligned_addr, aligned_size, MS_SYNC);
        }
        if (msync_res == -1) {
            fprintf(stderr, "[ERROR] Merged msync failed at addr %p (aligned from %p), size %lu (original %lu)\n", 
                   aligned_addr, checkpoint_base, aligned_size, total_sync_bytes);
            perror("msync");
            exit(1);
        }
        retire_reclaim_segments(parall_iter);
        
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        printf("All streams synced (%s) %.2f GB in %.2f ms (%.2f GB/s)\n",
               fd_direct >= 0 ? "fdatasync/O_DIRECT" : "merged msync",
               total_sync_bytes / 1e9, ms_double.count(), 
               (total_sync_bytes / 1e9) / (ms_double.count() / 1000.0));
        
        // 更新检查点元数据（沿用原pccheck的逻辑）
        struct checkpoint *curr_checkpoint = (struct checkpoint *)(PR_ADDR + OFFSET_SIZE * (parall_iter + 3));
        struct checkpoint *volatile last_check = *(struct checkpoint *volatile *)PR_ADDR;
        int64_t curr_counter = curr_checkpoint->counter;
        // 防御：若元数据被破坏（如出现负值或回退），重新生成单调递增的计数
        int64_t last_published = (*(struct checkpoint *volatile *)PR_ADDR)->counter;
        if (curr_counter <= 0 || curr_counter <= last_published)
        {
            // 重新获取一个新的计数并回写元数据，保证单调递增
            curr_counter = counter.fetch_add(1) + 1;
            curr_checkpoint->counter = curr_counter;
        }
        
        // 同步元数据
        int metadata_res = msync(curr_checkpoint, sizeof(struct checkpoint), MS_SYNC);
        if (metadata_res == -1) {
            printf("[ERROR] Proc %d msync during checkpoint sync, with addr %p\n", parall_iter, curr_checkpoint);
            perror("msync");
            exit(1);
        }
        
        // Serialize pointer+relocatable-record publication.  CAS alone orders
        // the process-local pointer, but without this lock an older publisher
        // could overwrite the relocatable record after a newer one wins.
        std::lock_guard<std::mutex> publish_lock(publish_mutex);
        last_check = *(struct checkpoint *volatile *)PR_ADDR;

        // CAS更新检查点指针
        while (true) {
            bool cas_res = __sync_bool_compare_and_swap((checkpoint **)PR_ADDR, last_check, curr_checkpoint);
            struct checkpoint *volatile check = *(struct checkpoint *volatile *)PR_ADDR;
            if (cas_res) {
                printf("CAS was successful! new counter is %lld\n", (long long)check->counter);
                BARRIER(PR_ADDR);
                persist_published_record(parall_iter, check->counter);
                
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
    
    // Error latch: give up an unpublished slot.  Outstanding native AIO for
    // this slot must finish first (it still owns pinned chunks and targets the
    // slot's file range); then the slot goes back to free_space without the
    // durability barrier or the CAS publish, so the previously published
    // checkpoint stays the recovery target.
    void abort_checkpoint(int parall_iter) {
        wait_async_direct(parall_iter);
        retire_reclaim_segments(parall_iter);
        BARRIER(PR_ADDR);
        free_space.enqueue(parall_iter, parall_iter);
        fprintf(stderr, "[WARN] Aborted slot %d: not published, returned to free_space\n",
                parall_iter);
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

    int get_latest_parall_iter() {
        published_checkpoint_record record{};
        memcpy(&record, (char *)PR_ADDR + PUBLISHED_RECORD_BYTE_OFFSET,
               sizeof(record));
        if (record.magic == PUBLISHED_RECORD_MAGIC &&
            record.checksum == published_record_checksum(record.area,
                                                          record.counter) &&
            record.area < 0) {
            // New-format file that never published: registered slots may hold
            // partial data, so there is no recoverable checkpoint.
            return -1;
        }
        if (record.magic == PUBLISHED_RECORD_MAGIC &&
            record.checksum == published_record_checksum(record.area,
                                                          record.counter) &&
            record.area >= 0 && record.area <= metadata_max_async) {
            struct checkpoint *slot = (struct checkpoint *)(
                PR_ADDR + OFFSET_SIZE * (record.area + 3));
            if (slot->area == record.area && slot->counter == record.counter) {
                return (int)record.area;
            }
        }

        // Legacy compatibility: a cleanly closed old file has no in-flight
        // registration, so the largest valid counter is its latest slot.  A
        // crash-time legacy file cannot distinguish registered from published
        // slots because its only commit marker was the process-local pointer.
        int best_area = -1;
        int64_t best_counter = 0;
        for (int area = 0; area <= metadata_max_async; ++area) {
            struct checkpoint *slot = (struct checkpoint *)(
                PR_ADDR + OFFSET_SIZE * (area + 3));
            if (slot->area == area && slot->counter > best_counter) {
                best_area = area;
                best_counter = slot->counter;
            }
        }
        return best_area;
    }

    uint64_t get_io_stat(int stat_id) const {
        switch (stat_id) {
        case 0: return fd_direct >= 0 ? 1 : 0;
        case 1: return direct_bytes.load(std::memory_order_relaxed);
        case 2: return buffered_boundary_bytes.load(std::memory_order_relaxed);
        case 3: return buffered_fallback_bytes.load(std::memory_order_relaxed);
        case 4: return direct_calls.load(std::memory_order_relaxed);
        case 5: return fallback_calls.load(std::memory_order_relaxed);
        case 6: return direct_errors.load(std::memory_order_relaxed);
        case 7: return durable_sync_calls.load(std::memory_order_relaxed);
        case 8: return durable_sync_ns.load(std::memory_order_relaxed);
        case 9: return direct_aio_ready.load(std::memory_order_relaxed) ? 1 : 0;
        case 10: return direct_aio_subrequests.load(std::memory_order_relaxed);
        case 11: return direct_aio_outstanding.load(std::memory_order_relaxed);
        default: return 0;
        }
    }

    void close_writer() {
        // Defensive drain for abnormal callers that close without publishing
        // every registered slot.  Normal finalize paths have already removed
        // their batches above.
        wait_async_direct(-1);
        if (PR_ADDR != nullptr && MAPPED_SIZE > 0) {
            munmap((void *)PR_ADDR, MAPPED_SIZE);
            PR_ADDR = nullptr;
            MAPPED_SIZE = 0;
        }
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
        if (fd_direct >= 0) {
            close(fd_direct);
            fd_direct = -1;
        }
    }
    
    // 清理资源
    ~NVM_write() {
        direct_aio_stop.store(true, std::memory_order_release);
        if (direct_aio_completion_thread.joinable()) {
            direct_aio_completion_thread.join();
        }
        if (direct_aio_context != 0) {
            io_destroy(direct_aio_context);
            direct_aio_context = 0;
        }
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
        int64_t curr_counter = counter.fetch_add((int64_t)1) + 1; // fetch_add 返回旧值，+1 得到新计数

        // 与已发布的最新检查点比较，确保单调递增
        struct checkpoint *volatile published = *(struct checkpoint *volatile *)PR_ADDR;
        int64_t published_counter = published ? published->counter : 0;
        if (curr_counter <= published_counter)
        {
            int64_t expected = curr_counter;
            int64_t desired = published_counter + 1;
            while (true)
            {
                if (counter.compare_exchange_weak(expected, desired))
                {
                    curr_counter = desired;
                    break;
                }
                // counter 已被其他线程更新，再次校验
                if (expected > published_counter)
                {
                    curr_counter = expected;
                    break;
                }
                desired = expected + 1;
            }
        }
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
        int64_t counter_num = checkp_info_new->counter;
        struct checkpoint *volatile last_check = *(struct checkpoint *volatile *)PR_ADDR;
        int64_t curr_counter = counter_num;
        // int parall_iter = parallel_iteration;

    printf("At savenvmNew, tid is %zu, parall_iter is %d, num_threads is %d, last counter is %lld, curr_counter is %lld\n", tid, parall_iter, num_threads, (long long)last_check->counter, (long long)curr_counter);
        if (last_check->counter > curr_counter)
        { // Room for optimization
            if (last_batch)
            {
                printf("At savenvmNew, tid is %zu, parall_iter is %d, num_threads is %d, last counter is %lld, curr_counter is %lld\n", tid, parall_iter, num_threads, (long long)last_check->counter, (long long)curr_counter);
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
                    printf("CAS was successful! new counter is %lld, is_distributed is %d\n", (long long)check->counter, is_distributed);
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
        nvmobj->set_metadata_max_async(max_async);
        // printf("%s\n", filename);
        // ✅ 内存优化：对于多流pccheck，num_batches应该至少为max_async + 1
        // 这样可以确保DRAMAlloc有足够的pinned buffer槽位供并行检查点复用
        // 如果传入的num_batches小于max_async + 1，则使用max_async + 1
        // size_t actual_num_batches = (num_batches < (size_t)(max_async + 1)) ? (max_async + 1) : num_batches;
        initialize(filename, max_async, batch_size_floats, num_batches, dist, dist_rank, world_size);
        return nvmobj;
    }

    NVM_write *reader(const char *filename, int max_async)
    {
        NVM_write *nvmobj = new NVM_write();
        nvmobj->set_metadata_max_async(max_async);
        initialize_for_read(filename, max_async);
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
        if (t == nullptr) {
            fprintf(stderr, "[ERROR] init_streams called with null handle\n");
            return -1;
        }
        return t->init_streams(num_streams, stream_sizes);
    }
    
    // 写入单个流的数据块
    void write_stream_chunk(NVM_write *t, int stream_id, float *data, 
                           size_t offset_in_stream, size_t chunk_size, 
                           int parall_iter, int num_threads) {
        if (t == nullptr) {
            fprintf(stderr, "[ERROR] write_stream_chunk called with null handle\n");
            return;
        }
        t->write_stream_chunk(stream_id, data, offset_in_stream, chunk_size, 
                             parall_iter, num_threads);
    }

    bool write_stream_chunk_async(NVM_write *t, int stream_id, float *data,
                                  float *base_buffer,
                                  size_t offset_in_stream,
                                  size_t chunk_size, int parall_iter) {
        if (t == nullptr) return false;
        return t->write_stream_chunk_async(
            stream_id, data, base_buffer, offset_in_stream,
            chunk_size, parall_iter);
    }

    // 读取单个流的数据块
    void read_stream_chunk(NVM_write *t, int stream_id, float *out,
                           size_t offset_in_stream, size_t chunk_size,
                           int parall_iter) {
        if (t == nullptr) {
            fprintf(stderr, "[ERROR] read_stream_chunk called with null handle\n");
            return;
        }
        t->read_stream_chunk(stream_id, out, offset_in_stream, chunk_size, parall_iter);
    }
    
    // 同步单个流到磁盘
    void sync_stream(NVM_write *t, int stream_id, int parall_iter) {
        if (t == nullptr) {
            fprintf(stderr, "[ERROR] sync_stream called with null handle\n");
            return;
        }
        t->sync_stream(stream_id, parall_iter);
    }
    
    // 同步所有流到磁盘（沿用原pccheck的槽位管理和元数据更新）
    void sync_all_streams(NVM_write *t, int parall_iter) {
        if (t == nullptr) {
            fprintf(stderr, "[ERROR] sync_all_streams called with null handle\n");
            return;
        }
        t->sync_all_streams(parall_iter);
    }

    void abort_checkpoint(NVM_write *t, int parall_iter) {
        if (t == nullptr) {
            fprintf(stderr, "[ERROR] abort_checkpoint called with null handle\n");
            return;
        }
        t->abort_checkpoint(parall_iter);
    }
    
    // ✅ DRAM槽位管理接口（用于多流pccheck内存复用）
    // 借用一块pinned buffer
    float* borrow_chunk(NVM_write *t) {
        if (t == nullptr) {
            fprintf(stderr, "[ERROR] borrow_chunk called with null handle\n");
            return nullptr;
        }
        return t->borrow_chunk();
    }
    
    // 归还pinned buffer
    void return_chunk(NVM_write *t, float* buffer) {
        if (t == nullptr) {
            fprintf(stderr, "[ERROR] return_chunk called with null handle\n");
            return;
        }
        t->return_chunk(buffer);
    }

    // 兼容旧接口
    float* borrow_cpu_slot(NVM_write *t, int parall_iter) {
        if (t == nullptr) {
            fprintf(stderr, "[ERROR] borrow_cpu_slot called with null handle\n");
            return nullptr;
        }
        return t->borrow_cpu_slot(parall_iter);
    }
    
    void return_cpu_slot(NVM_write *t, int parall_iter) {
        if (t == nullptr) {
            fprintf(stderr, "[ERROR] return_cpu_slot called with null handle\n");
            return;
        }
        t->return_cpu_slot(parall_iter);
    }

    int get_latest_parall_iter(NVM_write *t) {
        if (t == nullptr) {
            fprintf(stderr, "[ERROR] get_latest_parall_iter called with null handle\n");
            return -1;
        }
        return t->get_latest_parall_iter();
    }

    uint64_t get_io_stat(NVM_write *t, int stat_id) {
        if (t == nullptr) return 0;
        return t->get_io_stat(stat_id);
    }

    void close_writer(NVM_write *t) {
        if (t == nullptr) return;
        t->close_writer();
        delete t;
    }
}

int main(int argc, char **argv)
{
    return 0;
}
