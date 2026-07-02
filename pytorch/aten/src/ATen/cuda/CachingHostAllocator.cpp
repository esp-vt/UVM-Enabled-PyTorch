#include <ATen/cuda/CachingHostAllocator.h>

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAAllocatorConfig.h>

#include <cuda_runtime_api.h>

namespace at::cuda {
namespace {

// Note: cudaEventCreate when concurrently invoked from multiple threads can be
// very expensive (at least on certain device/driver combinations). Thus, we a)
// serialize event creation at a per-device level, and b) pool the events to
// avoid constantly calling cudaEventCreate/cudaEventDestroy. This results in
// significant improvements in multithreaded workloads with high allocation
// rates.
class EventPool {
 public:
  using Event = std::unique_ptr<
      at::cuda::CUDAEvent,
      std::function<void(at::cuda::CUDAEvent*)>>;
  EventPool() : pools_(at::cuda::device_count()) {}

  Event get(DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<DeviceIndex>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](at::cuda::CUDAEvent* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<at::cuda::CUDAEvent>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    return Event(
        std::make_unique<at::cuda::CUDAEvent>(cudaEventDisableTiming).release(),
        destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<at::cuda::CUDAEvent>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

using Block = HostBlock<CUDAStream>;

struct CUDACachingHostAllocatorImpl
    : public CachingHostAllocatorImpl<CUDAStream, EventPool::Event> {
 private:
  // Track which allocation method was used for each pointer
  // true = cudaMallocManaged, false = cudaHostAlloc
  ska::flat_hash_map<void*, bool> use_managed_memory;

  void allocate_host_memory(size_t size, void** ptr) override {
    // try allocating from reserve segment first before calling into expensive APIs
    if (get_reserve_segment().initialized()) {
      *ptr = get_reserve_segment().allocate(size);
      if (*ptr != nullptr) {
        return;
      }
    }
    allocate_host_memory_slowpath(size, ptr);
  }

  void allocate_host_memory_slowpath(size_t size, void** ptr) {
    // Pinned memory pointers allocated by any device can be directly used by
    // any other device, regardless of the current device at the time of
    // allocation, since we assume unified addressing. So we grab any existing
    // primary context, if available. See pytorch/pytorch#21081.
    // This can be a large performance hit if we cross NUMA nodes by allocating
    // and pinning memory on one side of the NUMA node and then using it on the
    // other side. Thankfully, we use one process per GPU, so we don't run into
    // this issue.
    at::OptionalDeviceGuard device_guard;
    auto primary_ctx_device_index =
        c10::cuda::getDeviceIndexWithPrimaryContext();
    if (primary_ctx_device_index.has_value()) {
      device_guard.reset_device(
          at::Device(at::DeviceType::CUDA, *primary_ctx_device_index));
    }

    auto start = std::chrono::steady_clock::now();
    bool use_managed = c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::pinned_use_cuda_malloc_managed();
    if (use_managed) {
      // Use cudaMallocManaged for allocating managed/unified memory
      C10_CUDA_CHECK(cudaMallocManaged(ptr, size));
    } else {
      // Use cudaHostAlloc for allocating pinned memory (default)
      C10_CUDA_CHECK(cudaHostAlloc(ptr, size, cudaHostAllocDefault));
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update the statistics and track which allocation method was used
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      use_managed_memory[*ptr] = use_managed;
      stats_.host_alloc_time.increase(duration.count());
    }
  }

  void free_block(Block* block) override {
    // We never free blocks from the reserve segment
    if (get_reserve_segment().initialized()) {
      // Check if the block is from the reserve segment
      if (get_reserve_segment().owns(block->ptr_)) {
        return;
      }
    }

    free_block_slowpath(block);
  }

  void free_block_slowpath(Block* block) {
    auto start = std::chrono::steady_clock::now();
    void* ptr = block->ptr_;

    // Check which allocation method was used for this pointer
    bool use_managed = false;
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      auto it = use_managed_memory.find(ptr);
      if (it != use_managed_memory.end()) {
        use_managed = it->second;
      }
    }

    if (use_managed) {
      // Free managed memory
      AT_CUDA_CHECK(cudaFree(ptr));
    } else {
      // Free pinned memory
      AT_CUDA_CHECK(cudaFreeHost(ptr));
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update the statistics and remove tracking entry
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      use_managed_memory.erase(ptr);
      stats_.host_free_time.increase(duration.count());
    }
  }

  void record_stream(
      std::optional<std::vector<EventPool::Event>>& events,
      CUDAStream stream) override {
    auto event = create_event_internal(stream.device_index());
    event->record(stream);
    events->push_back(std::move(event));
  }

  bool query_event(EventPool::Event& event) override {
    cudaError_t err = cudaEventQuery(*event);
    if (err == cudaErrorNotReady) {
      (void)cudaGetLastError(); // clear CUDA error
      return false;
    } else if (err != cudaSuccess) {
      C10_CUDA_CHECK(err);
    }
    return true;
  }

  EventPool::Event create_event_internal(DeviceIndex idx) {
    // Leak the event pool to avoid shutdown issue.
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
  }

  PinnedReserveSegment& get_reserve_segment() {
    static auto reserve_segment = [&]() {
      if (c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::pinned_reserve_segment_size_mb() > 0) {
        void *ptr;
        size_t sz = c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::pinned_reserve_segment_size_mb() * 1024 * 1024;
        allocate_host_memory_slowpath(sz, &ptr);
        return PinnedReserveSegment(ptr, sz);
      } else {
        return PinnedReserveSegment();
      }
    } ();
    return reserve_segment;
  }

};

DECLARE_HOST_ALLOCATOR(
    CUDACachingHostAllocator,
    CUDACachingHostAllocatorImpl,
    raw_local_deleter,
    caching_host_allocator)

REGISTER_HOST_ALLOCATOR(at::kCUDA, &caching_host_allocator)

} // anonymous namespace
} // namespace at::cuda
