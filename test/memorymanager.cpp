/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <af/memory.h>

#include <unordered_map>
#include <utility>
#include <cstdio>

namespace {
/**
 * An implementation of the C++ API's af::MemoryManagerBase to test a
 * custom memory manager. We only test functions which are called by user-facing
 * API functions. This is obviously NOT a fully-featured memory manager and
 * lacks basic features, like thread safety.
 */
#define MB(x) ((size_t)(x) << 20)
class TestMemoryManager : public af::MemoryManagerBase
{
  // Again, this is a test-implementation of a memory manager which blatently
  // does several things inefficiently for the purpose of testing that
  // public APIs that manipulate memory are working properly and are correctly
  // calling the custom implementation.
  // For the purposes of this allocator, for instance, we ignore the
  // "locked" memory abstraction and treat all memory identically.
  using Cache = std::unordered_map<void*, int>;
  Cache cache_; // naive cache, for testing
  size_t allocatedBytes_{0};
  size_t maxMemorySize_{0};
  unsigned maxBuffers_{256};
  // We'll ignore step size/just make sure the API works correctly
  size_t stepSize_{8};

public:
  // Some stateful members used for testing
  bool initialized_{false};
  bool shutdown_{false};
  bool garbageCollected_{false};
  bool printedMemInfo_{false};

  TestMemoryManager() = default;
  ~TestMemoryManager() = default;

  void initialize() override
  {
    initialized_ = true;
    this->setMaxMemorySize();
  }

  // Can't be tested directly - will be tested on test shutdown
  void shutdown() override {
    for (auto& entry : cache_) {
      unlock(entry.first, false);
    }
  }

  void setMaxMemorySize() override
  {
    // Set a fake bound, for testing; normally we'd inspect devices
    maxMemorySize_ = MB(1);
  }

  // Used by the OpenCL backend for organizing memory per-device. Ignored
  // for the purpose of this test; use one huge cache referencing memory
  // on multiple devices.
  void addMemoryManagement(int device) override {}
  void removeMemoryManagement(int device) override {}

  void *alloc(const size_t size, bool user_lock) override
  {
    void* ptr = nullptr;
    if (size > 0) {
      if (allocatedBytes_ + size > maxMemorySize_) {
        throw af::exception(
          "Memory manager out of memory",
          __FILE__,
          __LINE__,
          af_err::AF_ERR_NO_MEM
        );
      } else {
        ptr = nativeAlloc(size);
        cache_.emplace(std::make_pair(ptr, size));
        allocatedBytes_ += size;
      }
    }
    return ptr;
  }

  size_t allocated(void *ptr) override
  {
    if (cache_.find(ptr) != cache_.end()) {
      return cache_[ptr];
    } else {
      return 0;
    }
  }

  void unlock(void *ptr, bool user_unlock) override
  {
    auto size = cache_[ptr];
    cache_.erase(ptr);
    nativeFree(ptr);
    allocatedBytes_ -= size;
  }

  void bufferInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                  size_t *lock_bytes,  size_t *lock_buffers) {
    *alloc_bytes = allocatedBytes_;
    *alloc_buffers = cache_.size();
  }

  // noops since we're ignoring locked memory in this implementation
  void userLock(const void *ptr) {}
  void userUnlock(const void *ptr) {}
  bool isUserLocked(const void *ptr) {}

  bool checkMemoryLimit() override
  {
    return (allocatedBytes_ >= maxMemorySize_);
  }

  size_t getMaxBytes() override
  {
    return maxMemorySize_;
  }

  // Used by the JIT only; ignore for testing
  unsigned getMaxBuffers() override
  {
    return maxBuffers_;
  }

  void printInfo(const char *msg, const int device) {
    std::printf("%s\n", msg);
    std::printf("Bytes allocated: %zu | buffers allocated: %zu\n",
                allocatedBytes_,
                cache_.size());
    printedMemInfo_ = true;
  }

  // Ignore - just make sure the API is working
  void garbageCollect() override
  {
    garbageCollected_ = true;    
  }

  size_t getMemStepSize() override
  {
    return stepSize_;
  }
  
  void setMemStepSize(size_t new_step_size) override
  {
    stepSize_ = new_step_size;
  }

  // Return a reference to the cache for inspection
  const Cache& getCache() const {
    return cache_;
  }
};

} // namespace

/**
 * A custom implementation of the C API's af_memory_manager to test a
 * custom memory manager. For simplicity, no caching is implemented,
 * and most operations are defined as noops. The test simply sanity checks
 * methods are dispatched to properly from calls to the outer API.
 */
typedef struct af_memory_manager_impl {
  af_memory_manager memory_manager_base;
  size_t step_size;
  bool garbage_collected;
  size_t total_allocated_amt;
  size_t total_buffers_allocated;
  size_t max_bytes;
  size_t max_buffers;
} af_memory_manager_impl;

void af_memory_manager_initialize(af_memory_manager* base_inst)
{
  af_memory_manager_impl* inst = (af_memory_manager_impl*)base_inst;
  // set some defaults
  inst->step_size = 8;
  inst->garbage_collected = false;
  inst->total_allocated_amt = 0;
  inst->total_buffers_allocated = 0;
  // call another member function
  base_inst->af_memory_manager_set_max_memory_size(base_inst);
}

// noop
void af_memory_manager_shutdown(af_memory_manager*) {}

void af_memory_manager_set_max_memory_size(af_memory_manager* base_inst)
{
  af_memory_manager_impl* inst = (af_memory_manager_impl*)base_inst;
  inst->max_bytes = 256;
  inst->max_buffers = 16;
}

// noop
void af_memory_manager_add_memory_management(af_memory_manager*, int) {}

// noop
void af_memory_manager_remove_memory_management(af_memory_manager*, int) {}

void* af_memory_manager_alloc(
  af_memory_manager* base_inst,
  const size_t size,
  bool user_lock // ignore
) {
  af_memory_manager_impl* inst = (af_memory_manager_impl*)base_inst;
  void* ptr = NULL;
  if (size > 0) {
    ptr = base_inst->af_memory_manager_native_alloc(base_inst, size);
    if (ptr) {
      inst->total_allocated_amt += size;
      inst->total_buffers_allocated++;
    }
  }
  return ptr;
}

size_t af_memory_manager_allocated(af_memory_manager* inst, void* ptr)
{
  // not keeping track of allocation sizes in this trivial example
  return 0;
}

void af_memory_manager_unlock(
  af_memory_manager* inst,
  void* ptr,
  bool user_unlock
) {
  inst->af_memory_manager_native_free(inst, ptr);
}

// noop
void af_memory_manager_buffer_info(af_memory_manager*,
                                   size_t*,
                                   size_t*,
                                   size_t*,
                                   size_t*) {}

// noop
void af_memory_manager_user_lock(af_memory_manager*, const void*) {}

// noop
void af_memory_manager_user_unlock(af_memory_manager*, const void*) {}

// noop
bool af_memory_manager_is_user_locked(af_memory_manager*, const void*)
{
  return false;
}

// noop
bool af_memory_manager_check_memory_limit(af_memory_manager*) {
  return false;
}

size_t af_memory_manager_get_max_bytes(af_memory_manager* base_inst) {
  af_memory_manager_impl* inst = (af_memory_manager_impl*)base_inst;
  return inst->max_bytes;
}

unsigned af_memory_manager_get_max_buffers(af_memory_manager* base_inst)
{
  af_memory_manager_impl* inst = (af_memory_manager_impl*)base_inst;
  return inst->max_buffers;
}

// noop
void af_memory_manager_print_info(af_memory_manager*, const char*, const int) {}

// noop
void af_memory_manager_garbage_collect(af_memory_manager* base_inst)
{
  af_memory_manager_impl* inst = (af_memory_manager_impl*)base_inst;
  inst->garbage_collected = true;
}

size_t af_memory_manager_get_mem_step_size(af_memory_manager* base_inst)
{
  af_memory_manager_impl* inst = (af_memory_manager_impl*)base_inst;  
  return inst->step_size;
}

void af_memory_manager_set_mem_step_size(
  af_memory_manager* base_inst,
  size_t new_step_size
) {
  af_memory_manager_impl* inst = (af_memory_manager_impl*)base_inst;    
  inst->step_size = new_step_size;
}


TEST(Memory, CustomMemoryMangerCApi)
{
  af_memory_manager_impl* manager =
    (af_memory_manager_impl*)malloc(sizeof(af_memory_manager_impl));
  // Hook in function ptrs
  af_memory_manager* manager_base = (af_memory_manager*)manager;
  manager_base->af_memory_manager_initialize = &af_memory_manager_initialize;
  manager_base->af_memory_manager_shutdown = &af_memory_manager_shutdown;
  manager_base->af_memory_manager_set_max_memory_size =
    &af_memory_manager_set_max_memory_size;
  manager_base->af_memory_manager_add_memory_management =
    &af_memory_manager_add_memory_management;
  manager_base->af_memory_manager_remove_memory_management =
    &af_memory_manager_remove_memory_management;
  manager_base->af_memory_manager_alloc = &af_memory_manager_alloc;
  manager_base->af_memory_manager_allocated = &af_memory_manager_allocated;
  manager_base->af_memory_manager_unlock = &af_memory_manager_unlock;
  manager_base->af_memory_manager_buffer_info = &af_memory_manager_buffer_info;
  manager_base->af_memory_manager_user_lock = &af_memory_manager_user_lock;
  manager_base->af_memory_manager_user_unlock = &af_memory_manager_user_unlock;
  manager_base->af_memory_manager_is_user_locked =
    &af_memory_manager_is_user_locked;
  manager_base->af_memory_manager_check_memory_limit =
    &af_memory_manager_check_memory_limit;
  manager_base->af_memory_manager_get_max_bytes =
    &af_memory_manager_get_max_bytes;
  manager_base->af_memory_manager_get_max_buffers =
    &af_memory_manager_get_max_buffers;
  manager_base->af_memory_manager_print_info =
    &af_memory_manager_print_info;
  manager_base->af_memory_manager_garbage_collect =
    &af_memory_manager_garbage_collect;
  manager_base->af_memory_manager_get_mem_step_size =
    &af_memory_manager_get_mem_step_size;
  manager_base->af_memory_manager_set_mem_step_size =
    &af_memory_manager_set_mem_step_size;
  // Set the manager
  af_set_memory_manager(manager_base, AF_C_MEMORY_MANAGER_API);
  // ignore the pinned memory manager

  // For the purposes of our test, use memory with the C++ API
  ASSERT_EQ(manager->total_buffers_allocated, 0);
  {
    auto a = af::randu({2, 2}, af::dtype::f32); // 16 bytes
    ASSERT_EQ(manager->total_allocated_amt, 16);
    auto b = af::randu({4, 4}, af::dtype::f64); // 128 bytes
    ASSERT_EQ(manager->total_allocated_amt, 144);
    ASSERT_EQ(manager->total_buffers_allocated, 2);
  }

  ASSERT_FALSE(manager->garbage_collected);
  af_device_gc();
  ASSERT_TRUE(manager->garbage_collected);

  ASSERT_EQ(manager->step_size, 8);
  af_set_mem_step_size(16);
  ASSERT_EQ(manager->step_size, 16);
  
  // Our manager instance is freed for us internally
}

TEST(Memory, CustomMemoryManagerCppApi)
{
  int backend;
  af_get_available_backends(&backend);
  // Set a pinned manager if using the CUDA or OpenCL backends.
  if (backend == AF_BACKEND_OPENCL || backend == AF_BACKEND_CUDA) {
    TestMemoryManager* pinnedManager = new TestMemoryManager();
    af::setPinnedMemoryManager(pinnedManager);
    ASSERT_TRUE(pinnedManager->initialized_);
  }
  // Construct and set our memory manager implementation
  // DeviceManager will free when global singletons are destroyed
  TestMemoryManager* manager = new TestMemoryManager();
  af::setMemoryManager(manager);
  ASSERT_TRUE(manager->initialized_);
  
  // Test garbage collection (noop)
  ASSERT_FALSE(manager->garbageCollected_);
  af::deviceGC();
  ASSERT_TRUE(manager->garbageCollected_);

  // Test memory step size
  ASSERT_EQ(af::getMemStepSize(), 8); // default for toy implementation is 8
  af::setMemStepSize(16);
  ASSERT_EQ(af::getMemStepSize(), 16);

  auto& cache = manager->getCache();
  {
    // Allocate an array
    auto a = af::randu({2, 2}, af::dtype::f64); // 32 bytes
    size_t aSize = af::getSizeOf(a.type()) * a.elements();
    ASSERT_EQ(cache.size(), 1);
    void* addr = a.device<void>();
    // cl::Buffer locations aren't reliable as per raw device pointers
    // and aren't defined per the spec
    if (backend != AF_BACKEND_OPENCL) {
      ASSERT_EQ(cache.at(addr), aSize);
      ASSERT_EQ(a.allocated(), cache.at(addr));
    }
    size_t allocBytes, allocBuffers;
    af::deviceMemInfo(&allocBytes, &allocBuffers, 0, 0);
    ASSERT_EQ(allocBytes, aSize);
    ASSERT_EQ(allocBuffers, cache.size());
  } // array will be unlocked/freed
  ASSERT_EQ(cache.size(), 0);

  void* a = af::alloc(8, af::dtype::f64); // 64 bytes
  ASSERT_EQ(cache.size(), 1);
  if (backend != AF_BACKEND_OPENCL) {
    ASSERT_EQ(cache.at(a), 8 * 8);
  }
  ASSERT_FALSE(manager->printedMemInfo_);
  af::printMemInfo("Test information:");
  ASSERT_TRUE(manager->printedMemInfo_);
  af::free(a);
  ASSERT_EQ(cache.size(), 0);

  // Overflow behavior
  {
    auto m = af::randu({512, 512}, af::dtype::f32); // allocate 1 megabyte
    auto overflow = []() {
      auto a = af::randu({2, 2});
    };
    ASSERT_THROW(overflow(), af::exception);
  }
  ASSERT_EQ(cache.size(), 0);
}
