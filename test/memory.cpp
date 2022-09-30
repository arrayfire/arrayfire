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
#include <testHelpers.hpp>
#include <af/dim4.hpp>
#include <af/internal.h>
#include <af/memory.h>
#include <af/traits.hpp>

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using af::alloc;
using af::allocV2;
using af::array;
using af::cdouble;
using af::cfloat;
using af::deviceGC;
using af::deviceMemInfo;
using af::dim4;
using af::dtype;
using af::dtype_traits;
using af::freeV2;
using af::randu;
using af::seq;
using af::span;
using std::vector;

const size_t step_bytes = 1024;

TEST(Memory, Scope) {
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate();  // Clean up everything done so far

    {
        array a = randu(5, 5);

        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);

        ASSERT_EQ(alloc_bytes, 1 * step_bytes);
        ASSERT_EQ(lock_bytes, 1 * step_bytes);
    }

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 0u);  // 0 because a is out of scope

    ASSERT_EQ(alloc_bytes, 1 * step_bytes);
    ASSERT_EQ(lock_bytes, 0u);
}

template<typename T>
class MemAlloc : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble, int, unsigned int,
                         intl, uintl, char, unsigned char, short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(MemAlloc, TestTypes);

size_t roundUpToStep(size_t bytes) {
    if (step_bytes == 0) return bytes;

    size_t remainder = bytes % step_bytes;
    if (remainder == 0) return bytes;

    return bytes + step_bytes - remainder;
}

template<typename T>
void memAllocArrayScopeTest(int elements) {
    SUPPORTED_TYPE_CHECK(T);

    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate();  // Clean up everything done so far

    {
        array a = randu(elements, (af_dtype)dtype_traits<T>::af_type);

        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);

        ASSERT_EQ(alloc_bytes, roundUpToStep(elements * sizeof(T)));
        ASSERT_EQ(lock_bytes, roundUpToStep(elements * sizeof(T)));
    }

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 0u);  // 0 because a is out of scope

    ASSERT_EQ(alloc_bytes, roundUpToStep(elements * sizeof(T)));
    ASSERT_EQ(lock_bytes, 0u);
}

template<typename T>
void memAllocPtrScopeTest(int elements) {
    SUPPORTED_TYPE_CHECK(T);

    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate();  // Clean up everything done so far
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        T *ptr = alloc<T>(elements);

        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);

        ASSERT_EQ(alloc_bytes, roundUpToStep(elements * sizeof(T)));
        ASSERT_EQ(lock_bytes, roundUpToStep(elements * sizeof(T)));

        af::free(ptr);
#pragma GCC diagnostic pop
    }

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 0u);  // 0 because a is out of scope

    ASSERT_EQ(alloc_bytes, roundUpToStep(elements * sizeof(T)));
    ASSERT_EQ(lock_bytes, 0u);

    // Do without using templated alloc
    cleanSlate();  // Clean up everything done so far

    {
        void *ptr = allocV2(elements * sizeof(T));

        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);

        ASSERT_EQ(alloc_bytes, roundUpToStep(elements * sizeof(T)));
        ASSERT_EQ(lock_bytes, roundUpToStep(elements * sizeof(T)));

        af::freeV2(ptr);
    }

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 0u);  // 0 because a is out of scope

    ASSERT_EQ(alloc_bytes, roundUpToStep(elements * sizeof(T)));
    ASSERT_EQ(lock_bytes, 0u);
}

TYPED_TEST(MemAlloc, ArrayScope25) { memAllocArrayScopeTest<TypeParam>(25); }

TYPED_TEST(MemAlloc, ArrayScope2048) {
    memAllocArrayScopeTest<TypeParam>(2048);
}

TYPED_TEST(MemAlloc, ArrayScope2293) {
    memAllocArrayScopeTest<TypeParam>(2293);
}

TYPED_TEST(MemAlloc, PtrScope25) { memAllocPtrScopeTest<TypeParam>(25); }

TYPED_TEST(MemAlloc, PtrScope2048) { memAllocPtrScopeTest<TypeParam>(2048); }

TYPED_TEST(MemAlloc, PtrScope2293) { memAllocPtrScopeTest<TypeParam>(2293); }

TEST(Memory, SingleSizeLoop) {
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate();  // Clean up everything done so far

    {
        array a = randu(5, 5);

        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);
        ASSERT_EQ(alloc_bytes, 1 * step_bytes);
        ASSERT_EQ(lock_bytes, 1 * step_bytes);

        for (int i = 0; i < 100; i++) {
            a = randu(5, 5);

            deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes,
                          &lock_buffers);

            ASSERT_EQ(
                alloc_buffers,
                2u);  // 2 because a new one is created before a is destroyed
            ASSERT_EQ(lock_buffers, 1u);
            ASSERT_EQ(alloc_bytes, 2 * step_bytes);
            ASSERT_EQ(lock_bytes, 1 * step_bytes);
        }
    }
}

TEST(Memory, LargeLoop) {
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate();  // Clean up everything done so far

    const int num    = step_bytes / sizeof(float);
    size_t allocated = step_bytes;

    array a = randu(num);

    vector<float> hA(num);

    a.host(&hA[0]);

    // Run a large loop that allocates more and more memory at each step
    for (int i = 0; i < 250; i++) {
        array b        = randu(num * (i + 1));
        size_t current = (i + 1) * step_bytes;
        allocated += current;

        // Verify that new buffers are being allocated
        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        // Limit to 10 to check before memory cleanup
        if (i < 10) {
            ASSERT_EQ(alloc_buffers, (size_t)(i + 2));  // i is zero based
            ASSERT_EQ(lock_buffers, 2u);

            ASSERT_EQ(alloc_bytes, allocated);
            ASSERT_EQ(lock_bytes, current + step_bytes);
        }
    }

    size_t old_alloc_bytes   = alloc_bytes;
    size_t old_alloc_buffers = alloc_buffers;

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(old_alloc_bytes, alloc_bytes);
    ASSERT_EQ(old_alloc_buffers, alloc_buffers);

    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(lock_bytes, 1 * step_bytes);
}

TEST(Memory, IndexingOffset) {
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate();  // Clean up everything done so far

    const int num = step_bytes / sizeof(float);

    array a = randu(num);

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, 1 * step_bytes);
    ASSERT_EQ(lock_bytes, 1 * step_bytes);

    {
        array b = a(seq(1, num / 2));  // Should just be an offset

        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);
        ASSERT_EQ(alloc_bytes, 1 * step_bytes);
        ASSERT_EQ(lock_bytes, 1 * step_bytes);
    }

    // b should not have deleted a
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, 1 * step_bytes);
    ASSERT_EQ(lock_bytes, 1 * step_bytes);
}

TEST(Memory, IndexingCopy) {
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate();  // Clean up everything done so far

    const int num = step_bytes / sizeof(float);

    array a = randu(num);

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, 1 * step_bytes);
    ASSERT_EQ(lock_bytes, 1 * step_bytes);

    {
        // Should just a copy
        array b = a(seq(0, num / 2 - 1, 2));

        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 2u);
        ASSERT_EQ(lock_buffers, 2u);
        ASSERT_EQ(alloc_bytes, 2 * step_bytes);
        ASSERT_EQ(lock_bytes, 2 * step_bytes);
    }

    // b should not have deleted a
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 2u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, 2 * step_bytes);
    ASSERT_EQ(lock_bytes, 1 * step_bytes);
}

TEST(Memory, Assign) {
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate();  // Clean up everything done so far

    const int num = step_bytes / sizeof(float);

    array a = randu(num);

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, 1 * step_bytes);
    ASSERT_EQ(lock_bytes, 1 * step_bytes);

    {
        array b         = randu(num / 2);
        a(seq(num / 2)) = b;

        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 2u);
        ASSERT_EQ(lock_buffers, 2u);
        ASSERT_EQ(alloc_bytes, 2 * step_bytes);
        ASSERT_EQ(lock_bytes, 2 * step_bytes);
    }

    // b should not have deleted a
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 2u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, 2 * step_bytes);
    ASSERT_EQ(lock_bytes, 1 * step_bytes);
}

TEST(Memory, AssignLoop) {
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate();  // Clean up everything done so far

    const int num  = step_bytes / sizeof(float);
    const int cols = 100;

    array a = randu(num, cols);

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, cols * step_bytes);
    ASSERT_EQ(lock_bytes, cols * step_bytes);

    for (int i = 0; i < cols; i++) {
        array b    = randu(num);
        a(span, i) = b;

        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers,
                  2u);  // 3 because you need another scratch space for b
        ASSERT_EQ(lock_buffers, 2u);
        ASSERT_EQ(alloc_bytes, (cols + 1) * step_bytes);
        ASSERT_EQ(lock_bytes, (cols + 1) * step_bytes);
    }
}

TEST(Memory, AssignRef) {
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate();  // Clean up everything done so far

    const int num = step_bytes / sizeof(float);

    array a     = randu(num);
    array a_ref = a;

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, 1 * step_bytes);
    ASSERT_EQ(lock_bytes, 1 * step_bytes);

    {
        array b = randu(num / 2);
        // This should do a full copy of a
        a(seq(num / 2)) = b;

        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 3u);
        ASSERT_EQ(lock_buffers, 3u);
        ASSERT_EQ(alloc_bytes, 3 * step_bytes);
        ASSERT_EQ(lock_bytes, 3 * step_bytes);
    }

    // b should not have deleted a
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 3u);
    ASSERT_EQ(lock_buffers, 2u);  // a_ref
    ASSERT_EQ(alloc_bytes, 3 * step_bytes);
    ASSERT_EQ(lock_bytes, 2 * step_bytes);  // a_ref
}

TEST(Memory, AssignRefLoop) {
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate();  // Clean up everything done so far

    const int num  = step_bytes / sizeof(float);
    const int cols = 100;

    array a     = randu(num, cols);
    array a_ref = a;

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, cols * step_bytes);
    ASSERT_EQ(lock_bytes, cols * step_bytes);

    for (int i = 0; i < cols; i++) {
        array b    = randu(num);
        a(span, i) = b;

        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 3u);
        ASSERT_EQ(lock_buffers, 3u);
        ASSERT_EQ(alloc_bytes, (2 * cols + 1) * step_bytes);
        ASSERT_EQ(lock_bytes, (2 * cols + 1) * step_bytes);
    }

    // b should not have deleted a
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 3u);
    ASSERT_EQ(lock_buffers, 2u);  // a_ref
    ASSERT_EQ(alloc_bytes, (2 * cols + 1) * step_bytes);
    ASSERT_EQ(lock_bytes, 2 * cols * step_bytes);  // a_ref
}

TEST(Memory, device) {
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate();  // Clean up everything done so far

    {
        array a = randu(5, 5);

        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);
        ASSERT_EQ(alloc_bytes, 1 * step_bytes);
        ASSERT_EQ(lock_bytes, 1 * step_bytes);

        a.device<float>();

        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);
        ASSERT_EQ(alloc_bytes, 1 * step_bytes);
        ASSERT_EQ(lock_bytes, 1 * lock_bytes);

        a.unlock();  // to reset the lock flag
    }

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 0u);
    ASSERT_EQ(alloc_bytes, 1 * step_bytes);
    ASSERT_EQ(lock_bytes, 0u);
}

TEST(Memory, Assign2D) {
    size_t alloc_bytes, alloc_buffers;
    size_t alloc_bytes_after, alloc_buffers_after;
    size_t lock_bytes, lock_buffers;
    size_t lock_bytes_after, lock_buffers_after;

    cleanSlate();  // Clean up everything done so far
    {
        array a       = af::randu(10, 10, f32);
        unsigned hb[] = {3, 5, 6, 8, 9};
        array b(5, hb);
        array c = af::randu(5, f32);
        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
        a(b) = c;
    }

    deviceMemInfo(&alloc_bytes_after, &alloc_buffers_after, &lock_bytes_after,
                  &lock_buffers_after);

    // Check if assigned allocated extra buffers
    ASSERT_EQ(alloc_buffers, alloc_buffers_after);
    ASSERT_EQ(alloc_bytes, alloc_bytes_after);
}

TEST(Memory, unlock) {
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate();  // Clean up everything done so far

    const dim_t num = step_bytes / sizeof(float);

    vector<float> in(num);

    af_array arr = 0;
    ASSERT_SUCCESS(af_create_array(&arr, &in[0], 1, &num, f32));

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, step_bytes);
    ASSERT_EQ(lock_bytes, step_bytes);

    // arr1 gets released by end of the following code block
    {
        array a(arr);
        a.lock();

        // No new memory should be allocated
        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);
        ASSERT_EQ(alloc_bytes, step_bytes);
        ASSERT_EQ(lock_bytes, step_bytes);

        a.unlock();
    }

    // Making sure all unlocked buffers are freed
    deviceGC();

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 0u);
    ASSERT_EQ(lock_buffers, 0u);
    ASSERT_EQ(alloc_bytes, 0u);
    ASSERT_EQ(lock_bytes, 0u);
}

TEST(Memory, IndexedDevice) {
    // This test is checking to see if calling .device() will force copy to a
    // new buffer
    const int nx = 8;
    const int ny = 8;

    array in = randu(nx, ny);

    vector<float> in1(in.elements());
    in.host(&in1[0]);

    int offx = nx / 4;
    int offy = ny / 4;

    in = in(seq(offx, offx + nx / 2 - 1), seq(offy, offy + ny / 2 - 1));

    int nxo = (int)in.dims(0);
    int nyo = (int)in.dims(1);

    void *rawPtr = getRawPtr(in);
    void *devPtr = in.device<float>();
    ASSERT_NE(devPtr, rawPtr);
    in.unlock();

    vector<float> in2(in.elements());
    in.host(&in2[0]);

    for (int y = 0; y < nyo; y++) {
        for (int x = 0; x < nxo; x++) {
            ASSERT_EQ(in1[(offy + y) * nx + offx + x], in2[y * nxo + x]);
        }
    }
}

namespace {

template<typename T>
T *getMemoryManagerPayload(af_memory_manager manager) {
    void *payloadPtr;
    af_memory_manager_get_payload(manager, &payloadPtr);
    return (T *)payloadPtr;
}

/**
 * An extremely basic memory manager with a basic caching mechanism for testing
 * purposes. It is not thread safe or optimized.
 */
struct E2ETestPayload {
    int initializeCalledTimes{0};
    int shutdownCalledTimes{0};
    std::unordered_map<void *, size_t> table;
    std::unordered_set<void *> locked;
    size_t totalBytes{0};
    size_t totalBuffers{0};
    size_t lockedBytes{0};
    unsigned lastNdims;
    dim4 lastDims;
    unsigned lastElementSize;

    size_t maxBuffers{64};
    size_t maxBytes{1024};
    // Print info args
    std::string printInfoStringArg;
    int printInfoDevice{-1};
};

af_err allocated_fn(af_memory_manager manager, size_t *out, void *ptr) {
    auto &table = getMemoryManagerPayload<E2ETestPayload>(manager)->table;
    if (table.find(ptr) == table.end()) {
        *out = 0;
    } else {
        *out = table[ptr];
    }
    return AF_SUCCESS;
}

af_err user_lock_fn(af_memory_manager manager, void *ptr) {
    auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
    if (payload->locked.find(ptr) == payload->locked.end()) {
        payload->locked.insert(ptr);
        payload->lockedBytes += payload->table[ptr];
    }
    return AF_SUCCESS;
}

af_err is_user_locked_fn(af_memory_manager manager, int *out, void *ptr) {
    auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
    *out          = payload->locked.find(ptr) != payload->locked.end();
    return AF_SUCCESS;
}

af_err unlock_fn(af_memory_manager manager, void *ptr, int userLock) {
    if (!ptr) { return AF_SUCCESS; }

    auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);

    if (payload->table.find(ptr) == payload->table.end()) {
        return AF_SUCCESS;  // fast path
    }

    // For testing, treat user-allocated and AF-allocated memory identically
    if (payload->locked.find(ptr) != payload->locked.end()) {
        payload->locked.erase(ptr);
        payload->lockedBytes -= payload->table[ptr];
    }
    return AF_SUCCESS;
}

af_err user_unlock_fn(af_memory_manager manager, void *ptr) {
    auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
    af_err err    = unlock_fn(manager, ptr, /* user */ 1);
    payload->lockedBytes -= payload->table[ptr];
    return err;
}

af_err signal_memory_cleanup_fn(af_memory_manager manager) {
    auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
    // Free unlocked memory
    std::vector<void *> freed;
    for (auto &entry : payload->table) {
        int isUserLocked;
        is_user_locked_fn(manager, &isUserLocked, entry.first);
        if (!isUserLocked) {
            void *ptr = entry.first;
            af_memory_manager_native_free(manager, ptr);
            payload->totalBytes -= payload->table[entry.first];
            freed.push_back(entry.first);
        }
    }
    for (auto ptr : freed) { payload->table.erase(ptr); }
    return AF_SUCCESS;
}

af_err print_info_fn(af_memory_manager manager, char *c, int b) {
    auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
    payload->printInfoStringArg = std::string(c);
    payload->printInfoDevice    = b;
    return AF_SUCCESS;
}

af_err get_memory_pressure_fn(af_memory_manager manager, float *out) {
    auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
    if (payload->lockedBytes > payload->maxBytes ||
        payload->totalBuffers > payload->maxBuffers) {
        *out = 1.0;
    } else {
        *out = 0.0;
    }
    return AF_SUCCESS;
}

af_err jit_tree_exceeds_memory_pressure_fn(af_memory_manager manager, int *out,
                                           size_t bytes) {
    auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
    *out          = 2 * bytes > payload->totalBytes;
    return AF_SUCCESS;
}

af_err alloc_fn(af_memory_manager manager, void **ptr,
                /* bool */ int userLock, const unsigned ndims, dim_t *dims,
                const unsigned element_size) {
    size_t size = element_size;
    for (unsigned i = 0; i < ndims; ++i) { size *= dims[i]; }

    if (size > 0) {
        float pressure;
        get_memory_pressure_fn(manager, &pressure);
        float threshold;
        af_memory_manager_get_memory_pressure_threshold(manager, &threshold);
        if (pressure >= threshold) { signal_memory_cleanup_fn(manager); }

        if (af_err err = af_memory_manager_native_alloc(manager, ptr, size)) {
            return err;
        }

        auto *payload        = getMemoryManagerPayload<E2ETestPayload>(manager);
        payload->table[*ptr] = size;
        payload->totalBytes += size;
        payload->totalBuffers++;

        // Simple implementation: treat user and AF allocations the same
        payload->locked.insert(*ptr);
        payload->lockedBytes += size;

        payload->lastNdims       = ndims;
        payload->lastDims        = dim4(ndims, dims);
        payload->lastElementSize = element_size;
    }

    return AF_SUCCESS;
}

void add_memory_management_fn(af_memory_manager manager, int id) {}

void remove_memory_management_fn(af_memory_manager manager, int id) {}

}  // namespace

class MemoryManagerApi : public ::testing::Test {
   public:
    af_memory_manager manager;
    std::unique_ptr<E2ETestPayload> payload{new E2ETestPayload()};
    void SetUp() override {
        af_create_memory_manager(&manager);

        // Set payload_fn
        af_memory_manager_set_payload(manager, payload.get());

        auto initialize_fn = [](af_memory_manager manager) {
            auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
            payload->initializeCalledTimes++;
            return AF_SUCCESS;
        };
        af_memory_manager_set_initialize_fn(manager, initialize_fn);

        auto shutdown_fn = [](af_memory_manager manager) {
            auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
            payload->shutdownCalledTimes++;
            return AF_SUCCESS;
        };
        af_memory_manager_set_shutdown_fn(manager, shutdown_fn);

        // alloc
        af_memory_manager_set_alloc_fn(manager, alloc_fn);
        af_memory_manager_set_allocated_fn(manager, allocated_fn);
        af_memory_manager_set_unlock_fn(manager, unlock_fn);
        // utils
        af_memory_manager_set_signal_memory_cleanup_fn(
            manager, signal_memory_cleanup_fn);
        af_memory_manager_set_print_info_fn(manager, print_info_fn);
        // user lock/unlock
        af_memory_manager_set_user_lock_fn(manager, user_lock_fn);
        af_memory_manager_set_user_unlock_fn(manager, user_unlock_fn);
        af_memory_manager_set_is_user_locked_fn(manager, is_user_locked_fn);
        // memory pressure
        af_memory_manager_set_get_memory_pressure_fn(manager,
                                                     get_memory_pressure_fn);
        af_memory_manager_set_jit_tree_exceeds_memory_pressure_fn(
            manager, jit_tree_exceeds_memory_pressure_fn);
        // ocl
        af_memory_manager_set_add_memory_management_fn(
            manager, add_memory_management_fn);
        af_memory_manager_set_remove_memory_management_fn(
            manager, remove_memory_management_fn);

        af_set_memory_manager(manager);
    }

    void TearDown() override {
        af_device_gc();
        af_unset_memory_manager();
        af_release_memory_manager(manager);
    }
};

TEST_F(MemoryManagerApi, E2ETest1D) {
    size_t aSize = 8;

    array a = af::array(aSize, af::dtype::f32);
    ASSERT_EQ(payload->table.size(), 1);

    ASSERT_EQ(payload->table[a.device<float>()], aSize * sizeof(float));
    ASSERT_EQ(payload->lastNdims, 1);
    ASSERT_EQ(payload->lastDims, af::dim4(aSize));
    ASSERT_EQ(payload->lastElementSize, 4);
}

TEST_F(MemoryManagerApi, E2ETest2D) {
    size_t aSize = 8;

    af::array a = af::array(aSize, aSize, af::dtype::f32);
    ASSERT_EQ(payload->table.size(), 1);
    ASSERT_EQ(payload->table[a.device<float>()], aSize * aSize * sizeof(float));
    ASSERT_EQ(payload->lastElementSize, 4);

    // Currently this is set to 1 because all allocations request linear memory
    // This behavior will change in the future
    ASSERT_EQ(payload->lastNdims, 1);
    ASSERT_EQ(payload->lastDims, af::dim4(aSize * aSize));
}

TEST_F(MemoryManagerApi, E2ETest3D) {
    size_t aSize = 8;

    af::array a = af::array(aSize, aSize, aSize, af::dtype::f32);
    ASSERT_EQ(payload->table.size(), 1);
    ASSERT_EQ(payload->table[a.device<float>()],
              aSize * aSize * aSize * sizeof(float));
    ASSERT_EQ(payload->lastElementSize, 4);

    // Currently this is set to 1 because all allocations request linear memory
    // This behavior will change in the future
    ASSERT_EQ(payload->lastNdims, 1);
    ASSERT_EQ(payload->lastDims, af::dim4(aSize * aSize * aSize));
}

TEST_F(MemoryManagerApi, E2ETest4D) {
    size_t aSize = 8;

    af::array a = af::array(aSize, aSize, aSize, aSize, af::dtype::f32);
    ASSERT_EQ(payload->table.size(), 1);
    ASSERT_EQ(payload->table[a.device<float>()],
              aSize * aSize * aSize * aSize * sizeof(float));
    ASSERT_EQ(payload->lastElementSize, 4);

    // Currently this is set to 1 because all allocations request linear memory
    // This behavior will change in the future
    ASSERT_EQ(payload->lastNdims, 1);
    ASSERT_EQ(payload->lastDims, af::dim4(aSize * aSize * aSize * aSize));
    af::sync();
}

TEST_F(MemoryManagerApi, E2ETest4DComplexDouble) {
    size_t aSize = 8;

    af::array a = af::array(aSize, aSize, aSize, aSize, af::dtype::c64);
    ASSERT_EQ(payload->table.size(), 1);
    ASSERT_EQ(payload->table[a.device<float>()],
              aSize * aSize * aSize * aSize * sizeof(double) * 2);
    ASSERT_EQ(payload->lastElementSize, 16);

    // Currently this is set to 1 because all allocations request linear memory
    // This behavior will change in the future
    ASSERT_EQ(payload->lastNdims, 1);
    ASSERT_EQ(payload->lastDims, af::dim4(aSize * aSize * aSize * aSize));
}

TEST_F(MemoryManagerApi, E2ETestMultipleAllocations) {
    size_t aSize = 8;

    af::array a = af::array(aSize, af::dtype::c64);
    ASSERT_EQ(payload->lastElementSize, 16);

    af::array b = af::array(aSize, af::dtype::f64);
    ASSERT_EQ(payload->lastElementSize, 8);

    ASSERT_EQ(payload->table.size(), 2);
    ASSERT_EQ(payload->table[a.device<float>()], aSize * sizeof(double) * 2);
    ASSERT_EQ(payload->table[b.device<float>()], aSize * sizeof(double));

    // Currently this is set to 1 because all allocations request linear memory
    // This behavior will change in the future
    ASSERT_EQ(payload->lastNdims, 1);
    ASSERT_EQ(payload->lastDims, af::dim4(aSize));
}

TEST_F(MemoryManagerApi, OutOfMemory) {
    af::array a;
    const unsigned N = 99999;
    try {
        a = af::randu({N, N, N}, af::dtype::f32);
        FAIL();
    } catch (af::exception &ex) {
        ASSERT_EQ(ex.err(), AF_ERR_NO_MEM);
    } catch (...) { FAIL(); }
}

TEST(MemoryManagerE2E, E2ETest) {
    af_memory_manager manager;
    af_create_memory_manager(&manager);

    // Set payload_fn
    std::unique_ptr<E2ETestPayload> payload(new E2ETestPayload());
    af_memory_manager_set_payload(manager, payload.get());

    auto initialize_fn = [](af_memory_manager manager) {
        auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
        payload->initializeCalledTimes++;
        return AF_SUCCESS;
    };
    af_memory_manager_set_initialize_fn(manager, initialize_fn);

    auto shutdown_fn = [](af_memory_manager manager) {
        auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
        payload->shutdownCalledTimes++;
        return AF_SUCCESS;
    };
    af_memory_manager_set_shutdown_fn(manager, shutdown_fn);

    // alloc
    af_memory_manager_set_alloc_fn(manager, alloc_fn);
    af_memory_manager_set_allocated_fn(manager, allocated_fn);
    af_memory_manager_set_unlock_fn(manager, unlock_fn);
    // utils
    af_memory_manager_set_signal_memory_cleanup_fn(manager,
                                                   signal_memory_cleanup_fn);
    af_memory_manager_set_print_info_fn(manager, print_info_fn);
    // user lock/unlock
    af_memory_manager_set_user_lock_fn(manager, user_lock_fn);
    af_memory_manager_set_user_unlock_fn(manager, user_unlock_fn);
    af_memory_manager_set_is_user_locked_fn(manager, is_user_locked_fn);
    // memory pressure
    af_memory_manager_set_get_memory_pressure_fn(manager,
                                                 get_memory_pressure_fn);
    af_memory_manager_set_jit_tree_exceeds_memory_pressure_fn(
        manager, jit_tree_exceeds_memory_pressure_fn);
    // ocl
    af_memory_manager_set_add_memory_management_fn(manager,
                                                   add_memory_management_fn);
    af_memory_manager_set_remove_memory_management_fn(
        manager, remove_memory_management_fn);

    af_set_memory_manager(manager);
    {
        size_t aSize = 8;

        void *a = af::allocV2(aSize * sizeof(float));
        ASSERT_EQ(payload->table.size(), 1);

        ASSERT_EQ(payload->table[a], aSize * sizeof(float));
        ASSERT_EQ(payload->lastNdims, 1);
        ASSERT_EQ(payload->lastDims, af::dim4(aSize) * sizeof(float));
        ASSERT_EQ(payload->lastElementSize, 1);

        dim_t bDim = 2;
        auto b     = af::randu({bDim, bDim});

        ASSERT_EQ(payload->totalBytes, aSize * sizeof(float) + b.bytes());
        ASSERT_EQ(payload->totalBuffers, 2);
        ASSERT_EQ(payload->lockedBytes, aSize * sizeof(float) + b.bytes());
        ASSERT_EQ(payload->locked.size(), 2);
        ASSERT_EQ(payload->lastNdims, 1);
        ASSERT_EQ(payload->lastDims, af::dim4(bDim * b.numdims()));
        ASSERT_EQ(payload->lastElementSize, sizeof(float));

        af::freeV2(a);

        ASSERT_EQ(payload->totalBytes, aSize * sizeof(float) + b.bytes());
        ASSERT_EQ(payload->totalBuffers, 2);
        ASSERT_EQ(payload->lockedBytes, b.bytes());
        ASSERT_EQ(payload->locked.size(), 1);
    }

    // gc
    af::deviceGC();
    ASSERT_EQ(payload->table.size(), 0);

    // printInfo
    std::string printInfoMsg = "testPrintInfo";
    int printInfoDeviceId    = 0;
    af::printMemInfo(printInfoMsg.c_str(), printInfoDeviceId);
    ASSERT_EQ(printInfoMsg, payload->printInfoStringArg);
    ASSERT_EQ(printInfoDeviceId, payload->printInfoDevice);

    // step size (throws with a custom memory manager)
    ASSERT_THROW(af::setMemStepSize(500), af::exception);
    ASSERT_THROW(af::getMemStepSize(), af::exception);

    ASSERT_EQ(payload->table.size(), 0);
    af_unset_memory_manager();
    af_release_memory_manager(manager);
    ASSERT_EQ(payload->initializeCalledTimes, 1);
    ASSERT_EQ(payload->shutdownCalledTimes, af::getDeviceCount());
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
TEST(Memory, AfAllocDeviceCPUC) {
    af_backend active_backend;
    ASSERT_SUCCESS(af_get_active_backend(&active_backend));

    if (active_backend == AF_BACKEND_CPU) {
        void *ptr;
        ASSERT_SUCCESS(af_alloc_device(&ptr, sizeof(float)));

        // This is the CPU backend so we can assign to the pointer
        *static_cast<float *>(ptr) = 5;
        ASSERT_SUCCESS(af_free_device(ptr));
    }
}
#pragma GCC diagnostic pop

TEST(Memory, AfAllocDeviceV2CPUC) {
    af_backend active_backend;
    ASSERT_SUCCESS(af_get_active_backend(&active_backend));

    if (active_backend == AF_BACKEND_CPU) {
        void *ptr;
        ASSERT_SUCCESS(af_alloc_device_v2(&ptr, sizeof(float)));

        // This is the CPU backend so we can assign to the pointer
        *static_cast<float *>(ptr) = 5;
        ASSERT_SUCCESS(af_free_device_v2(ptr));
    }
}

TEST(Memory, SNIPPET_AllocCPU) {
    af_backend active_backend;
    ASSERT_SUCCESS(af_get_active_backend(&active_backend));

    if (active_backend == AF_BACKEND_CPU) {
        //! [ex_alloc_v2_cpu]

        // Allocate one float and cast to float*
        void *ptr   = af::allocV2(sizeof(float));
        float *dptr = static_cast<float *>(ptr);

        // This is the CPU backend so we can assign to the pointer
        dptr[0] = 5.0f;
        freeV2(ptr);

        //! [ex_alloc_v2_cpu]

        ASSERT_EQ(*dptr, 5.0f);
    }
}
