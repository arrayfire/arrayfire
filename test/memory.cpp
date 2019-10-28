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
#include <af/event.h>
#include <af/internal.h>
#include <af/memory.h>
#include <af/traits.hpp>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using af::alloc;
using af::array;
using af::buffer_info;
using af::cdouble;
using af::cfloat;
using af::deviceGC;
using af::deviceMemInfo;
using af::dtype;
using af::dtype_traits;
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
TYPED_TEST_CASE(MemAlloc, TestTypes);

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
        T *ptr = alloc<T>(elements);

        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);

        ASSERT_EQ(alloc_bytes, roundUpToStep(elements * sizeof(T)));
        ASSERT_EQ(lock_bytes, roundUpToStep(elements * sizeof(T)));

        af::free(ptr);
    }

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 0u);  // 0 because a is out of scope

    ASSERT_EQ(alloc_bytes, roundUpToStep(elements * sizeof(T)));
    ASSERT_EQ(lock_bytes, 0u);

    // Do without using templated alloc
    cleanSlate();  // Clean up everything done so far

    {
        void *ptr = alloc(elements, (af_dtype)dtype_traits<T>::af_type);

        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);

        ASSERT_EQ(alloc_bytes, roundUpToStep(elements * sizeof(T)));
        ASSERT_EQ(lock_bytes, roundUpToStep(elements * sizeof(T)));

        af::free(ptr);
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

        // Limit to 10 to check before garbage collection
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

TEST(BufferInfo, SimpleCreateDelete) {
    af_event event;
    ASSERT_SUCCESS(af_create_event(&event));
    af_buffer_info pair;

    void *ptr = af::alloc(1, dtype::f32);
    ASSERT_SUCCESS(af_create_buffer_info(&pair, ptr, event));
    ASSERT_SUCCESS(af_delete_buffer_info(pair));
}

TEST(BufferInfo, Unlock) {
    af_event event;
    ASSERT_SUCCESS(af_create_event(&event));
    af_buffer_info pair;
    void *ptr = af::alloc(1, dtype::f32);
    ASSERT_SUCCESS(af_create_buffer_info(&pair, ptr, event));

    void *curPtr;
    ASSERT_SUCCESS(af_unlock_buffer_info_ptr(&curPtr, pair));
    ASSERT_EQ(curPtr, ptr);
    void *zeroPtr;
    ASSERT_SUCCESS(af_buffer_info_get_ptr(&zeroPtr, pair));
    ASSERT_EQ(zeroPtr, nullptr);
    ASSERT_SUCCESS(af_unlock_buffer_info_ptr(&zeroPtr, pair));
    ASSERT_EQ(zeroPtr, nullptr);

    af_event curEvent;
    ASSERT_SUCCESS(af_unlock_buffer_info_event(&curEvent, pair));
    ASSERT_EQ(curEvent, event);
    void *zeroEvent;
    ASSERT_SUCCESS(af_buffer_info_get_ptr(&zeroEvent, pair));
    ASSERT_EQ(zeroEvent, nullptr);
    ASSERT_SUCCESS(af_unlock_buffer_info_ptr(&zeroEvent, pair));
    ASSERT_EQ(zeroEvent, nullptr);

    ASSERT_SUCCESS(af_delete_buffer_info(pair));
    ASSERT_SUCCESS(af_release_event(event));
    af::free(ptr);
}

TEST(BufferInfo, EventAndPtrAttributes) {
    af_event event;
    ASSERT_SUCCESS(af_create_event(&event));
    void *ptr = af::alloc(1, dtype::f32);
    af_buffer_info pair;
    ASSERT_SUCCESS(af_create_buffer_info(&pair, ptr, event));
    af_event anEvent;
    ASSERT_SUCCESS(af_buffer_info_get_event(&anEvent, pair));
    ASSERT_EQ(event, anEvent);
    void *somePtr;
    ASSERT_SUCCESS(af_buffer_info_get_ptr(&somePtr, pair));
    ASSERT_EQ(ptr, somePtr);

    af_event anotherEvent;
    ASSERT_SUCCESS(af_create_event(&anotherEvent));
    ASSERT_SUCCESS(af_buffer_info_set_event(pair, anotherEvent));
    af_event yetAnotherEvent;
    ASSERT_SUCCESS(af_buffer_info_get_event(&yetAnotherEvent, pair));
    ASSERT_NE(yetAnotherEvent, event);
    ASSERT_EQ(yetAnotherEvent, anotherEvent);

    void *anotherPtr = af::alloc(1, dtype::f32);
    ASSERT_SUCCESS(af_buffer_info_set_ptr(pair, anotherPtr));
    void *yetAnotherPtr;
    ASSERT_SUCCESS(af_buffer_info_get_ptr(&yetAnotherPtr, pair));
    ASSERT_NE(yetAnotherPtr, ptr);
    ASSERT_EQ(yetAnotherPtr, anotherPtr);

    ASSERT_SUCCESS(af_delete_buffer_info(pair));
    ASSERT_SUCCESS(af_release_event(event));
    af::free(ptr);
}

TEST(BufferInfo, BufferInfoCreateMove) {
    af_event event;
    ASSERT_SUCCESS(af_create_event(&event));
    void *ptr = af::alloc(1, dtype::f32);
    std::unique_ptr<buffer_info> bufferInfo;
    bufferInfo.reset(new buffer_info(ptr, event));
    ASSERT_EQ(bufferInfo->getEvent(), event);
    ASSERT_EQ(bufferInfo->getPtr(), ptr);

    void *anotherPtr = af::alloc(1, dtype::f32);
    bufferInfo->setPtr(anotherPtr);
    ASSERT_EQ(bufferInfo->getPtr(), anotherPtr);

    af_event anotherEvent;
    ASSERT_SUCCESS(af_create_event(&anotherEvent));
    bufferInfo->setEvent(anotherEvent);
    ASSERT_EQ(bufferInfo->getEvent(), anotherEvent);

    auto anotherBufferInfo = std::move(bufferInfo);
    ASSERT_EQ(anotherBufferInfo->getPtr(), anotherPtr);
    ASSERT_EQ(anotherBufferInfo->getEvent(), anotherEvent);

    af_release_event(event);
    af::free(ptr);
}

TEST(BufferInfo, UnlockCpp) {
    af_event event;
    ASSERT_SUCCESS(af_create_event(&event));
    void *ptr = af::alloc(1, dtype::f32);
    std::unique_ptr<buffer_info> bufferInfo;
    bufferInfo.reset(new buffer_info(ptr, event));

    void *anotherPtr = bufferInfo->unlockPtr();
    ASSERT_EQ(ptr, anotherPtr);
    af_event anotherEvent = bufferInfo->unlockEvent();
    ASSERT_EQ(event, anotherEvent);

    ASSERT_SUCCESS(af_release_event(anotherEvent));
    af::free(ptr);
}

namespace {

template<typename T>
T *getMemoryManagerPayload(af_memory_manager manager) {
    void *payloadPtr;
    af_memory_manager_get_payload(manager, &payloadPtr);
    return (T *)payloadPtr;
}

struct InitializeShutdownPayload {
    bool initializeCalled = false;
    bool shutdownCalled   = false;
};

}  // namespace

TEST(MemoryManagerApi, InitializeShutdown) {
    af_memory_manager manager;
    af_create_memory_manager(&manager);

    // Set payload
    std::unique_ptr<InitializeShutdownPayload> payload;
    payload.reset(new InitializeShutdownPayload());
    af_memory_manager_set_payload(manager, payload.get());

    auto initialize_fn = [](af_memory_manager manager) {
        auto *payload =
            getMemoryManagerPayload<InitializeShutdownPayload>(manager);
        payload->initializeCalled = true;
    };
    af_memory_manager_set_initialize_fn(manager, initialize_fn);

    auto shutdown_fn = [](af_memory_manager manager) {
        auto *payload =
            getMemoryManagerPayload<InitializeShutdownPayload>(manager);
        payload->shutdownCalled = true;
    };
    af_memory_manager_set_shutdown_fn(manager, shutdown_fn);

    af_set_memory_manager(manager);
    af_unset_memory_manager();
    af_release_memory_manager(manager);
    ASSERT_TRUE(payload->initializeCalled);
    ASSERT_TRUE(payload->shutdownCalled);
}

namespace {

/**
 * Below is an extremely basic memory manager with a basic
 * caching mechanism for testing purposes. It is not thread safe or optimized.
 */
struct E2ETestPayload {
    std::unordered_map<void *, size_t> table;
    std::unordered_set<void *> locked;
    size_t totalBytes{0};
    size_t totalBuffers{0};
    size_t lockedBytes{0};
    size_t memStepSize{8};

    size_t maxBuffers{64};
    size_t maxBytes{1024};
    // Print info args
    std::string printInfoStringArg;
    int printInfoDevice{-1};
};

size_t allocated_fn(af_memory_manager manager, void *ptr) {
    return getMemoryManagerPayload<E2ETestPayload>(manager)->table[ptr];
}

void user_lock_fn(af_memory_manager manager, void *ptr) {
    auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
    if (payload->locked.find(ptr) == payload->locked.end()) {
        payload->locked.insert(ptr);
        payload->lockedBytes += payload->table[ptr];
    }
}

int is_user_locked_fn(af_memory_manager manager, void *ptr) {
    auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
    return payload->locked.find(ptr) != payload->locked.end();
}

void unlock_fn(af_memory_manager manager, void *ptr, af_event event,
               int userLock) {
    af_release_event(event);
    if (!ptr) { return; }

    auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);

    if (payload->table.find(ptr) == payload->table.end()) {
        return;  // fast path
    }

    // For testing, treat user-allocated and AF-allocated memory identically
    if (payload->locked.find(ptr) != payload->locked.end()) {
        payload->locked.erase(ptr);
        payload->lockedBytes -= payload->table[ptr];
    }
}

void user_unlock_fn(af_memory_manager manager, void *ptr) {
    auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
    af_event event;
    af_create_event(&event);
    af_mark_event(event);
    unlock_fn(manager, ptr, event, /* user */ 1);
    payload->lockedBytes -= payload->table[ptr];
}

void garbage_collect_fn(af_memory_manager manager) {
    auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
    // Free unlocked memory
    std::vector<void *> freed;
    for (auto &entry : payload->table) {
        if (!is_user_locked_fn(manager, entry.first)) {
            void *ptr = entry.first;
            af_memory_manager_native_free(manager, ptr);
            payload->totalBytes -= payload->table[entry.first];
            freed.push_back(entry.first);
        }
    }
    for (auto ptr : freed) { payload->table.erase(ptr); }
}

void print_info_fn(af_memory_manager manager, char *c, int b) {
    auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
    payload->printInfoStringArg = std::string(c);
    payload->printInfoDevice    = b;
}

void usage_info_fn(af_memory_manager manager, size_t *alloc_bytes,
                   size_t *alloc_buffers, size_t *lock_bytes,
                   size_t *lock_buffers) {
    auto *payload  = getMemoryManagerPayload<E2ETestPayload>(manager);
    *alloc_bytes   = payload->totalBytes;
    *alloc_buffers = payload->totalBuffers;
    *lock_bytes    = payload->lockedBytes;
    *lock_buffers  = payload->locked.size();
}

size_t get_mem_step_size_fn(af_memory_manager manager) {
    return getMemoryManagerPayload<E2ETestPayload>(manager)->memStepSize;
}

size_t get_max_bytes_fn(af_memory_manager manager) {
    return getMemoryManagerPayload<E2ETestPayload>(manager)->maxBytes;
}

unsigned get_max_buffers_fn(af_memory_manager manager) {
    return getMemoryManagerPayload<E2ETestPayload>(manager)->maxBuffers;
}

void set_mem_step_size_fn(af_memory_manager manager, size_t step) {
    getMemoryManagerPayload<E2ETestPayload>(manager)->memStepSize = step;
}

int check_memory_limit_fn(af_memory_manager manager) {
    auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
    return payload->totalBytes < get_max_bytes_fn(manager) &&
           payload->totalBuffers < get_max_buffers_fn(manager);
}

af_buffer_info alloc_fn(af_memory_manager manager, size_t size,
                        /* bool */ int userLock) {
    af_event event;
    af_create_event(&event);
    af_mark_event(event);
    af_buffer_info bufferInfo;
    af_create_buffer_info(&bufferInfo, nullptr, event);

    if (size > 0) {
        if (check_memory_limit_fn(manager)) { garbage_collect_fn(manager); }

        void *piece;
        af_memory_manager_native_alloc(manager, &piece, size);
        af_buffer_info_set_ptr(bufferInfo, piece);

        auto *payload = getMemoryManagerPayload<E2ETestPayload>(manager);
        payload->table[piece] = size;
        payload->totalBytes += size;
        payload->totalBuffers++;

        // Simple implementation: treat user and AF allocations the same
        payload->locked.insert(piece);
        payload->lockedBytes += size;
    }

    return bufferInfo;
}

void add_memory_management_fn(af_memory_manager manager, int id) {}

void remove_memory_management_fn(af_memory_manager manager, int id) {}

}  // namespace

TEST(MemoryManagerApi, E2ETest) {
    af_memory_manager manager;
    af_create_memory_manager(&manager);

    // Set payload_fn
    std::unique_ptr<E2ETestPayload> payload(new E2ETestPayload());
    af_memory_manager_set_payload(manager, payload.get());

    auto initialize_fn = [](af_memory_manager) {};
    auto shutdown_fn   = [](af_memory_manager) {};
    af_memory_manager_set_initialize_fn(manager, initialize_fn);
    af_memory_manager_set_shutdown_fn(manager, shutdown_fn);

    // alloc
    af_memory_manager_set_alloc_fn(manager, alloc_fn);
    af_memory_manager_set_allocated_fn(manager, allocated_fn);
    af_memory_manager_set_unlock_fn(manager, unlock_fn);
    // utils
    af_memory_manager_set_garbage_collect_fn(manager, garbage_collect_fn);
    af_memory_manager_set_print_info_fn(manager, print_info_fn);
    af_memory_manager_set_usage_info_fn(manager, usage_info_fn);
    // user lock/unlock
    af_memory_manager_set_user_lock_fn(manager, user_lock_fn);
    af_memory_manager_set_user_unlock_fn(manager, user_unlock_fn);
    af_memory_manager_set_is_user_locked_fn(manager, is_user_locked_fn);
    // limits and step size
    af_memory_manager_set_get_mem_step_size_fn(manager, get_mem_step_size_fn);
    af_memory_manager_set_get_max_bytes_fn(manager, get_max_bytes_fn);
    af_memory_manager_set_get_max_buffers_fn(manager, get_max_buffers_fn);
    af_memory_manager_set_set_mem_step_size_fn(manager, set_mem_step_size_fn);
    af_memory_manager_set_check_memory_limit_fn(manager, check_memory_limit_fn);
    // ocl
    af_memory_manager_set_add_memory_management_fn(manager,
                                                   add_memory_management_fn);
    af_memory_manager_set_remove_memory_management_fn(
        manager, remove_memory_management_fn);

    af_set_memory_manager(manager);
    {
        size_t aSize = 8;

        void *a = af::alloc(8, af::dtype::f32);
        ASSERT_EQ(payload->table.size(), 1);
        ASSERT_EQ(payload->table[a], aSize * sizeof(float));

        auto b = af::randu({2, 2});

        // Usage info
        size_t allocBytes, allocBuffers, lockBytes, lockBuffers;
        af::deviceMemInfo(&allocBytes, &allocBuffers, &lockBytes, &lockBuffers);
        ASSERT_EQ(allocBytes, aSize * sizeof(float) + b.bytes());
        ASSERT_EQ(allocBuffers, 2);
        ASSERT_EQ(lockBytes, aSize * sizeof(float) + b.bytes());
        ASSERT_EQ(lockBuffers, 2);

        af::free(a);

        af::deviceMemInfo(&allocBytes, &allocBuffers, &lockBytes, &lockBuffers);
        ASSERT_EQ(allocBytes, aSize * sizeof(float) + b.bytes());
        ASSERT_EQ(allocBuffers, 2);
        ASSERT_EQ(lockBytes, b.bytes());
        ASSERT_EQ(lockBuffers, 1);

        ASSERT_EQ(payload->table.size(), 2);
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

    // step size
    size_t stepSizeTest = 64;
    af::setMemStepSize(stepSizeTest);
    ASSERT_EQ(af::getMemStepSize(), stepSizeTest);
    ASSERT_EQ(stepSizeTest, payload->memStepSize);

    ASSERT_EQ(payload->table.size(), 0);
    af_release_memory_manager(manager);
}
