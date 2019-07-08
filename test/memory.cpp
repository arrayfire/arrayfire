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
#include <af/traits.hpp>
#include <iostream>
#include <vector>

using af::alloc;
using af::array;
using af::cdouble;
using af::cfloat;
using af::deviceGC;
using af::deviceMemInfo;
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
        array a    = af::randu(10, 10, f32);
        unsigned hb[] = {3, 5, 6, 8, 9};
        array b(5, hb);
        array c   = af::randu(5, f32);
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
