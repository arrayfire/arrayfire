/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>
#include <af/internal.h>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::cfloat;
using af::cdouble;

const size_t step_bytes = 1024;

TEST(Memory, Scope)
{
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate(); // Clean up everything done so far

    {
        af::array a = af::randu(5, 5);

        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                          &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);

        ASSERT_EQ(alloc_bytes, 1 * step_bytes);
        ASSERT_EQ(lock_bytes, 1 * step_bytes);
    }


    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 0u); // 0 because a is out of scope

    ASSERT_EQ(alloc_bytes, 1 * step_bytes);
    ASSERT_EQ(lock_bytes, 0u);
}

template<typename T>
class MemAlloc: public ::testing::Test
{
    public:
        virtual void SetUp() { }
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble,
                         int, unsigned int, intl, uintl,
                         char, unsigned char, short, ushort
                        > TestTypes;

// register the type list
TYPED_TEST_CASE(MemAlloc, TestTypes);

size_t roundUpToStep(size_t bytes)
{
    if (step_bytes == 0)
        return bytes;

    size_t remainder = bytes % step_bytes;
    if (remainder == 0)
        return bytes;

    return bytes + step_bytes - remainder;
}

template<typename T>
void memAllocArrayScopeTest(int elements)
{
    if (noDoubleTests<T>()) return;

    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate(); // Clean up everything done so far

    {
        af::array a = af::randu(elements, (af_dtype)af::dtype_traits<T>::af_type);

        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                          &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);

        ASSERT_EQ(alloc_bytes, roundUpToStep(elements * sizeof(T)));
        ASSERT_EQ(lock_bytes, roundUpToStep(elements * sizeof(T)));
    }

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 0u); // 0 because a is out of scope

    ASSERT_EQ(alloc_bytes, roundUpToStep(elements * sizeof(T)));
    ASSERT_EQ(lock_bytes, 0u);
}

template<typename T>
void memAllocPtrScopeTest(int elements)
{
    if (noDoubleTests<T>()) return;

    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate(); // Clean up everything done so far

    {
        T *ptr = af::alloc<T>(elements);

        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                          &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);

        ASSERT_EQ(alloc_bytes, roundUpToStep(elements * sizeof(T)));
        ASSERT_EQ(lock_bytes, roundUpToStep(elements * sizeof(T)));

        af::free(ptr);
    }

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 0u); // 0 because a is out of scope

    ASSERT_EQ(alloc_bytes, roundUpToStep(elements * sizeof(T)));
    ASSERT_EQ(lock_bytes, 0u);

    // Do without using templated alloc
    cleanSlate(); // Clean up everything done so far

    {
        void *ptr = af::alloc(elements, (af_dtype)af::dtype_traits<T>::af_type);

        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                          &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);

        ASSERT_EQ(alloc_bytes, roundUpToStep(elements * sizeof(T)));
        ASSERT_EQ(lock_bytes, roundUpToStep(elements * sizeof(T)));

        af::free(ptr);
    }

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 0u); // 0 because a is out of scope

    ASSERT_EQ(alloc_bytes, roundUpToStep(elements * sizeof(T)));
    ASSERT_EQ(lock_bytes, 0u);
}

TYPED_TEST(MemAlloc, ArrayScope25)
{
    memAllocArrayScopeTest<TypeParam>(25);
}

TYPED_TEST(MemAlloc, ArrayScope2048)
{
    memAllocArrayScopeTest<TypeParam>(2048);
}

TYPED_TEST(MemAlloc, ArrayScope2293)
{
    memAllocArrayScopeTest<TypeParam>(2293);
}

TYPED_TEST(MemAlloc, PtrScope25)
{
    memAllocPtrScopeTest<TypeParam>(25);
}

TYPED_TEST(MemAlloc, PtrScope2048)
{
    memAllocPtrScopeTest<TypeParam>(2048);
}

TYPED_TEST(MemAlloc, PtrScope2293)
{
    memAllocPtrScopeTest<TypeParam>(2293);
}

TEST(Memory, SingleSizeLoop)
{
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate(); // Clean up everything done so far

    {
        af::array a = af::randu(5, 5);

        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);
        ASSERT_EQ(alloc_bytes, 1 * step_bytes);
        ASSERT_EQ(lock_bytes, 1 * step_bytes);

        for (int i = 0; i < 100; i++) {

            a = af::randu(5,5);

            af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                              &lock_bytes, &lock_buffers);

            ASSERT_EQ(alloc_buffers, 2u); //2 because a new one is created before a is destroyed
            ASSERT_EQ(lock_buffers, 1u);
            ASSERT_EQ(alloc_bytes, 2 * step_bytes);
            ASSERT_EQ(lock_bytes, 1 * step_bytes);
        }
    }
}

TEST(Memory, LargeLoop)
{
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate(); // Clean up everything done so far

    const int num = step_bytes / sizeof(float);
    size_t allocated = step_bytes;

    af::array a = af::randu(num);

    std::vector<float> hA(num);

    a.host(&hA[0]);

    // Run a large loop that allocates more and more memory at each step
    for (int i = 0; i < 250; i++) {
        af::array b = af::randu(num * (i + 1));
        size_t current = (i + 1) * step_bytes;
        allocated += current;

        // Verify that new buffers are being allocated
        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                          &lock_bytes, &lock_buffers);

        // Limit to 10 to check before garbage collection
        if (i < 10) {
            ASSERT_EQ(alloc_buffers, (size_t)(i + 2)); //i is zero based
            ASSERT_EQ(lock_buffers, 2u);

            ASSERT_EQ(alloc_bytes, allocated);
            ASSERT_EQ(lock_bytes, current + step_bytes);
        }
    }

    size_t old_alloc_bytes = alloc_bytes;
    size_t old_alloc_buffers = alloc_buffers;

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(old_alloc_bytes, alloc_bytes);
    ASSERT_EQ(old_alloc_buffers, alloc_buffers);

    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(lock_bytes, 1 * step_bytes);
}

TEST(Memory, IndexingOffset)
{
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate(); // Clean up everything done so far

    const int num = step_bytes / sizeof(float);

    af::array a = af::randu(num);

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, 1 * step_bytes);
    ASSERT_EQ(lock_bytes, 1 * step_bytes);

    {
        af::array b = a(af::seq(1, num/2)); // Should just be an offset

        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                          &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);
        ASSERT_EQ(alloc_bytes, 1 * step_bytes);
        ASSERT_EQ(lock_bytes, 1 * step_bytes);
    }


    // b should not have deleted a
    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, 1 * step_bytes);
    ASSERT_EQ(lock_bytes, 1 * step_bytes);

}

TEST(Memory, IndexingCopy)
{
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate(); // Clean up everything done so far

    const int num = step_bytes / sizeof(float);

    af::array a = af::randu(num);

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, 1 * step_bytes);
    ASSERT_EQ(lock_bytes, 1 * step_bytes);

    {
        // Should just a copy
        af::array b = a(af::seq(0, num/2-1, 2));

        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                          &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 2u);
        ASSERT_EQ(lock_buffers, 2u);
        ASSERT_EQ(alloc_bytes, 2 * step_bytes);
        ASSERT_EQ(lock_bytes, 2 * step_bytes);
    }


    // b should not have deleted a
    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 2u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, 2 * step_bytes);
    ASSERT_EQ(lock_bytes, 1 * step_bytes);

}

TEST(Memory, Assign)
{
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate(); // Clean up everything done so far

    const int num = step_bytes / sizeof(float);

    af::array a = af::randu(num);

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, 1 * step_bytes);
    ASSERT_EQ(lock_bytes, 1 * step_bytes);

    {
        af::array b = af::randu(num / 2);
        a(af::seq(num / 2)) = b;

        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                          &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 2u);
        ASSERT_EQ(lock_buffers, 2u);
        ASSERT_EQ(alloc_bytes, 2 * step_bytes);
        ASSERT_EQ(lock_bytes, 2 * step_bytes);
    }


    // b should not have deleted a
    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 2u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, 2 * step_bytes);
    ASSERT_EQ(lock_bytes, 1 * step_bytes);

}

TEST(Memory, AssignLoop)
{
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate(); // Clean up everything done so far

    const int num = step_bytes / sizeof(float);
    const int cols = 100;

    af::array a = af::randu(num, cols);

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, cols * step_bytes);
    ASSERT_EQ(lock_bytes, cols * step_bytes);

    for (int i = 0; i < cols; i++) {

        af::array b = af::randu(num);
        a(af::span, i) = b;

        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                          &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 2u); // 3 because you need another scratch space for b
        ASSERT_EQ(lock_buffers, 2u);
        ASSERT_EQ(alloc_bytes, (cols + 1) * step_bytes);
        ASSERT_EQ(lock_bytes, (cols + 1) * step_bytes);
    }
}

TEST(Memory, AssignRef)
{
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate(); // Clean up everything done so far

    const int num = step_bytes / sizeof(float);

    af::array a = af::randu(num);
    af::array a_ref = a;

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, 1 * step_bytes);
    ASSERT_EQ(lock_bytes, 1 * step_bytes);

    {
        af::array b = af::randu(num / 2);
        // This should do a full copy of a
        a(af::seq(num / 2)) = b;

        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                          &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 3u);
        ASSERT_EQ(lock_buffers, 3u);
        ASSERT_EQ(alloc_bytes, 3 * step_bytes);
        ASSERT_EQ(lock_bytes, 3 * step_bytes);
    }


    // b should not have deleted a
    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 3u);
    ASSERT_EQ(lock_buffers, 2u); // a_ref
    ASSERT_EQ(alloc_bytes, 3 * step_bytes);
    ASSERT_EQ(lock_bytes, 2 * step_bytes); // a_ref

}

TEST(Memory, AssignRefLoop)
{
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate(); // Clean up everything done so far

    const int num = step_bytes / sizeof(float);
    const int cols = 100;

    af::array a = af::randu(num, cols);
    af::array a_ref = a;

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, cols * step_bytes);
    ASSERT_EQ(lock_bytes, cols * step_bytes);

    for (int i = 0; i < cols; i++) {

        af::array b = af::randu(num);
        a(af::span, i) = b;

        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                          &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 3u);
        ASSERT_EQ(lock_buffers, 3u);
        ASSERT_EQ(alloc_bytes, (2 * cols + 1) * step_bytes);
        ASSERT_EQ(lock_bytes, (2 * cols + 1) * step_bytes);
    }


    // b should not have deleted a
    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 3u);
    ASSERT_EQ(lock_buffers, 2u); // a_ref
    ASSERT_EQ(alloc_bytes, (2 * cols + 1) * step_bytes);
    ASSERT_EQ(lock_bytes, 2 * cols * step_bytes); // a_ref

}

TEST(Memory, device)
{
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate(); // Clean up everything done so far

    {
        af::array a = af::randu(5, 5);

        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                          &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);
        ASSERT_EQ(alloc_bytes, 1 * step_bytes);
        ASSERT_EQ(lock_bytes, 1 * step_bytes);

        a.device<float>();

        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                          &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);
        ASSERT_EQ(alloc_bytes, 1 * step_bytes);
        ASSERT_EQ(lock_bytes, 1 * lock_bytes);

        a.unlock(); //to reset the lock flag
    }

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 0u);
    ASSERT_EQ(alloc_bytes, 1 * step_bytes);
    ASSERT_EQ(lock_bytes, 0u);
}

TEST(Memory, unlock)
{

    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate(); // Clean up everything done so far

    const dim_t num = step_bytes / sizeof(float);

    std::vector<float> in(num);

    af_array arr = 0;
    ASSERT_EQ(AF_SUCCESS, af_create_array(&arr, &in[0], 1, &num, f32));

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, step_bytes);
    ASSERT_EQ(lock_bytes, step_bytes);

    // arr1 gets released by end of the following code block
    {
        af::array a(arr);
        a.lock();

        // No new memory should be allocated
        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                          &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);
        ASSERT_EQ(alloc_bytes, step_bytes);
        ASSERT_EQ(lock_bytes, step_bytes);

        a.unlock();
    }

    // Making sure all unlocked buffers are freed
    af::deviceGC();

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 0u);
    ASSERT_EQ(lock_buffers, 0u);
    ASSERT_EQ(alloc_bytes, 0u);
    ASSERT_EQ(lock_bytes, 0u);
}

TEST(Memory, IndexedDevice)
{
    // This test is checking to see if calling .device() will force copy to a new buffer
    const int nx = 8;
    const int ny = 8;

    af::array in = af::randu(nx, ny);

    std::vector<float> in1(in.elements());
    in.host(&in1[0]);

    int offx = nx / 4;
    int offy = ny / 4;

    in = in(af::seq(offx, offx + nx/2 - 1),
            af::seq(offy, offy + ny/2- 1));

    int nxo = (int)in.dims(0);
    int nyo = (int)in.dims(1);

    void *rawPtr = af::getRawPtr(in);
    void *devPtr = in.device<float>();
    ASSERT_NE(devPtr, rawPtr);
    in.unlock();

    std::vector<float> in2(in.elements());
    in.host(&in2[0]);

    for (int y = 0; y < nyo; y++) {
        for (int x = 0; x < nxo; x++) {
            ASSERT_EQ(in1[(offy + y) * nx + offx + x], in2[y * nxo + x]);
        }
    }
}
