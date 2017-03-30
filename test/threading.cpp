/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cstddef>
#include <gtest/gtest.h>
#include <arrayfire.h>
#include <testHelpers.hpp>
#include <thread>
#include <chrono>

using namespace af;

using std::vector;
using std::string;

static const int THREAD_COUNT = 32;

#if defined(AF_CPU)
static const unsigned ITERATION_COUNT = 10;
#else
static const unsigned ITERATION_COUNT = 1000;
#endif

void morphTest(const array input, const array mask, const bool isDilation,
               const array gold, int targetDevice)
{
    auto start = std::chrono::high_resolution_clock::now();

    af::setDevice(targetDevice);

    vector<float> goldData(gold.elements());
    vector<float> outData(gold.elements());

    gold.host((void*)goldData.data());

    af::array out;

    for (unsigned i=0; i<ITERATION_COUNT; ++i)
        out = isDilation ? dilate(input, mask) : erode(input, mask);

    out.host((void*)outData.data());

    ASSERT_EQ(true, compareArraysRMSD(gold.elements(), goldData.data(), outData.data(), 0.018f));

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;

    std::cout << "Thread(" << std::this_thread::get_id()
              << "): time taken for "
              << ITERATION_COUNT
              <<" is "
              << diff.count() << " s\n";
}

TEST(Threading, SetPerThreadActiveDevice)
{
    if (noImageIOTests()) return;

    vector<bool> isDilationFlags;
    vector<bool> isColorFlags;
    vector<string> files;

    files.push_back( string(TEST_DIR "/morph/gray.test") );
    isDilationFlags.push_back(true);
    isColorFlags.push_back(false);

    files.push_back( string(TEST_DIR "/morph/color.test") );
    isDilationFlags.push_back(false);
    isColorFlags.push_back(true);

    vector<std::thread> tests;
    unsigned totalTestCount = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for(size_t pos = 0; pos<files.size(); ++pos)
    {
        const bool isDilation = isDilationFlags[pos];
        const bool isColor    = isColorFlags[pos];

        vector<dim4>    inDims;
        vector<string>  inFiles;
        vector<dim_t>   outSizes;
        vector<string>  outFiles;

        readImageTests(files[pos], inDims, inFiles, outSizes, outFiles);

        const unsigned testCount = inDims.size();

        const dim4 maskdims(3,3,1,1);

        for (size_t testId=0; testId<testCount; ++testId)
        {
            int trgtDeviceId = totalTestCount % af::getDeviceCount();

            //prefix full path to image file names
            inFiles[testId].insert(0,string(TEST_DIR "/morph/"));
            outFiles[testId].insert(0,string(TEST_DIR "/morph/"));

            af::setDevice(trgtDeviceId);

            const array mask = constant(1.0, maskdims);

            array input= loadImage(inFiles[testId].c_str(), isColor);
            array gold = loadImage(outFiles[testId].c_str(), isColor);

            //Push the new test as a new thread of execution
            tests.emplace_back(morphTest, input, mask, isDilation, gold, trgtDeviceId);

            std::cout<<"morph test launched with the following params on device ("
                     <<trgtDeviceId<<"):"<<std::endl;
            std::cout<<"\t Input image dims: "<<input.dims()<<std::endl;
            std::cout<<"\t Mask dims: "<<mask.dims()<<std::endl;
            std::cout<<"\t IsDilation : "<< (isDilation ? "True" : "False") <<std::endl;

            totalTestCount++;
        }
    }
    std::cout<< std::endl << "Waiting for results ..." << std::endl << std::endl;

    for (size_t testId=0; testId<tests.size(); ++testId)
    {
        if (tests[testId].joinable()) {
            std::cout<<"Attempting join for test ..."<<testId<<std::endl;
            tests[testId].join();
            std::cout<<"test "<< testId <<" completed." << std::endl;
        }
        std::cout<<std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;

    std::cout << "Total time taken for test : " << diff.count() << " s\n";
}

enum ArithOp
{
    ADD, SUB, DIV, MUL
};

void calc(ArithOp opcode, array op1, array op2, float outValue)
{
    af::setDevice(0);
    array res;
    for (unsigned i=0; i<ITERATION_COUNT; ++i)
    {
        switch(opcode) {
            case ADD: res = op1 + op2; break;
            case SUB: res = op1 - op2; break;
            case DIV: res = op1 / op2; break;
            case MUL: res = op1 * op2; break;
        }
    }

    std::vector<float> out(res.elements());
    res.host((void*)out.data());

    for (unsigned i=0; i <out.size(); ++i)
        ASSERT_EQ(out[i], outValue);
}

TEST(Threading, SimultaneousRead)
{
    af::setDevice(0);
    af::array A = af::constant(1.0, 100, 100);
    af::array B = af::constant(1.0, 100, 100);

    vector<std::thread> tests;

    for (int t=0; t<THREAD_COUNT; ++t)
    {
        ArithOp op;
        float outValue;

        switch(t%4) {
            case 0: op = ADD; outValue = 2.0f; break;
            case 1: op = SUB; outValue = 0.0f; break;
            case 2: op = DIV; outValue = 1.0f; break;
            case 3: op = MUL; outValue = 1.0f; break;
        }

        tests.emplace_back(calc, op, A, B, outValue);
    }

    for (int t=0; t<THREAD_COUNT; ++t)
        if (tests[t].joinable())
            tests[t].join();
}

int nextTargetDeviceId()
{
    static int nextId = 0;
    return nextId++;
}

static void cleanSlate()
{
    const size_t step_bytes = 1024;

    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    af::deviceGC();

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 0u);
    ASSERT_EQ(lock_buffers, 0u);
    ASSERT_EQ(alloc_bytes, 0u);
    ASSERT_EQ(lock_bytes, 0u);

    af::setMemStepSize(step_bytes);

    ASSERT_EQ(af::getMemStepSize(), step_bytes);
}

void doubleAllocationTest()
{
    af::setDevice(0);

    af::array a = randu(5, 5);

    for (int i = 0; i < 100; ++i)
    {
        a = randu(5, 5);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

TEST(Threading, MemoryManagement_Double_Alloc)
{
    cleanSlate(); // Clean up everything done so far

    vector<std::thread> tests;

    for (int t=0; t<THREAD_COUNT; ++t)
        tests.emplace_back(doubleAllocationTest);

    for (int t=0; t<THREAD_COUNT; ++t)
        if (tests[t].joinable())
            tests[t].join();

    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
            &lock_bytes, &lock_buffers);

    ASSERT_EQ( lock_buffers,     0u);
    ASSERT_EQ(  lock_bytes,      0u);
    ASSERT_LE(alloc_buffers,    64u);
    ASSERT_GT(alloc_buffers,    32u);
    ASSERT_LE(  alloc_bytes, 65536u);
    ASSERT_GT(  alloc_bytes, 32768u);
}

void singleAllocationTest()
{
    af::setDevice(0);

    for (int i = 0; i < 100; ++i)
    {
        af::array a = af::randu(5, 5);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

TEST(Threading, MemoryManagement_Single_Alloc)
{
    cleanSlate(); // Clean up everything done so far

    vector<std::thread> tests;

    for (int t=0; t<THREAD_COUNT; ++t)
        tests.emplace_back(singleAllocationTest);

    for (int t=0; t<THREAD_COUNT; ++t)
        if (tests[t].joinable())
            tests[t].join();

    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
            &lock_bytes, &lock_buffers);

    ASSERT_EQ( lock_buffers,     0u);
    ASSERT_EQ(  lock_bytes,      0u);
    ASSERT_LE(alloc_buffers,    32u);
    ASSERT_GT(alloc_buffers,     0u);
    ASSERT_LE(  alloc_bytes, 32768u);
    ASSERT_GT(  alloc_bytes,     0u);
}

void jitAllocationTest()
{
    af::setDevice(0);

    for (int i = 0; i < 100; ++i)
        af::array a = af::constant(1, 5, 5);
}

TEST(Threading, MemoryManagement_JIT_Node)
{
    cleanSlate(); // Clean up everything done so far

    vector<std::thread> tests;

    for (int t=0; t<THREAD_COUNT; ++t)
        tests.emplace_back(jitAllocationTest);

    for (int t=0; t<THREAD_COUNT; ++t)
        if (tests[t].joinable())
            tests[t].join();

    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
            &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers,     0u);
    ASSERT_EQ( lock_buffers,     0u);
    ASSERT_EQ(  alloc_bytes,     0u);
    ASSERT_EQ(  lock_bytes,      0u);
}
