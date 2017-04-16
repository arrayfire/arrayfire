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
#include <af/traits.hpp>
#include <thread>
#include <complex>
#include <chrono>
#include <thread>
#include <iterator>
#include <testHelpers.hpp>
#include <solve_common.hpp>
#include <sparse_common.hpp>

using namespace af;

using std::vector;
using std::string;

static const int THREAD_COUNT = 32;

#if defined(AF_CPU)
static const unsigned ITERATION_COUNT = 10;
#else
static const unsigned ITERATION_COUNT = 1000;
#endif

int nextTargetDeviceId()
{
    static int nextId = 0;
    return nextId++;
}

void morphTest(const array input, const array mask, const bool isDilation,
               const array gold, int targetDevice)
{
    af::setDevice(targetDevice);

    vector<float> goldData(gold.elements());
    vector<float> outData(gold.elements());

    gold.host((void*)goldData.data());

    af::array out;

    for (unsigned i=0; i<ITERATION_COUNT; ++i)
        out = isDilation ? dilate(input, mask) : erode(input, mask);

    out.host((void*)outData.data());

    ASSERT_EQ(true, compareArraysRMSD(gold.elements(), goldData.data(), outData.data(), 0.018f));
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

            totalTestCount++;
        }
    }

    for (size_t testId=0; testId<tests.size(); ++testId)
        if (tests[testId].joinable())
            tests[testId].join();
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

template<typename inType, typename outType, bool isInverse>
void fftTest(int targetDevice, string pTestFile, dim_t pad0=0, dim_t pad1=0, dim_t pad2=0)
{
    if (noDoubleTests<inType>()) return;
    if (noDoubleTests<outType>()) return;

    vector<af::dim4>        numDims;
    vector<vector<inType> >       in;
    vector<vector<outType> >   tests;

    readTestsFromFile<inType, outType>(pTestFile, numDims, in, tests);

    af::dim4 dims       = numDims[0];
    af_array outArray   = 0;
    af_array inArray    = 0;

    ASSERT_EQ(AF_SUCCESS, af_set_device(targetDevice));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()),
                dims.ndims(), dims.get(), (af_dtype)af::dtype_traits<inType>::af_type));

    if (isInverse){
        switch (dims.ndims()) {
            case 1 : ASSERT_EQ(AF_SUCCESS, af_ifft (&outArray, inArray, 1.0, pad0));              break;
            case 2 : ASSERT_EQ(AF_SUCCESS, af_ifft2(&outArray, inArray, 1.0, pad0, pad1));        break;
            case 3 : ASSERT_EQ(AF_SUCCESS, af_ifft3(&outArray, inArray, 1.0, pad0, pad1, pad2));  break;
            default: throw std::runtime_error("This error shouldn't happen, pls check");
        }
    } else {
        switch(dims.ndims()) {
            case 1 : ASSERT_EQ(AF_SUCCESS, af_fft (&outArray, inArray, 1.0, pad0));               break;
            case 2 : ASSERT_EQ(AF_SUCCESS, af_fft2(&outArray, inArray, 1.0, pad0, pad1));         break;
            case 3 : ASSERT_EQ(AF_SUCCESS, af_fft3(&outArray, inArray, 1.0, pad0, pad1, pad2));   break;
            default: throw std::runtime_error("This error shouldn't happen, pls check");
        }
    }

    size_t out_size = tests[0].size();
    outType *outData= new outType[out_size];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    vector<outType> goldBar(tests[0].begin(), tests[0].end());

    size_t test_size = 0;
    switch(dims.ndims()) {
        case 1  : test_size = dims[0]/2+1;                       break;
        case 2  : test_size = dims[1] * (dims[0]/2+1);           break;
        case 3  : test_size = dims[2] * dims[1] * (dims[0]/2+1); break;
        default : test_size = dims[0]/2+1;                       break;
    }
    outType output_scale = (outType)(isInverse ? test_size : 1);
    for (size_t elIter=0; elIter<test_size; ++elIter) {
        bool isUnderTolerance = abs(goldBar[elIter]-outData[elIter])<0.001;
        ASSERT_EQ(true, isUnderTolerance)<<
            "Expected value="<<goldBar[elIter] <<"\t Actual Value="<<
            (output_scale*outData[elIter]) << " at: " << elIter <<
            " from thread: "<< std::this_thread::get_id() << std::endl;
    }

    // cleanup
    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

#define INSTANTIATE_TEST(func, name, is_inverse, in_t, out_t, file)                         \
    {                                                                                       \
        int targetDevice = nextTargetDeviceId() % numDevices;                               \
        tests.emplace_back(fftTest<in_t, out_t, is_inverse>, targetDevice, file, 0, 0, 0);  \
    }

#define INSTANTIATE_TEST_TP(func, name, is_inverse, in_t, out_t, file, p0, p1)              \
    {                                                                                       \
        int targetDevice = nextTargetDeviceId() % numDevices;                               \
        tests.emplace_back(fftTest<in_t, out_t, is_inverse>, targetDevice, file, p0, p1, 0);\
    }

TEST(Threading, FFT_R2C)
{
    cleanSlate(); // Clean up everything done so far

    vector<std::thread> tests;

    int numDevices = 1;
    ASSERT_EQ(AF_SUCCESS, af_get_device_count(&numDevices));

    // Real to complex transforms
    INSTANTIATE_TEST(fft ,  R2C_Float, false,  float,  cfloat, string(TEST_DIR"/signal/fft_r2c.test") );
    INSTANTIATE_TEST(fft , R2C_Double, false, double, cdouble, string(TEST_DIR"/signal/fft_r2c.test") );
    INSTANTIATE_TEST(fft2,  R2C_Float, false,  float,  cfloat, string(TEST_DIR"/signal/fft2_r2c.test"));
    INSTANTIATE_TEST(fft2, R2C_Double, false, double, cdouble, string(TEST_DIR"/signal/fft2_r2c.test"));
    INSTANTIATE_TEST(fft3,  R2C_Float, false,  float,  cfloat, string(TEST_DIR"/signal/fft3_r2c.test"));
    INSTANTIATE_TEST(fft3, R2C_Double, false, double, cdouble, string(TEST_DIR"/signal/fft3_r2c.test"));

    // Factors 7, 11, 13
    INSTANTIATE_TEST(fft , R2C_Float_7_11_13 , false, float  , cfloat , string(TEST_DIR"/signal/fft_r2c_7_11_13.test") );
    INSTANTIATE_TEST(fft , R2C_Double_7_11_13, false, double , cdouble, string(TEST_DIR"/signal/fft_r2c_7_11_13.test") );
    INSTANTIATE_TEST(fft2, R2C_Float_7_11_13 , false, float  , cfloat , string(TEST_DIR"/signal/fft2_r2c_7_11_13.test") );
    INSTANTIATE_TEST(fft2, R2C_Double_7_11_13, false, double , cdouble, string(TEST_DIR"/signal/fft2_r2c_7_11_13.test") );
    INSTANTIATE_TEST(fft3, R2C_Float_7_11_13 , false, float  , cfloat , string(TEST_DIR"/signal/fft3_r2c_7_11_13.test") );
    INSTANTIATE_TEST(fft3, R2C_Double_7_11_13, false, double , cdouble, string(TEST_DIR"/signal/fft3_r2c_7_11_13.test") );

    // transforms on padded and truncated arrays
    INSTANTIATE_TEST_TP(fft2,  R2C_Float_Trunc, false,  float,  cfloat, string(TEST_DIR"/signal/fft2_r2c_trunc.test"), 16, 16);
    INSTANTIATE_TEST_TP(fft2, R2C_Double_Trunc, false, double, cdouble, string(TEST_DIR"/signal/fft2_r2c_trunc.test"), 16, 16);

    for (size_t testId=0; testId<tests.size(); ++testId)
        if (tests[testId].joinable())
            tests[testId].join();
}

TEST(Threading, FFT_C2C)
{
    cleanSlate(); // Clean up everything done so far

    vector<std::thread> tests;

    int numDevices = 1;
    ASSERT_EQ(AF_SUCCESS, af_get_device_count(&numDevices));

    // complex to complex transforms
    INSTANTIATE_TEST(fft ,  C2C_Float, false,  cfloat,  cfloat, string(TEST_DIR"/signal/fft_c2c.test") );
    INSTANTIATE_TEST(fft , C2C_Double, false, cdouble, cdouble, string(TEST_DIR"/signal/fft_c2c.test") );
    INSTANTIATE_TEST(fft2,  C2C_Float, false,  cfloat,  cfloat, string(TEST_DIR"/signal/fft2_c2c.test"));
    INSTANTIATE_TEST(fft2, C2C_Double, false, cdouble, cdouble, string(TEST_DIR"/signal/fft2_c2c.test"));
    INSTANTIATE_TEST(fft3,  C2C_Float, false,  cfloat,  cfloat, string(TEST_DIR"/signal/fft3_c2c.test"));
    INSTANTIATE_TEST(fft3, C2C_Double, false, cdouble, cdouble, string(TEST_DIR"/signal/fft3_c2c.test"));

    INSTANTIATE_TEST(fft , C2C_Float_7_11_13 , false, cfloat  , cfloat , string(TEST_DIR"/signal/fft_c2c_7_11_13.test") );
    INSTANTIATE_TEST(fft , C2C_Double_7_11_13, false, cdouble , cdouble, string(TEST_DIR"/signal/fft_c2c_7_11_13.test") );
    INSTANTIATE_TEST(fft2, C2C_Float_7_11_13 , false, cfloat  , cfloat , string(TEST_DIR"/signal/fft2_c2c_7_11_13.test") );
    INSTANTIATE_TEST(fft2, C2C_Double_7_11_13, false, cdouble , cdouble, string(TEST_DIR"/signal/fft2_c2c_7_11_13.test") );
    INSTANTIATE_TEST(fft3, C2C_Float_7_11_13 , false, cfloat  , cfloat , string(TEST_DIR"/signal/fft3_c2c_7_11_13.test") );
    INSTANTIATE_TEST(fft3, C2C_Double_7_11_13, false, cdouble , cdouble, string(TEST_DIR"/signal/fft3_c2c_7_11_13.test") );

    // transforms on padded and truncated arrays
    INSTANTIATE_TEST_TP(fft2,  C2C_Float_Pad, false,  cfloat,  cfloat, string(TEST_DIR"/signal/fft2_c2c_pad.test"), 16, 16);
    INSTANTIATE_TEST_TP(fft2, C2C_Double_Pad, false, cdouble, cdouble, string(TEST_DIR"/signal/fft2_c2c_pad.test"), 16, 16);

    // inverse transforms
    // complex to complex transforms
    INSTANTIATE_TEST(ifft ,  C2C_Float, true,  cfloat,  cfloat, string(TEST_DIR"/signal/ifft_c2c.test") );
    INSTANTIATE_TEST(ifft , C2C_Double, true, cdouble, cdouble, string(TEST_DIR"/signal/ifft_c2c.test") );
    INSTANTIATE_TEST(ifft2,  C2C_Float, true,  cfloat,  cfloat, string(TEST_DIR"/signal/ifft2_c2c.test"));
    INSTANTIATE_TEST(ifft2, C2C_Double, true, cdouble, cdouble, string(TEST_DIR"/signal/ifft2_c2c.test"));
    INSTANTIATE_TEST(ifft3,  C2C_Float, true,  cfloat,  cfloat, string(TEST_DIR"/signal/ifft3_c2c.test"));
    INSTANTIATE_TEST(ifft3, C2C_Double, true, cdouble, cdouble, string(TEST_DIR"/signal/ifft3_c2c.test"));

    for (size_t testId=0; testId<tests.size(); ++testId)
        if (tests[testId].joinable())
            tests[testId].join();
}

TEST(Threading, FFT_ALL)
{
    cleanSlate(); // Clean up everything done so far

    vector<std::thread> tests;

    int numDevices = 1;
    ASSERT_EQ(AF_SUCCESS, af_get_device_count(&numDevices));

    // Real to complex transforms
    INSTANTIATE_TEST(fft ,  R2C_Float, false,  float,  cfloat, string(TEST_DIR"/signal/fft_r2c.test") );
    INSTANTIATE_TEST(fft , R2C_Double, false, double, cdouble, string(TEST_DIR"/signal/fft_r2c.test") );
    INSTANTIATE_TEST(fft2,  R2C_Float, false,  float,  cfloat, string(TEST_DIR"/signal/fft2_r2c.test"));
    INSTANTIATE_TEST(fft2, R2C_Double, false, double, cdouble, string(TEST_DIR"/signal/fft2_r2c.test"));
    INSTANTIATE_TEST(fft3,  R2C_Float, false,  float,  cfloat, string(TEST_DIR"/signal/fft3_r2c.test"));
    INSTANTIATE_TEST(fft3, R2C_Double, false, double, cdouble, string(TEST_DIR"/signal/fft3_r2c.test"));

    // Factors 7, 11, 13
    INSTANTIATE_TEST(fft , R2C_Float_7_11_13 , false, float  , cfloat , string(TEST_DIR"/signal/fft_r2c_7_11_13.test") );
    INSTANTIATE_TEST(fft , R2C_Double_7_11_13, false, double , cdouble, string(TEST_DIR"/signal/fft_r2c_7_11_13.test") );
    INSTANTIATE_TEST(fft2, R2C_Float_7_11_13 , false, float  , cfloat , string(TEST_DIR"/signal/fft2_r2c_7_11_13.test") );
    INSTANTIATE_TEST(fft2, R2C_Double_7_11_13, false, double , cdouble, string(TEST_DIR"/signal/fft2_r2c_7_11_13.test") );
    INSTANTIATE_TEST(fft3, R2C_Float_7_11_13 , false, float  , cfloat , string(TEST_DIR"/signal/fft3_r2c_7_11_13.test") );
    INSTANTIATE_TEST(fft3, R2C_Double_7_11_13, false, double , cdouble, string(TEST_DIR"/signal/fft3_r2c_7_11_13.test") );

    // transforms on padded and truncated arrays
    INSTANTIATE_TEST_TP(fft2,  R2C_Float_Trunc, false,  float,  cfloat, string(TEST_DIR"/signal/fft2_r2c_trunc.test"), 16, 16);
    INSTANTIATE_TEST_TP(fft2, R2C_Double_Trunc, false, double, cdouble, string(TEST_DIR"/signal/fft2_r2c_trunc.test"), 16, 16);

    // complex to complex transforms
    INSTANTIATE_TEST(fft ,  C2C_Float, false,  cfloat,  cfloat, string(TEST_DIR"/signal/fft_c2c.test") );
    INSTANTIATE_TEST(fft , C2C_Double, false, cdouble, cdouble, string(TEST_DIR"/signal/fft_c2c.test") );
    INSTANTIATE_TEST(fft2,  C2C_Float, false,  cfloat,  cfloat, string(TEST_DIR"/signal/fft2_c2c.test"));
    INSTANTIATE_TEST(fft2, C2C_Double, false, cdouble, cdouble, string(TEST_DIR"/signal/fft2_c2c.test"));
    INSTANTIATE_TEST(fft3,  C2C_Float, false,  cfloat,  cfloat, string(TEST_DIR"/signal/fft3_c2c.test"));
    INSTANTIATE_TEST(fft3, C2C_Double, false, cdouble, cdouble, string(TEST_DIR"/signal/fft3_c2c.test"));

    INSTANTIATE_TEST(fft , C2C_Float_7_11_13 , false, cfloat  , cfloat , string(TEST_DIR"/signal/fft_c2c_7_11_13.test") );
    INSTANTIATE_TEST(fft , C2C_Double_7_11_13, false, cdouble , cdouble, string(TEST_DIR"/signal/fft_c2c_7_11_13.test") );
    INSTANTIATE_TEST(fft2, C2C_Float_7_11_13 , false, cfloat  , cfloat , string(TEST_DIR"/signal/fft2_c2c_7_11_13.test") );
    INSTANTIATE_TEST(fft2, C2C_Double_7_11_13, false, cdouble , cdouble, string(TEST_DIR"/signal/fft2_c2c_7_11_13.test") );
    INSTANTIATE_TEST(fft3, C2C_Float_7_11_13 , false, cfloat  , cfloat , string(TEST_DIR"/signal/fft3_c2c_7_11_13.test") );
    INSTANTIATE_TEST(fft3, C2C_Double_7_11_13, false, cdouble , cdouble, string(TEST_DIR"/signal/fft3_c2c_7_11_13.test") );

    // transforms on padded and truncated arrays
    INSTANTIATE_TEST_TP(fft2,  C2C_Float_Pad, false,  cfloat,  cfloat, string(TEST_DIR"/signal/fft2_c2c_pad.test"), 16, 16);
    INSTANTIATE_TEST_TP(fft2, C2C_Double_Pad, false, cdouble, cdouble, string(TEST_DIR"/signal/fft2_c2c_pad.test"), 16, 16);

    // inverse transforms
    // complex to complex transforms
    INSTANTIATE_TEST(ifft ,  C2C_Float, true,  cfloat,  cfloat, string(TEST_DIR"/signal/ifft_c2c.test") );
    INSTANTIATE_TEST(ifft , C2C_Double, true, cdouble, cdouble, string(TEST_DIR"/signal/ifft_c2c.test") );
    INSTANTIATE_TEST(ifft2,  C2C_Float, true,  cfloat,  cfloat, string(TEST_DIR"/signal/ifft2_c2c.test"));
    INSTANTIATE_TEST(ifft2, C2C_Double, true, cdouble, cdouble, string(TEST_DIR"/signal/ifft2_c2c.test"));
    INSTANTIATE_TEST(ifft3,  C2C_Float, true,  cfloat,  cfloat, string(TEST_DIR"/signal/ifft3_c2c.test"));
    INSTANTIATE_TEST(ifft3, C2C_Double, true, cdouble, cdouble, string(TEST_DIR"/signal/ifft3_c2c.test"));

    for (size_t testId=0; testId<tests.size(); ++testId)
        if (tests[testId].joinable())
            tests[testId].join();
}

template<typename T, bool isBVector>
void cppMatMulCheck(int targetDevice, string TestFile)
{
    if (noDoubleTests<T>()) return;

    using std::vector;
    vector<af::dim4> numDims;

    vector<vector<T> > hData;
    vector<vector<T> > tests;
    readTests<T,T,int>(TestFile, numDims, hData, tests);

    af::setDevice(targetDevice);

    af::array a(numDims[0], &hData[0].front());
    af::array b(numDims[1], &hData[1].front());

    af::dim4 atdims = numDims[0];
    {
        dim_t f  =    atdims[0];
        atdims[0]   =    atdims[1];
        atdims[1]   =    f;
    }
    af::dim4 btdims = numDims[1];
    {
        dim_t f = btdims[0];
        btdims[0] = btdims[1];
        btdims[1] = f;
    }

    af::array aT = moddims(a, atdims.ndims(), atdims.get());
    af::array bT = moddims(b, btdims.ndims(), btdims.get());

    vector<af::array> out(tests.size());
    if(isBVector) {
        out[0] = af::matmul(aT, b,    AF_MAT_NONE,    AF_MAT_NONE);
        out[1] = af::matmul(bT, a,   AF_MAT_NONE,    AF_MAT_NONE);
        out[2] = af::matmul(b, a,    AF_MAT_TRANS,       AF_MAT_NONE);
        out[3] = af::matmul(bT, aT,   AF_MAT_NONE,    AF_MAT_TRANS);
        out[4] = af::matmul(b, aT,    AF_MAT_TRANS,       AF_MAT_TRANS);
    }
    else {
        out[0] = af::matmul(a, b, AF_MAT_NONE,   AF_MAT_NONE);
        out[1] = af::matmul(a, bT, AF_MAT_NONE,   AF_MAT_TRANS);
        out[2] = af::matmul(a, bT, AF_MAT_TRANS,      AF_MAT_NONE);
        out[3] = af::matmul(aT, bT, AF_MAT_TRANS,      AF_MAT_TRANS);
    }

    for(size_t i = 0; i < tests.size(); i++) {
        dim_t elems = out[i].elements();
        vector<T> h_out(elems);
        out[i].host((void*)&h_out.front());

        if (false == equal(h_out.begin(), h_out.end(), tests[i].begin())) {

            std::cout << "Failed test " << i << "\nCalculated: " << std::endl;
            std::copy(h_out.begin(), h_out.end(), std::ostream_iterator<T>(std::cout, ", "));
            std::cout << "Expected: " << std::endl;
            std::copy(tests[i].begin(), tests[i].end(), std::ostream_iterator<T>(std::cout, ", "));
            FAIL();
        }
    }
}

#define TEST_FOR_TYPE(TypeName)                                   \
    tests.emplace_back(cppMatMulCheck<TypeName, false>,                             \
            nextTargetDeviceId()%numDevices, TEST_DIR "/blas/Basic.test");          \
    tests.emplace_back(cppMatMulCheck<TypeName, false>,                             \
            nextTargetDeviceId()%numDevices, TEST_DIR "/blas/NonSquare.test");      \
    tests.emplace_back(cppMatMulCheck<TypeName, true>,                              \
            nextTargetDeviceId()%numDevices, TEST_DIR "/blas/SquareVector.test");   \
    tests.emplace_back(cppMatMulCheck<TypeName, true>,                              \
            nextTargetDeviceId()%numDevices, TEST_DIR "/blas/RectangleVector.test");

TEST(Threading, BLAS)
{
    cleanSlate(); // Clean up everything done so far

    vector<std::thread> tests;

    int numDevices = 1;
    ASSERT_EQ(AF_SUCCESS, af_get_device_count(&numDevices));

    TEST_FOR_TYPE(      float);
    TEST_FOR_TYPE( af::cfloat);
    TEST_FOR_TYPE(     double);
    TEST_FOR_TYPE(af::cdouble);

    for (size_t testId=0; testId<tests.size(); ++testId)
        if (tests[testId].joinable())
            tests[testId].join();
}

#if !defined(AF_OPENCL)

#define SOLVE_LU_TESTS(T, eps)                                                                          \
    tests.emplace_back(solveLUTester<T>, 1000, 100, eps, nextTargetDeviceId()%numDevices);              \
    tests.emplace_back(solveLUTester<T>, 2048, 512, eps, nextTargetDeviceId()%numDevices);              \
    std::this_thread::sleep_for(std::chrono::seconds(2));   \
    tests.emplace_back(solveTriangleTester<T>, 1000, 100, true, eps, nextTargetDeviceId()%numDevices);  \
    tests.emplace_back(solveTriangleTester<T>, 2048, 512, true, eps, nextTargetDeviceId()%numDevices);  \
    std::this_thread::sleep_for(std::chrono::seconds(2));   \
    tests.emplace_back(solveTriangleTester<T>, 1000, 100, false, eps, nextTargetDeviceId()%numDevices); \
    tests.emplace_back(solveTriangleTester<T>, 2048, 512, false, eps, nextTargetDeviceId()%numDevices); \
    std::this_thread::sleep_for(std::chrono::seconds(2));   \
    tests.emplace_back(solveTester<T>, 1000, 1000, 100, eps, nextTargetDeviceId()%numDevices);          \
    tests.emplace_back(solveTester<T>, 2048, 2048, 512, eps, nextTargetDeviceId()%numDevices);          \
    std::this_thread::sleep_for(std::chrono::seconds(2));   \
    tests.emplace_back(solveTester<T>, 800, 1000, 200, eps, nextTargetDeviceId()%numDevices);           \
    tests.emplace_back(solveTester<T>, 1536, 2048, 400, eps, nextTargetDeviceId()%numDevices);          \
    std::this_thread::sleep_for(std::chrono::seconds(2));   \
    tests.emplace_back(solveTester<T>, 800, 600, 64, eps, nextTargetDeviceId()%numDevices);             \
    tests.emplace_back(solveTester<T>, 1536, 1024, 1, eps, nextTargetDeviceId()%numDevices);

// Added 2s sleep for every two test threads to make sure
// we are not running out of memory.
TEST(Threading, SolveDense)
{
    cleanSlate(); // Clean up everything done so far

    vector<std::thread> tests;

    int numDevices = 1;
    ASSERT_EQ(AF_SUCCESS, af_get_device_count(&numDevices));

    SOLVE_LU_TESTS(float, 0.01);
    SOLVE_LU_TESTS(double, 1E-5);
    SOLVE_LU_TESTS(cfloat, 0.01);
    SOLVE_LU_TESTS(cdouble, 1E-5);

    for (size_t testId=0; testId<tests.size(); ++testId)
        if (tests[testId].joinable())
            tests[testId].join();
}

#undef SOLVE_LU_TESTS

#define SPARSE_TESTS(T, eps)                                                    \
        tests.emplace_back(sparseTester<T>, 1000, 1000, 100, 5, eps);           \
        tests.emplace_back(sparseTester<T>, 2048, 1024, 512, 3, eps);           \
        tests.emplace_back(sparseTester<T>, 500, 1000, 250, 1, eps);            \
        tests.emplace_back(sparseTester<T>, 625, 1331, 1, 2, eps);              \
        tests.emplace_back(sparseTransposeTester<T>, 625, 1331, 1, 2, eps);     \
        tests.emplace_back(sparseTransposeTester<T>, 1000, 1000, 100, 5, eps);  \
        tests.emplace_back(sparseTransposeTester<T>, 2048, 1024, 512, 3, eps);  \
        tests.emplace_back(sparseTransposeTester<T>, 453, 751, 397, 1, eps);    \
        tests.emplace_back(convertCSR<T>, 2345, 5678, 0.5);                     \
        std::this_thread::sleep_for(std::chrono::seconds(5));

// Added 2s sleep for every two test threads to make sure
// we are not running out of memory.
TEST(Threading, Sparse)
{
    cleanSlate(); // Clean up everything done so far

    vector<std::thread> tests;

    int numDevices = 1;
    ASSERT_EQ(AF_SUCCESS, af_get_device_count(&numDevices));

    SPARSE_TESTS(  float, 1E-3);
    SPARSE_TESTS( double, 1E-5);
    SPARSE_TESTS( cfloat, 1E-3);
    SPARSE_TESTS(cdouble, 1E-5);

    for (size_t testId=0; testId<tests.size(); ++testId)
        if (tests[testId].joinable())
            tests[testId].join();
}

#endif
