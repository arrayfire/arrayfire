/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <sparse_common.hpp>
#include <testHelpers.hpp>
#include <af/traits.hpp>
#include <chrono>
#include <complex>
#include <condition_variable>
#include <cstddef>
#include <iterator>
#include <thread>
#include <vector>

using namespace af;

using std::cout;
using std::endl;
using std::string;
using std::vector;

static const int THREAD_COUNT = 32;

#if defined(AF_CPU)
static const unsigned ITERATION_COUNT = 10;
#else
static const unsigned ITERATION_COUNT = 1000;
#endif

enum ArithOp { ADD, SUB, DIV, MUL };

void calc(ArithOp opcode, array op1, array op2, float outValue,
          int iteration_count) {
    setDevice(0);
    array res;
    for (int i = 0; i < iteration_count; ++i) {
        switch (opcode) {
            case ADD: res = op1 + op2; break;
            case SUB: res = op1 - op2; break;
            case DIV: res = op1 / op2; break;
            case MUL: res = op1 * op2; break;
        }
    }

    vector<float> out(res.elements());
    res.host((void*)out.data());

    for (unsigned i = 0; i < out.size(); ++i) ASSERT_FLOAT_EQ(out[i], outValue);
    af::sync();
}

TEST(Threading, SimultaneousRead) {
    setDevice(0);

    array A = constant(1.0, 100, 100);
    array B = constant(1.0, 100, 100);

    vector<std::thread> tests;

    int thread_count    = 8;
    int iteration_count = 30;
    for (int t = 0; t < thread_count; ++t) {
        ArithOp op;
        float outValue;

        switch (t % 4) {
            case 0:
                op       = ADD;
                outValue = 2.0f;
                break;
            case 1:
                op       = SUB;
                outValue = 0.0f;
                break;
            case 2:
                op       = DIV;
                outValue = 1.0f;
                break;
            case 3:
                op       = MUL;
                outValue = 1.0f;
                break;
        }

        tests.emplace_back(calc, op, A, B, outValue, iteration_count);
    }

    for (int t = 0; t < thread_count; ++t)
        if (tests[t].joinable()) tests[t].join();
}

std::condition_variable cv;
std::mutex cvMutex;
size_t counter = THREAD_COUNT;

void doubleAllocationTest() {
    setDevice(0);

    // Block until all threads are launched and the
    // counter variable hits zero
    std::unique_lock<std::mutex> lock(cvMutex);
    // Check for current thread launch counter value
    // if reached zero, notify others to continue
    // otherwise block current thread
    if (--counter == 0)
        cv.notify_all();
    else
        cv.wait(lock, [] { return counter == 0; });
    lock.unlock();

    array a = randu(5, 5);

    // Wait for for other threads to hit randu call
    // while this thread's variable a is still in scope.
    std::this_thread::sleep_for(std::chrono::seconds(2));
}

int nextTargetDeviceId() {
    static int nextId = 0;
    return nextId++;
}

void morphTest(const array input, const array mask, const bool isDilation,
               const array gold, int targetDevice) {
    setDevice(targetDevice);

    array out;

    try {
        for (unsigned i = 0; i < ITERATION_COUNT; ++i)
            out = isDilation ? dilate(input, mask) : erode(input, mask);
    } catch FUNCTION_UNSUPPORTED

    ASSERT_IMAGES_NEAR(gold, out, 0.018f);
}

TEST(Threading, SetPerThreadActiveDevice) {
    IMAGEIO_ENABLED_CHECK();

    vector<bool> isDilationFlags;
    vector<bool> isColorFlags;
    vector<string> files;

    files.push_back(string(TEST_DIR "/morph/gray.test"));
    isDilationFlags.push_back(true);
    isColorFlags.push_back(false);

    files.push_back(string(TEST_DIR "/morph/color.test"));
    isDilationFlags.push_back(false);
    isColorFlags.push_back(true);

    vector<std::thread> tests;
    unsigned totalTestCount = 0;

    for (size_t pos = 0; pos < files.size(); ++pos) {
        const bool isDilation = isDilationFlags[pos];
        const bool isColor    = isColorFlags[pos];

        vector<dim4> inDims;
        vector<string> inFiles;
        vector<dim_t> outSizes;
        vector<string> outFiles;

        readImageTests(files[pos], inDims, inFiles, outSizes, outFiles);

        const unsigned testCount = inDims.size();

        const dim4 maskdims(3, 3, 1, 1);

        for (size_t testId = 0; testId < testCount; ++testId) {
            int trgtDeviceId = totalTestCount % getDeviceCount();

            // prefix full path to image file names
            inFiles[testId].insert(0, string(TEST_DIR "/morph/"));
            outFiles[testId].insert(0, string(TEST_DIR "/morph/"));

            setDevice(trgtDeviceId);

            const array mask = constant(1.0, maskdims);

            array input = loadImage(inFiles[testId].c_str(), isColor);
            array gold  = loadImage(outFiles[testId].c_str(), isColor);

            // Push the new test as a new thread of execution
            tests.emplace_back(morphTest, input, mask, isDilation, gold,
                               trgtDeviceId);

            totalTestCount++;
        }
    }

    for (size_t testId = 0; testId < tests.size(); ++testId)
        if (tests[testId].joinable()) tests[testId].join();
}

TEST(Threading, MemoryManagementScope) {
    setDevice(0);
    cleanSlate();  // Clean up everything done so far

    vector<std::thread> tests;

    for (int t = 0; t < THREAD_COUNT; ++t)
        tests.emplace_back(doubleAllocationTest);

    for (int t = 0; t < THREAD_COUNT; ++t)
        if (tests[t].joinable()) tests[t].join();

    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(lock_buffers, 0u);
    ASSERT_EQ(lock_bytes, 0u);
    ASSERT_EQ(alloc_buffers, 32u);
    ASSERT_EQ(alloc_bytes, 32768u);
}

void jitAllocationTest() {
    setDevice(0);

    for (int i = 0; i < 100; ++i) array a = constant(1, 5, 5);
}

TEST(Threading, MemoryManagement_JIT_Node) {
    cleanSlate();  // Clean up everything done so far

    vector<std::thread> tests;

    for (int t = 0; t < THREAD_COUNT; ++t)
        tests.emplace_back(jitAllocationTest);

    for (int t = 0; t < THREAD_COUNT; ++t)
        if (tests[t].joinable()) tests[t].join();

    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 0u);
    ASSERT_EQ(lock_buffers, 0u);
    ASSERT_EQ(alloc_bytes, 0u);
    ASSERT_EQ(lock_bytes, 0u);
}

template<typename inType, typename outType, bool isInverse>
void fftTest(int targetDevice, string pTestFile, dim_t pad0 = 0, dim_t pad1 = 0,
             dim_t pad2 = 0) {
    SUPPORTED_TYPE_CHECK(inType);
    SUPPORTED_TYPE_CHECK(outType);

    vector<dim4> numDims;
    vector<vector<inType>> in;
    vector<vector<outType>> tests;

    readTestsFromFile<inType, outType>(pTestFile, numDims, in, tests);

    dim4 dims         = numDims[0];
    af_array outArray = 0;
    af_array inArray  = 0;

    ASSERT_SUCCESS(af_set_device(targetDevice));

    ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<inType>::af_type));

    if (isInverse) {
        switch (dims.ndims()) {
            case 1:
                ASSERT_SUCCESS(af_ifft(&outArray, inArray, 1.0, pad0));
                break;
            case 2:
                ASSERT_SUCCESS(af_ifft2(&outArray, inArray, 1.0, pad0, pad1));
                break;
            case 3:
                ASSERT_SUCCESS(
                    af_ifft3(&outArray, inArray, 1.0, pad0, pad1, pad2));
                break;
            default:
                throw std::runtime_error(
                    "This error shouldn't happen, pls check");
        }
    } else {
        switch (dims.ndims()) {
            case 1:
                ASSERT_SUCCESS(af_fft(&outArray, inArray, 1.0, pad0));
                break;
            case 2:
                ASSERT_SUCCESS(af_fft2(&outArray, inArray, 1.0, pad0, pad1));
                break;
            case 3:
                ASSERT_SUCCESS(
                    af_fft3(&outArray, inArray, 1.0, pad0, pad1, pad2));
                break;
            default:
                throw std::runtime_error(
                    "This error shouldn't happen, pls check");
        }
    }

    size_t out_size  = tests[0].size();
    outType* outData = new outType[out_size];
    ASSERT_SUCCESS(af_get_data_ptr((void*)outData, outArray));

    vector<outType> goldBar(tests[0].begin(), tests[0].end());

    size_t test_size = 0;
    switch (dims.ndims()) {
        case 1: test_size = dims[0] / 2 + 1; break;
        case 2: test_size = dims[1] * (dims[0] / 2 + 1); break;
        case 3: test_size = dims[2] * dims[1] * (dims[0] / 2 + 1); break;
        default: test_size = dims[0] / 2 + 1; break;
    }
    outType output_scale = (outType)(isInverse ? test_size : 1);
    for (size_t elIter = 0; elIter < test_size; ++elIter) {
        bool isUnderTolerance = abs(goldBar[elIter] - outData[elIter]) < 0.001;
        ASSERT_EQ(true, isUnderTolerance)
            << "Expected value=" << goldBar[elIter]
            << "\t Actual Value=" << (output_scale * outData[elIter])
            << " at: " << elIter
            << " from thread: " << std::this_thread::get_id() << endl;
    }

    // cleanup
    delete[] outData;
    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

#define INSTANTIATE_TEST(func, name, is_inverse, in_t, out_t, file)        \
    {                                                                      \
        int targetDevice = nextTargetDeviceId() % numDevices;              \
        tests.emplace_back(fftTest<in_t, out_t, is_inverse>, targetDevice, \
                           file, 0, 0, 0);                                 \
    }

#define INSTANTIATE_TEST_TP(func, name, is_inverse, in_t, out_t, file, p0, p1) \
    {                                                                          \
        int targetDevice = nextTargetDeviceId() % numDevices;                  \
        tests.emplace_back(fftTest<in_t, out_t, is_inverse>, targetDevice,     \
                           file, p0, p1, 0);                                   \
    }

TEST(Threading, FFT_R2C) {
    cleanSlate();  // Clean up everything done so far

    vector<std::thread> tests;

    int numDevices = 0;
    ASSERT_SUCCESS(af_get_device_count(&numDevices));
    ASSERT_EQ(true, numDevices > 0);

    // Real to complex transforms
    INSTANTIATE_TEST(fft, R2C_Float, false, float, cfloat,
                     string(TEST_DIR "/signal/fft_r2c.test"));
    INSTANTIATE_TEST(fft2, R2C_Float, false, float, cfloat,
                     string(TEST_DIR "/signal/fft2_r2c.test"));
    INSTANTIATE_TEST(fft3, R2C_Float, false, float, cfloat,
                     string(TEST_DIR "/signal/fft3_r2c.test"));

    // Factors 7, 11, 13
    INSTANTIATE_TEST(fft, R2C_Float_7_11_13, false, float, cfloat,
                     string(TEST_DIR "/signal/fft_r2c_7_11_13.test"));
    INSTANTIATE_TEST(fft2, R2C_Float_7_11_13, false, float, cfloat,
                     string(TEST_DIR "/signal/fft2_r2c_7_11_13.test"));
    INSTANTIATE_TEST(fft3, R2C_Float_7_11_13, false, float, cfloat,
                     string(TEST_DIR "/signal/fft3_r2c_7_11_13.test"));

    // transforms on padded and truncated arrays
    INSTANTIATE_TEST_TP(fft2, R2C_Float_Trunc, false, float, cfloat,
                        string(TEST_DIR "/signal/fft2_r2c_trunc.test"), 16, 16);

    if (noDoubleTests(f64)) {
        // Real to complex transforms
        INSTANTIATE_TEST(fft, R2C_Double, false, double, cdouble,
                         string(TEST_DIR "/signal/fft_r2c.test"));
        INSTANTIATE_TEST(fft2, R2C_Double, false, double, cdouble,
                         string(TEST_DIR "/signal/fft2_r2c.test"));
        INSTANTIATE_TEST(fft3, R2C_Double, false, double, cdouble,
                         string(TEST_DIR "/signal/fft3_r2c.test"));

        // Factors 7, 11, 13
        INSTANTIATE_TEST(fft, R2C_Double_7_11_13, false, double, cdouble,
                         string(TEST_DIR "/signal/fft_r2c_7_11_13.test"));
        INSTANTIATE_TEST(fft2, R2C_Double_7_11_13, false, double, cdouble,
                         string(TEST_DIR "/signal/fft2_r2c_7_11_13.test"));
        INSTANTIATE_TEST(fft3, R2C_Double_7_11_13, false, double, cdouble,
                         string(TEST_DIR "/signal/fft3_r2c_7_11_13.test"));

        // transforms on padded and truncated arrays
        INSTANTIATE_TEST_TP(fft2, R2C_Double_Trunc, false, double, cdouble,
                            string(TEST_DIR "/signal/fft2_r2c_trunc.test"), 16,
                            16);
    }

    for (size_t testId = 0; testId < tests.size(); ++testId)
        if (tests[testId].joinable()) tests[testId].join();
}

TEST(Threading, FFT_C2C) {
    cleanSlate();  // Clean up everything done so far

    vector<std::thread> tests;

    int numDevices = 0;
    ASSERT_SUCCESS(af_get_device_count(&numDevices));
    ASSERT_EQ(true, numDevices > 0);

    // complex to complex transforms
    INSTANTIATE_TEST(fft, C2C_Float, false, cfloat, cfloat,
                     string(TEST_DIR "/signal/fft_c2c.test"));
    INSTANTIATE_TEST(fft2, C2C_Float, false, cfloat, cfloat,
                     string(TEST_DIR "/signal/fft2_c2c.test"));
    INSTANTIATE_TEST(fft3, C2C_Float, false, cfloat, cfloat,
                     string(TEST_DIR "/signal/fft3_c2c.test"));

    INSTANTIATE_TEST(fft, C2C_Float_7_11_13, false, cfloat, cfloat,
                     string(TEST_DIR "/signal/fft_c2c_7_11_13.test"));
    INSTANTIATE_TEST(fft2, C2C_Float_7_11_13, false, cfloat, cfloat,
                     string(TEST_DIR "/signal/fft2_c2c_7_11_13.test"));
    INSTANTIATE_TEST(fft3, C2C_Float_7_11_13, false, cfloat, cfloat,
                     string(TEST_DIR "/signal/fft3_c2c_7_11_13.test"));

    // transforms on padded and truncated arrays
    INSTANTIATE_TEST_TP(fft2, C2C_Float_Pad, false, cfloat, cfloat,
                        string(TEST_DIR "/signal/fft2_c2c_pad.test"), 16, 16);

    // inverse transforms
    // complex to complex transforms
    INSTANTIATE_TEST(ifft, C2C_Float, true, cfloat, cfloat,
                     string(TEST_DIR "/signal/ifft_c2c.test"));
    INSTANTIATE_TEST(ifft2, C2C_Float, true, cfloat, cfloat,
                     string(TEST_DIR "/signal/ifft2_c2c.test"));
    INSTANTIATE_TEST(ifft3, C2C_Float, true, cfloat, cfloat,
                     string(TEST_DIR "/signal/ifft3_c2c.test"));

    if (noDoubleTests(f64)) {
        INSTANTIATE_TEST(fft, C2C_Double, false, cdouble, cdouble,
                         string(TEST_DIR "/signal/fft_c2c.test"));
        INSTANTIATE_TEST(fft2, C2C_Double, false, cdouble, cdouble,
                         string(TEST_DIR "/signal/fft2_c2c.test"));
        INSTANTIATE_TEST(fft3, C2C_Double, false, cdouble, cdouble,
                         string(TEST_DIR "/signal/fft3_c2c.test"));

        INSTANTIATE_TEST(fft, C2C_Double_7_11_13, false, cdouble, cdouble,
                         string(TEST_DIR "/signal/fft_c2c_7_11_13.test"));
        INSTANTIATE_TEST(fft2, C2C_Double_7_11_13, false, cdouble, cdouble,
                         string(TEST_DIR "/signal/fft2_c2c_7_11_13.test"));
        INSTANTIATE_TEST(fft3, C2C_Double_7_11_13, false, cdouble, cdouble,
                         string(TEST_DIR "/signal/fft3_c2c_7_11_13.test"));

        INSTANTIATE_TEST_TP(fft2, C2C_Double_Pad, false, cdouble, cdouble,
                            string(TEST_DIR "/signal/fft2_c2c_pad.test"), 16,
                            16);

        INSTANTIATE_TEST(ifft, C2C_Double, true, cdouble, cdouble,
                         string(TEST_DIR "/signal/ifft_c2c.test"));
        INSTANTIATE_TEST(ifft2, C2C_Double, true, cdouble, cdouble,
                         string(TEST_DIR "/signal/ifft2_c2c.test"));
        INSTANTIATE_TEST(ifft3, C2C_Double, true, cdouble, cdouble,
                         string(TEST_DIR "/signal/ifft3_c2c.test"));
    }

    for (size_t testId = 0; testId < tests.size(); ++testId)
        if (tests[testId].joinable()) tests[testId].join();
}

TEST(Threading, FFT_ALL) {
    cleanSlate();  // Clean up everything done so far

    vector<std::thread> tests;

    int numDevices = 0;
    ASSERT_SUCCESS(af_get_device_count(&numDevices));
    ASSERT_EQ(true, numDevices > 0);

    // Real to complex transforms
    INSTANTIATE_TEST(fft, R2C_Float, false, float, cfloat,
                     string(TEST_DIR "/signal/fft_r2c.test"));
    INSTANTIATE_TEST(fft2, R2C_Float, false, float, cfloat,
                     string(TEST_DIR "/signal/fft2_r2c.test"));
    INSTANTIATE_TEST(fft3, R2C_Float, false, float, cfloat,
                     string(TEST_DIR "/signal/fft3_r2c.test"));

    // Factors 7, 11, 13
    INSTANTIATE_TEST(fft, R2C_Float_7_11_13, false, float, cfloat,
                     string(TEST_DIR "/signal/fft_r2c_7_11_13.test"));
    INSTANTIATE_TEST(fft2, R2C_Float_7_11_13, false, float, cfloat,
                     string(TEST_DIR "/signal/fft2_r2c_7_11_13.test"));
    INSTANTIATE_TEST(fft3, R2C_Float_7_11_13, false, float, cfloat,
                     string(TEST_DIR "/signal/fft3_r2c_7_11_13.test"));

    // transforms on padded and truncated arrays
    INSTANTIATE_TEST_TP(fft2, R2C_Float_Trunc, false, float, cfloat,
                        string(TEST_DIR "/signal/fft2_r2c_trunc.test"), 16, 16);

    // complex to complex transforms
    INSTANTIATE_TEST(fft, C2C_Float, false, cfloat, cfloat,
                     string(TEST_DIR "/signal/fft_c2c.test"));
    INSTANTIATE_TEST(fft2, C2C_Float, false, cfloat, cfloat,
                     string(TEST_DIR "/signal/fft2_c2c.test"));
    INSTANTIATE_TEST(fft3, C2C_Float, false, cfloat, cfloat,
                     string(TEST_DIR "/signal/fft3_c2c.test"));

    INSTANTIATE_TEST(fft, C2C_Float_7_11_13, false, cfloat, cfloat,
                     string(TEST_DIR "/signal/fft_c2c_7_11_13.test"));
    INSTANTIATE_TEST(fft2, C2C_Float_7_11_13, false, cfloat, cfloat,
                     string(TEST_DIR "/signal/fft2_c2c_7_11_13.test"));
    INSTANTIATE_TEST(fft3, C2C_Float_7_11_13, false, cfloat, cfloat,
                     string(TEST_DIR "/signal/fft3_c2c_7_11_13.test"));

    // transforms on padded and truncated arrays
    INSTANTIATE_TEST_TP(fft2, C2C_Float_Pad, false, cfloat, cfloat,
                        string(TEST_DIR "/signal/fft2_c2c_pad.test"), 16, 16);

    // inverse transforms
    // complex to complex transforms
    INSTANTIATE_TEST(ifft, C2C_Float, true, cfloat, cfloat,
                     string(TEST_DIR "/signal/ifft_c2c.test"));
    INSTANTIATE_TEST(ifft2, C2C_Float, true, cfloat, cfloat,
                     string(TEST_DIR "/signal/ifft2_c2c.test"));
    INSTANTIATE_TEST(ifft3, C2C_Float, true, cfloat, cfloat,
                     string(TEST_DIR "/signal/ifft3_c2c.test"));

    if (noDoubleTests(f64)) {
        INSTANTIATE_TEST(fft, R2C_Double, false, double, cdouble,
                         string(TEST_DIR "/signal/fft_r2c.test"));
        INSTANTIATE_TEST(fft2, R2C_Double, false, double, cdouble,
                         string(TEST_DIR "/signal/fft2_r2c.test"));
        INSTANTIATE_TEST(fft3, R2C_Double, false, double, cdouble,
                         string(TEST_DIR "/signal/fft3_r2c.test"));
        INSTANTIATE_TEST(fft, R2C_Double_7_11_13, false, double, cdouble,
                         string(TEST_DIR "/signal/fft_r2c_7_11_13.test"));
        INSTANTIATE_TEST(fft2, R2C_Double_7_11_13, false, double, cdouble,
                         string(TEST_DIR "/signal/fft2_r2c_7_11_13.test"));
        INSTANTIATE_TEST(fft3, R2C_Double_7_11_13, false, double, cdouble,
                         string(TEST_DIR "/signal/fft3_r2c_7_11_13.test"));
        INSTANTIATE_TEST_TP(fft2, R2C_Double_Trunc, false, double, cdouble,
                            string(TEST_DIR "/signal/fft2_r2c_trunc.test"), 16,
                            16);
        INSTANTIATE_TEST(fft, C2C_Double, false, cdouble, cdouble,
                         string(TEST_DIR "/signal/fft_c2c.test"));
        INSTANTIATE_TEST(fft2, C2C_Double, false, cdouble, cdouble,
                         string(TEST_DIR "/signal/fft2_c2c.test"));
        INSTANTIATE_TEST(fft3, C2C_Double, false, cdouble, cdouble,
                         string(TEST_DIR "/signal/fft3_c2c.test"));
        INSTANTIATE_TEST(fft, C2C_Double_7_11_13, false, cdouble, cdouble,
                         string(TEST_DIR "/signal/fft_c2c_7_11_13.test"));
        INSTANTIATE_TEST(fft2, C2C_Double_7_11_13, false, cdouble, cdouble,
                         string(TEST_DIR "/signal/fft2_c2c_7_11_13.test"));
        INSTANTIATE_TEST(fft3, C2C_Double_7_11_13, false, cdouble, cdouble,
                         string(TEST_DIR "/signal/fft3_c2c_7_11_13.test"));
        INSTANTIATE_TEST_TP(fft2, C2C_Double_Pad, false, cdouble, cdouble,
                            string(TEST_DIR "/signal/fft2_c2c_pad.test"), 16,
                            16);
        INSTANTIATE_TEST(ifft, C2C_Double, true, cdouble, cdouble,
                         string(TEST_DIR "/signal/ifft_c2c.test"));
        INSTANTIATE_TEST(ifft2, C2C_Double, true, cdouble, cdouble,
                         string(TEST_DIR "/signal/ifft2_c2c.test"));
        INSTANTIATE_TEST(ifft3, C2C_Double, true, cdouble, cdouble,
                         string(TEST_DIR "/signal/ifft3_c2c.test"));
    }

    for (size_t testId = 0; testId < tests.size(); ++testId)
        if (tests[testId].joinable()) tests[testId].join();
}

template<typename T, bool isBVector>
void cppMatMulCheck(int targetDevice, string TestFile) {
    SUPPORTED_TYPE_CHECK(T);

    using std::vector;
    vector<dim4> numDims;

    vector<vector<T>> hData;
    vector<vector<T>> tests;
    readTests<T, T, int>(TestFile, numDims, hData, tests);

    setDevice(targetDevice);

    array a(numDims[0], &hData[0].front());
    array b(numDims[1], &hData[1].front());

    dim4 atdims = numDims[0];
    {
        dim_t f   = atdims[0];
        atdims[0] = atdims[1];
        atdims[1] = f;
    }
    dim4 btdims = numDims[1];
    {
        dim_t f   = btdims[0];
        btdims[0] = btdims[1];
        btdims[1] = f;
    }

    array aT = moddims(a, atdims.ndims(), atdims.get());
    array bT = moddims(b, btdims.ndims(), btdims.get());

    vector<array> out(tests.size());
    if (isBVector) {
        out[0] = matmul(aT, b, AF_MAT_NONE, AF_MAT_NONE);
        out[1] = matmul(bT, a, AF_MAT_NONE, AF_MAT_NONE);
        out[2] = matmul(b, a, AF_MAT_TRANS, AF_MAT_NONE);
        out[3] = matmul(bT, aT, AF_MAT_NONE, AF_MAT_TRANS);
        out[4] = matmul(b, aT, AF_MAT_TRANS, AF_MAT_TRANS);
    } else {
        out[0] = matmul(a, b, AF_MAT_NONE, AF_MAT_NONE);
        out[1] = matmul(a, bT, AF_MAT_NONE, AF_MAT_TRANS);
        out[2] = matmul(a, bT, AF_MAT_TRANS, AF_MAT_NONE);
        out[3] = matmul(aT, bT, AF_MAT_TRANS, AF_MAT_TRANS);
    }

    for (size_t i = 0; i < tests.size(); i++) {
        dim_t elems = out[i].elements();
        vector<T> h_out(elems);
        out[i].host((void*)&h_out.front());

        if (false == equal(h_out.begin(), h_out.end(), tests[i].begin())) {
            cout << "Failed test " << i << "\nCalculated: " << endl;
            std::copy(h_out.begin(), h_out.end(),
                      std::ostream_iterator<T>(cout, ", "));
            cout << "Expected: " << endl;
            std::copy(tests[i].begin(), tests[i].end(),
                      std::ostream_iterator<T>(cout, ", "));
            FAIL();
        }
    }
}

#define TEST_BLAS_FOR_TYPE(TypeName)                        \
    tests.emplace_back(cppMatMulCheck<TypeName, false>,     \
                       nextTargetDeviceId() % numDevices,   \
                       TEST_DIR "/blas/Basic.test");        \
    tests.emplace_back(cppMatMulCheck<TypeName, false>,     \
                       nextTargetDeviceId() % numDevices,   \
                       TEST_DIR "/blas/NonSquare.test");    \
    tests.emplace_back(cppMatMulCheck<TypeName, true>,      \
                       nextTargetDeviceId() % numDevices,   \
                       TEST_DIR "/blas/SquareVector.test"); \
    tests.emplace_back(cppMatMulCheck<TypeName, true>,      \
                       nextTargetDeviceId() % numDevices,   \
                       TEST_DIR "/blas/RectangleVector.test");

TEST(Threading, BLAS) {
    cleanSlate();  // Clean up everything done so far

    vector<std::thread> tests;

    int numDevices = 0;
    ASSERT_SUCCESS(af_get_device_count(&numDevices));
    ASSERT_EQ(true, numDevices > 0);

    TEST_BLAS_FOR_TYPE(float);
    TEST_BLAS_FOR_TYPE(cfloat);

    if (noDoubleTests(f64)) {
        TEST_BLAS_FOR_TYPE(double);
        TEST_BLAS_FOR_TYPE(cdouble);
    }

    for (size_t testId = 0; testId < tests.size(); ++testId)
        if (tests[testId].joinable()) tests[testId].join();
}

#define SPARSE_TESTS(T, eps)                                            \
    tests.emplace_back(sparseTester<T>, 1000, 1000, 100, 5, eps,        \
                       nextTargetDeviceId() % numDevices);              \
    tests.emplace_back(sparseTester<T>, 500, 1000, 250, 1, eps,         \
                       nextTargetDeviceId() % numDevices);              \
    tests.emplace_back(sparseTester<T>, 625, 1331, 1, 2, eps,           \
                       nextTargetDeviceId() % numDevices);              \
    tests.emplace_back(sparseTransposeTester<T>, 625, 1331, 1, 2, eps,  \
                       nextTargetDeviceId() % numDevices);              \
    tests.emplace_back(sparseTransposeTester<T>, 453, 751, 397, 1, eps, \
                       nextTargetDeviceId() % numDevices);              \
    tests.emplace_back(convertCSR<T>, 2345, 5678, 0.5,                  \
                       nextTargetDeviceId() % numDevices);

TEST(Threading, Sparse) {
    cleanSlate();  // Clean up everything done so far

    vector<std::thread> tests;

    int numDevices = 0;
    ASSERT_SUCCESS(af_get_device_count(&numDevices));
    ASSERT_EQ(true, numDevices > 0);

    SPARSE_TESTS(float, 1E-3);
    SPARSE_TESTS(cfloat, 1E-3);
    if (noDoubleTests(f64)) {
        SPARSE_TESTS(double, 1E-5);
        SPARSE_TESTS(cdouble, 1E-5);
    }

    for (size_t testId = 0; testId < tests.size(); ++testId)
        if (tests[testId].joinable()) tests[testId].join();
}

TEST(Threading, DISABLED_MemoryManagerStressTest) {
    vector<std::thread> threads;
    for (int i = 0; i < THREAD_COUNT; i++) {
        threads.emplace_back([] {
            vector<array> arrg;
            int size     = 100;
            int ex_count = 0;

            // Continue until the memory runs out multiple times
            while (true) {
                try {
                    // constantly change size of the array allocated
                    size += 10;
                    arrg.push_back(randu(size));

                    // delete some values intermittently
                    if (!(size % 200)) {
                        arrg.erase(std::begin(arrg), std::begin(arrg) + 5);
                    }
                } catch (const exception& ex) {
                    if (ex_count++ > 3) { break; }
                }
            }
        });
    }
    for (auto& t : threads) { t.join(); }
}

TEST(Threading, DISABLED_Sort) {
    cleanSlate();  // Clean up everything done so far

    vector<std::thread> tests;

    ASSERT_SUCCESS(af_set_device(0));

    for (int i = 0; i < THREAD_COUNT; ++i) {
        tests.emplace_back([] {
            array a = randu(100, 100);
            for (int k = 0; k < 100; ++k) array b = sort(a);
        });
    }

    for (auto& t : tests)
        if (t.joinable()) t.join();
}
