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
#include <string>
#include <vector>
#include <stdexcept>
#include <testHelpers.hpp>

using std::string;
using std::vector;
using af::af_cfloat;
using af::af_cdouble;

TEST(fft, Invalid_Array)
{
    vector<float>   in(10000,1);

    af_array inArray   = 0;
    af_array outArray  = 0;

    af::dim4 dims(10,10,10,1);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in.front()),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_ERR_SIZE, af_fft(&outArray, inArray, 1.0, 0));
}

TEST(fft2, Invalid_Array)
{
    vector<float>   in(100,1);

    af_array inArray   = 0;
    af_array outArray  = 0;

    af::dim4 dims(5,5,2,2);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in.front()),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_ERR_SIZE, af_fft2(&outArray, inArray, 1.0, 0, 0));
}

TEST(fft3, Invalid_Array)
{
    vector<float>   in(100,1);

    af_array inArray   = 0;
    af_array outArray  = 0;

    af::dim4 dims(10,10,1,1);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in.front()),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_ERR_SIZE, af_fft3(&outArray, inArray, 1.0, 0, 0, 0));
}

TEST(ifft1, Invalid_Array)
{
    vector<float>   in(100,1);

    af_array inArray   = 0;
    af_array outArray  = 0;

    af::dim4 dims(5,5,4,1);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in.front()),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<af_cfloat>::af_type));

    ASSERT_EQ(AF_ERR_SIZE, af_ifft(&outArray, inArray, 0.01, 0));
}

TEST(ifft2, Invalid_Array)
{
    vector<float>   in(100,1);

    af_array inArray   = 0;
    af_array outArray  = 0;

    af::dim4 dims(100,1,1,1);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in.front()),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<af_cfloat>::af_type));

    ASSERT_EQ(AF_ERR_SIZE, af_ifft2(&outArray, inArray, 0.01, 0, 0));
}

TEST(ifft3, Invalid_Array)
{
    vector<float>   in(100,1);

    af_array inArray   = 0;
    af_array outArray  = 0;

    af::dim4 dims(10,10,1,1);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in.front()),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<af_cfloat>::af_type));

    ASSERT_EQ(AF_ERR_SIZE, af_ifft3(&outArray, inArray, 0.01, 0, 0, 0));
}

template<typename inType, typename outType, bool isInverse>
void fftTest(string pTestFile, dim_type pad0=0, dim_type pad1=0, dim_type pad2=0)
{
    vector<af::dim4>        numDims;
    vector<vector<inType>>       in;
    vector<vector<outType>>   tests;

    readTestsFromFile<inType, outType>(pTestFile, numDims, in, tests);

    af::dim4 dims       = numDims[0];
    af_array outArray   = 0;
    af_array inArray    = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()),
                dims.ndims(), dims.get(), (af_dtype)af::dtype_traits<inType>::af_type));

    if (isInverse){
        double norm_factor = 1.0;
#if defined(AF_OPENCL)
        norm_factor *= (dims.elements());
#endif
        switch(dims.ndims()) {
            case 1 : ASSERT_EQ(AF_SUCCESS, af_ifft (&outArray, inArray, norm_factor, pad0));              break;
            case 2 : ASSERT_EQ(AF_SUCCESS, af_ifft2(&outArray, inArray, norm_factor, pad0, pad1));        break;
            case 3 : ASSERT_EQ(AF_SUCCESS, af_ifft3(&outArray, inArray, norm_factor, pad0, pad1, pad2));  break;
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

    vector<outType> goldBar(begin(tests[0]), end(tests[0]));

    size_t test_size = 0;
    switch(dims.ndims()) {
        case 1  : test_size = dims[0]/2+1;                       break;
        case 2  : test_size = dims[1] * (dims[0]/2+1);           break;
        case 3  : test_size = dims[2] * dims[1] * (dims[0]/2+1); break;
        default : test_size = dims[0]/2+1;                       break;
    }
    for (size_t elIter=0; elIter<test_size; ++elIter) {
        bool isUnderTolerance = std::abs(goldBar[elIter]-outData[elIter])<0.001;
        ASSERT_EQ(true, isUnderTolerance)<<
            "Expected value="<<goldBar[elIter] <<"\t Actual Value="<<
            outData[elIter] << " at: " << elIter<< std::endl;
    }

    // cleanup
    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(outArray));
}

#define INSTANTIATE_TEST(func, name, is_inverse, in_t, out_t, ...)  \
    TEST(func, name)                                                \
    {                                                               \
        fftTest<in_t, out_t, is_inverse>(__VA_ARGS__);              \
    }

// Real to complex transforms
INSTANTIATE_TEST(fft ,  R2C_Float, false,  float,  af_cfloat, string(TEST_DIR"/signal/fft_r2c.test") );
INSTANTIATE_TEST(fft , R2C_Double, false, double, af_cdouble, string(TEST_DIR"/signal/fft_r2c.test") );
INSTANTIATE_TEST(fft2,  R2C_Float, false,  float,  af_cfloat, string(TEST_DIR"/signal/fft2_r2c.test"));
INSTANTIATE_TEST(fft2, R2C_Double, false, double, af_cdouble, string(TEST_DIR"/signal/fft2_r2c.test"));
INSTANTIATE_TEST(fft3,  R2C_Float, false,  float,  af_cfloat, string(TEST_DIR"/signal/fft3_r2c.test"));
INSTANTIATE_TEST(fft3, R2C_Double, false, double, af_cdouble, string(TEST_DIR"/signal/fft3_r2c.test"));

// complex to complex transforms
INSTANTIATE_TEST(fft ,  C2C_Float, false,  af_cfloat,  af_cfloat, string(TEST_DIR"/signal/fft_c2c.test") );
INSTANTIATE_TEST(fft , C2C_Double, false, af_cdouble, af_cdouble, string(TEST_DIR"/signal/fft_c2c.test") );
INSTANTIATE_TEST(fft2,  C2C_Float, false,  af_cfloat,  af_cfloat, string(TEST_DIR"/signal/fft2_c2c.test"));
INSTANTIATE_TEST(fft2, C2C_Double, false, af_cdouble, af_cdouble, string(TEST_DIR"/signal/fft2_c2c.test"));
INSTANTIATE_TEST(fft3,  C2C_Float, false,  af_cfloat,  af_cfloat, string(TEST_DIR"/signal/fft3_c2c.test"));
INSTANTIATE_TEST(fft3, C2C_Double, false, af_cdouble, af_cdouble, string(TEST_DIR"/signal/fft3_c2c.test"));

// transforms on padded and truncated arrays
INSTANTIATE_TEST(fft2,  R2C_Float_Trunc, false,  float,  af_cfloat, string(TEST_DIR"/signal/fft2_r2c_trunc.test"), 16, 16);
INSTANTIATE_TEST(fft2, R2C_Double_Trunc, false, double, af_cdouble, string(TEST_DIR"/signal/fft2_r2c_trunc.test"), 16, 16);

INSTANTIATE_TEST(fft2,  C2C_Float_Pad, false,  af_cfloat,  af_cfloat, string(TEST_DIR"/signal/fft2_c2c_pad.test"), 16, 16);
INSTANTIATE_TEST(fft2, C2C_Double_Pad, false, af_cdouble, af_cdouble, string(TEST_DIR"/signal/fft2_c2c_pad.test"), 16, 16);

// inverse transforms
// complex to complex transforms
INSTANTIATE_TEST(ifft ,  C2C_Float, true,  af_cfloat,  af_cfloat, string(TEST_DIR"/signal/ifft_c2c.test") );
INSTANTIATE_TEST(ifft , C2C_Double, true, af_cdouble, af_cdouble, string(TEST_DIR"/signal/ifft_c2c.test") );
INSTANTIATE_TEST(ifft2,  C2C_Float, true,  af_cfloat,  af_cfloat, string(TEST_DIR"/signal/ifft2_c2c.test"));
INSTANTIATE_TEST(ifft2, C2C_Double, true, af_cdouble, af_cdouble, string(TEST_DIR"/signal/ifft2_c2c.test"));
INSTANTIATE_TEST(ifft3,  C2C_Float, true,  af_cfloat,  af_cfloat, string(TEST_DIR"/signal/ifft3_c2c.test"));
INSTANTIATE_TEST(ifft3, C2C_Double, true, af_cdouble, af_cdouble, string(TEST_DIR"/signal/ifft3_c2c.test"));


template<typename inType, typename outType, int rank, bool isInverse>
void fftBatchTest(string pTestFile, dim_type pad0=0, dim_type pad1=0, dim_type pad2=0)
{
    vector<af::dim4>        numDims;
    vector<vector<inType>>       in;
    vector<vector<outType>>   tests;

    readTestsFromFile<inType, outType>(pTestFile, numDims, in, tests);

    af::dim4 dims       = numDims[0];
    af_array outArray   = 0;
    af_array inArray    = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()),
                dims.ndims(), dims.get(), (af_dtype)af::dtype_traits<inType>::af_type));

    if(isInverse) {
        double norm_factor = 1.0;
#if defined(AF_OPENCL)
        dim_type cnt = dims.ndims();
        for(dim_type r=0; r<cnt-1; r++)
            norm_factor *= dims[r];
#endif
        switch(rank) {
            case 1 : ASSERT_EQ(AF_SUCCESS, af_ifft (&outArray, inArray, norm_factor, pad0));              break;
            case 2 : ASSERT_EQ(AF_SUCCESS, af_ifft2(&outArray, inArray, norm_factor, pad0, pad1));        break;
            case 3 : ASSERT_EQ(AF_SUCCESS, af_ifft3(&outArray, inArray, norm_factor, pad0, pad1, pad2));  break;
            default: throw std::runtime_error("This error shouldn't happen, pls check");
        }
    } else {
        switch(rank) {
            case 1 : ASSERT_EQ(AF_SUCCESS, af_fft (&outArray, inArray, 1.0, pad0));               break;
            case 2 : ASSERT_EQ(AF_SUCCESS, af_fft2(&outArray, inArray, 1.0, pad0, pad1));         break;
            case 3 : ASSERT_EQ(AF_SUCCESS, af_fft3(&outArray, inArray, 1.0, pad0, pad1, pad2));   break;
            default: throw std::runtime_error("This error shouldn't happen, pls check");
        }
    }

    size_t out_size = tests[0].size();
    outType *outData= new outType[out_size];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    vector<outType> goldBar(begin(tests[0]), end(tests[0]));

    size_t test_size = 0;
    size_t batch_count = dims[rank];
    switch(rank) {
        case 1  : test_size = dims[0]/2+1;                       break;
        case 2  : test_size = dims[1] * (dims[0]/2+1);           break;
        case 3  : test_size = dims[2] * dims[1] * (dims[0]/2+1); break;
        default : test_size = dims[0]/2+1;                       break;
    }

    size_t batch_stride = 1;
    for(int i=0; i<rank; ++i) batch_stride *= dims[i];

    for(size_t batchId=0; batchId<batch_count; ++batchId) {
        size_t off = batchId*batch_stride;
        for (size_t elIter=0; elIter<test_size; ++elIter) {
            bool isUnderTolerance = std::abs(goldBar[elIter+off]-outData[elIter+off])<0.001;
            ASSERT_EQ(true, isUnderTolerance)<<"Batch id = "<<batchId<<
                "; Expected value="<<goldBar[elIter+off] <<"\t Actual Value="<<
                outData[elIter+off] << " at: " << elIter<< std::endl;
        }
    }

    // cleanup
    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(outArray));
}

#define INSTANTIATE_BATCH_TEST(func, name, rank, is_inverse, in_t, out_t, ...) \
    TEST(func, name##_Batch)                                                   \
    {                                                                          \
        fftBatchTest<in_t, out_t, rank, is_inverse>(__VA_ARGS__);              \
    }

// real to complex transforms
INSTANTIATE_BATCH_TEST(fft , R2C_Float, 1, false, float, af_cfloat, string(TEST_DIR"/signal/fft_r2c_batch.test") );
INSTANTIATE_BATCH_TEST(fft2, R2C_Float, 2, false, float, af_cfloat, string(TEST_DIR"/signal/fft2_r2c_batch.test"));
INSTANTIATE_BATCH_TEST(fft3, R2C_Float, 3, false, float, af_cfloat, string(TEST_DIR"/signal/fft3_r2c_batch.test"));

// complex to complex transforms
INSTANTIATE_BATCH_TEST(fft , C2C_Float, 1, false, af_cfloat, af_cfloat, string(TEST_DIR"/signal/fft_c2c_batch.test") );
INSTANTIATE_BATCH_TEST(fft2, C2C_Float, 2, false, af_cfloat, af_cfloat, string(TEST_DIR"/signal/fft2_c2c_batch.test"));
INSTANTIATE_BATCH_TEST(fft3, C2C_Float, 3, false, af_cfloat, af_cfloat, string(TEST_DIR"/signal/fft3_c2c_batch.test"));

// inverse transforms
// complex to complex transforms
INSTANTIATE_BATCH_TEST(ifft , C2C_Float, 1, true, af_cfloat, af_cfloat, string(TEST_DIR"/signal/ifft_c2c_batch.test") );
INSTANTIATE_BATCH_TEST(ifft2, C2C_Float, 2, true, af_cfloat, af_cfloat, string(TEST_DIR"/signal/ifft2_c2c_batch.test"));
INSTANTIATE_BATCH_TEST(ifft3, C2C_Float, 3, true, af_cfloat, af_cfloat, string(TEST_DIR"/signal/ifft3_c2c_batch.test"));

// transforms on padded and truncated arrays
INSTANTIATE_BATCH_TEST(fft2,  R2C_Float_Trunc, 2, false,  float,  af_cfloat, string(TEST_DIR"/signal/fft2_r2c_trunc_batch.test"), 16, 16);
INSTANTIATE_BATCH_TEST(fft2, R2C_Double_Trunc, 2, false, double, af_cdouble, string(TEST_DIR"/signal/fft2_r2c_trunc_batch.test"), 16, 16);
INSTANTIATE_BATCH_TEST(fft2,  C2C_Float_Pad, 2, false,  af_cfloat,  af_cfloat, string(TEST_DIR"/signal/fft2_c2c_pad_batch.test"), 16, 16);
INSTANTIATE_BATCH_TEST(fft2, C2C_Double_Pad, 2, false, af_cdouble, af_cdouble, string(TEST_DIR"/signal/fft2_c2c_pad_batch.test"), 16, 16);


/////////////////////////////////////// CPP ////////////////////////////////////
//
template<typename inType, typename outType, bool isInverse>
void cppFFTTest(string pTestFile, dim_type pad0=0, dim_type pad1=0, dim_type pad2=0)
{
    vector<af::dim4>        numDims;
    vector<vector<inType>>       in;
    vector<vector<outType>>   tests;

    readTestsFromFile<inType, outType>(pTestFile, numDims, in, tests);

    af::dim4 dims = numDims[0];
    af::array signal(dims, &(in[0].front()));
    af::array output;

    if (isInverse){
        double norm_factor = 1.0;
#if defined(AF_OPENCL)
        norm_factor *= (dims.elements());
#endif
        output = ifft3(signal, norm_factor);
    } else {
        output = fft3(signal, 1.0);
    }

    size_t out_size = tests[0].size();
    af_cfloat *outData= new af_cfloat[out_size];
    output.host((void*)outData);

    vector<af_cfloat> goldBar(begin(tests[0]), end(tests[0]));

    size_t test_size = 0;
    switch(dims.ndims()) {
        case 1  : test_size = dims[0]/2+1;                       break;
        case 2  : test_size = dims[1] * (dims[0]/2+1);           break;
        case 3  : test_size = dims[2] * dims[1] * (dims[0]/2+1); break;
        default : test_size = dims[0]/2+1;                       break;
    }
    for (size_t elIter=0; elIter<test_size; ++elIter) {
        bool isUnderTolerance = std::abs(goldBar[elIter]-outData[elIter])<0.001;
        ASSERT_EQ(true, isUnderTolerance)<<
            "Expected value="<<goldBar[elIter] <<"\t Actual Value="<<
            outData[elIter] << " at: " << elIter<< std::endl;
    }
    // cleanup
    delete[] outData;
}


TEST(fft3, CPP)
{
    cppFFTTest<af_cfloat, af_cfloat, false>(string(TEST_DIR"/signal/fft3_c2c.test"));
}

TEST(ifft3, CPP)
{
    cppFFTTest<af_cfloat, af_cfloat, true>(string(TEST_DIR"/signal/ifft3_c2c.test"));
}
