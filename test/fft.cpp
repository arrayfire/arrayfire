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
using af::cfloat;
using af::cdouble;

TEST(fft, Invalid_Type)
{
    vector<char>   in(100,1);

    af_array inArray   = 0;
    af_array outArray  = 0;

    af::dim4 dims(5 * 5 * 2 * 2);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in.front()),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<char>::af_type));

    ASSERT_EQ(AF_ERR_TYPE, af_fft(&outArray, inArray, 1.0, 0));
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}

TEST(fft2, Invalid_Array)
{
    if (noDoubleTests<float>()) return;

    vector<float>   in(100,1);

    af_array inArray   = 0;
    af_array outArray  = 0;

    af::dim4 dims(5 * 5 * 2 * 2);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in.front()),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_ERR_SIZE, af_fft2(&outArray, inArray, 1.0, 0, 0));
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}

TEST(fft3, Invalid_Array)
{
    if (noDoubleTests<float>()) return;

    vector<float>   in(100,1);

    af_array inArray   = 0;
    af_array outArray  = 0;

    af::dim4 dims(10,10,1,1);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in.front()),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_ERR_SIZE, af_fft3(&outArray, inArray, 1.0, 0, 0, 0));
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}

TEST(ifft2, Invalid_Array)
{
    if (noDoubleTests<float>()) return;

    vector<float>   in(100,1);

    af_array inArray   = 0;
    af_array outArray  = 0;

    af::dim4 dims(100,1,1,1);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in.front()),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<cfloat>::af_type));

    ASSERT_EQ(AF_ERR_SIZE, af_ifft2(&outArray, inArray, 0.01, 0, 0));
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}

TEST(ifft3, Invalid_Array)
{
    if (noDoubleTests<float>()) return;

    vector<float>   in(100,1);

    af_array inArray   = 0;
    af_array outArray  = 0;

    af::dim4 dims(10,10,1,1);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in.front()),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<cfloat>::af_type));

    ASSERT_EQ(AF_ERR_SIZE, af_ifft3(&outArray, inArray, 0.01, 0, 0, 0));
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}

template<typename inType, typename outType, bool isInverse>
void fftTest(string pTestFile, dim_t pad0=0, dim_t pad1=0, dim_t pad2=0)
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
            (output_scale*outData[elIter]) << " at: " << elIter<< std::endl;
    }

    // cleanup
    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

#define INSTANTIATE_TEST(func, name, is_inverse, in_t, out_t, ...)  \
    TEST(func, name)                                                \
    {                                                               \
        fftTest<in_t, out_t, is_inverse>(__VA_ARGS__);              \
    }

// Real to complex transforms
INSTANTIATE_TEST(fft ,  R2C_Float, false,  float,  cfloat, string(TEST_DIR"/signal/fft_r2c.test") );
INSTANTIATE_TEST(fft , R2C_Double, false, double, cdouble, string(TEST_DIR"/signal/fft_r2c.test") );
INSTANTIATE_TEST(fft2,  R2C_Float, false,  float,  cfloat, string(TEST_DIR"/signal/fft2_r2c.test"));
INSTANTIATE_TEST(fft2, R2C_Double, false, double, cdouble, string(TEST_DIR"/signal/fft2_r2c.test"));
INSTANTIATE_TEST(fft3,  R2C_Float, false,  float,  cfloat, string(TEST_DIR"/signal/fft3_r2c.test"));
INSTANTIATE_TEST(fft3, R2C_Double, false, double, cdouble, string(TEST_DIR"/signal/fft3_r2c.test"));

// complex to complex transforms
INSTANTIATE_TEST(fft ,  C2C_Float, false,  cfloat,  cfloat, string(TEST_DIR"/signal/fft_c2c.test") );
INSTANTIATE_TEST(fft , C2C_Double, false, cdouble, cdouble, string(TEST_DIR"/signal/fft_c2c.test") );
INSTANTIATE_TEST(fft2,  C2C_Float, false,  cfloat,  cfloat, string(TEST_DIR"/signal/fft2_c2c.test"));
INSTANTIATE_TEST(fft2, C2C_Double, false, cdouble, cdouble, string(TEST_DIR"/signal/fft2_c2c.test"));
INSTANTIATE_TEST(fft3,  C2C_Float, false,  cfloat,  cfloat, string(TEST_DIR"/signal/fft3_c2c.test"));
INSTANTIATE_TEST(fft3, C2C_Double, false, cdouble, cdouble, string(TEST_DIR"/signal/fft3_c2c.test"));

// transforms on padded and truncated arrays
INSTANTIATE_TEST(fft2,  R2C_Float_Trunc, false,  float,  cfloat, string(TEST_DIR"/signal/fft2_r2c_trunc.test"), 16, 16);
INSTANTIATE_TEST(fft2, R2C_Double_Trunc, false, double, cdouble, string(TEST_DIR"/signal/fft2_r2c_trunc.test"), 16, 16);

INSTANTIATE_TEST(fft2,  C2C_Float_Pad, false,  cfloat,  cfloat, string(TEST_DIR"/signal/fft2_c2c_pad.test"), 16, 16);
INSTANTIATE_TEST(fft2, C2C_Double_Pad, false, cdouble, cdouble, string(TEST_DIR"/signal/fft2_c2c_pad.test"), 16, 16);

// inverse transforms
// complex to complex transforms
INSTANTIATE_TEST(ifft ,  C2C_Float, true,  cfloat,  cfloat, string(TEST_DIR"/signal/ifft_c2c.test") );
INSTANTIATE_TEST(ifft , C2C_Double, true, cdouble, cdouble, string(TEST_DIR"/signal/ifft_c2c.test") );
INSTANTIATE_TEST(ifft2,  C2C_Float, true,  cfloat,  cfloat, string(TEST_DIR"/signal/ifft2_c2c.test"));
INSTANTIATE_TEST(ifft2, C2C_Double, true, cdouble, cdouble, string(TEST_DIR"/signal/ifft2_c2c.test"));
INSTANTIATE_TEST(ifft3,  C2C_Float, true,  cfloat,  cfloat, string(TEST_DIR"/signal/ifft3_c2c.test"));
INSTANTIATE_TEST(ifft3, C2C_Double, true, cdouble, cdouble, string(TEST_DIR"/signal/ifft3_c2c.test"));


template<typename inType, typename outType, int rank, bool isInverse>
void fftBatchTest(string pTestFile, dim_t pad0=0, dim_t pad1=0, dim_t pad2=0)
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

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()),
                dims.ndims(), dims.get(), (af_dtype)af::dtype_traits<inType>::af_type));

    if(isInverse) {
        switch(rank) {
            case 1 : ASSERT_EQ(AF_SUCCESS, af_ifft (&outArray, inArray, 1.0, pad0));              break;
            case 2 : ASSERT_EQ(AF_SUCCESS, af_ifft2(&outArray, inArray, 1.0, pad0, pad1));        break;
            case 3 : ASSERT_EQ(AF_SUCCESS, af_ifft3(&outArray, inArray, 1.0, pad0, pad1, pad2));  break;
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

    vector<outType> goldBar(tests[0].begin(), tests[0].end());

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

    outType output_scale = (outType)(isInverse ? test_size : 1);
    for(size_t batchId=0; batchId<batch_count; ++batchId) {
        size_t off = batchId*batch_stride;
        for (size_t elIter=0; elIter<test_size; ++elIter) {
            bool isUnderTolerance = abs(goldBar[elIter+off]-outData[elIter+off])<0.001;
            ASSERT_EQ(true, isUnderTolerance)<<"Batch id = "<<batchId<<
                "; Expected value="<<goldBar[elIter+off] <<"\t Actual Value="<<
                (output_scale*outData[elIter+off]) << " at: " << elIter<< std::endl;
        }
    }

    // cleanup
    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

#define INSTANTIATE_BATCH_TEST(func, name, rank, is_inverse, in_t, out_t, ...) \
    TEST(func, name##_Batch)                                                   \
    {                                                                          \
        fftBatchTest<in_t, out_t, rank, is_inverse>(__VA_ARGS__);              \
    }

// real to complex transforms
INSTANTIATE_BATCH_TEST(fft , R2C_Float, 1, false, float, cfloat, string(TEST_DIR"/signal/fft_r2c_batch.test") );
INSTANTIATE_BATCH_TEST(fft2, R2C_Float, 2, false, float, cfloat, string(TEST_DIR"/signal/fft2_r2c_batch.test"));
INSTANTIATE_BATCH_TEST(fft3, R2C_Float, 3, false, float, cfloat, string(TEST_DIR"/signal/fft3_r2c_batch.test"));

// complex to complex transforms
INSTANTIATE_BATCH_TEST(fft , C2C_Float, 1, false, cfloat, cfloat, string(TEST_DIR"/signal/fft_c2c_batch.test") );
INSTANTIATE_BATCH_TEST(fft2, C2C_Float, 2, false, cfloat, cfloat, string(TEST_DIR"/signal/fft2_c2c_batch.test"));
INSTANTIATE_BATCH_TEST(fft3, C2C_Float, 3, false, cfloat, cfloat, string(TEST_DIR"/signal/fft3_c2c_batch.test"));

// inverse transforms
// complex to complex transforms
INSTANTIATE_BATCH_TEST(ifft , C2C_Float, 1, true, cfloat, cfloat, string(TEST_DIR"/signal/ifft_c2c_batch.test") );
INSTANTIATE_BATCH_TEST(ifft2, C2C_Float, 2, true, cfloat, cfloat, string(TEST_DIR"/signal/ifft2_c2c_batch.test"));
INSTANTIATE_BATCH_TEST(ifft3, C2C_Float, 3, true, cfloat, cfloat, string(TEST_DIR"/signal/ifft3_c2c_batch.test"));

// transforms on padded and truncated arrays
INSTANTIATE_BATCH_TEST(fft2,  R2C_Float_Trunc, 2, false,  float,  cfloat, string(TEST_DIR"/signal/fft2_r2c_trunc_batch.test"), 16, 16);
INSTANTIATE_BATCH_TEST(fft2, R2C_Double_Trunc, 2, false, double, cdouble, string(TEST_DIR"/signal/fft2_r2c_trunc_batch.test"), 16, 16);
INSTANTIATE_BATCH_TEST(fft2,  C2C_Float_Pad, 2, false,  cfloat,  cfloat, string(TEST_DIR"/signal/fft2_c2c_pad_batch.test"), 16, 16);
INSTANTIATE_BATCH_TEST(fft2, C2C_Double_Pad, 2, false, cdouble, cdouble, string(TEST_DIR"/signal/fft2_c2c_pad_batch.test"), 16, 16);


/////////////////////////////////////// CPP ////////////////////////////////////
//
template<typename inType, typename outType, bool isInverse>
void cppFFTTest(string pTestFile, dim_t pad0=0, dim_t pad1=0, dim_t pad2=0)
{
    if (noDoubleTests<inType>()) return;
    if (noDoubleTests<outType>()) return;

    vector<af::dim4>        numDims;
    vector<vector<inType> >       in;
    vector<vector<outType> >   tests;

    readTestsFromFile<inType, outType>(pTestFile, numDims, in, tests);

    af::dim4 dims = numDims[0];
    af::array signal(dims, &(in[0].front()));
    af::array output;

    if (isInverse){
        output = ifft3Norm(signal, 1.0);
    } else {
        output = fft3Norm(signal, 1.0);
    }

    size_t out_size = tests[0].size();
    cfloat *outData= new cfloat[out_size];
    output.host((void*)outData);

    vector<cfloat> goldBar(tests[0].begin(), tests[0].end());

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
            (output_scale*outData[elIter]) << " at: " << elIter<< std::endl;
    }
    // cleanup
    delete[] outData;
}

template<typename inType, typename outType, bool isInverse>
void cppDFTTest(string pTestFile, dim_t pad0=0, dim_t pad1=0, dim_t pad2=0)
{
    if (noDoubleTests<inType>()) return;
    if (noDoubleTests<outType>()) return;

    vector<af::dim4>        numDims;
    vector<vector<inType> >       in;
    vector<vector<outType> >   tests;

    readTestsFromFile<inType, outType>(pTestFile, numDims, in, tests);

    af::dim4 dims = numDims[0];
    af::array signal(dims, &(in[0].front()));
    af::array output;

    if (isInverse){
        output = idft(signal);
    } else {
        output = dft(signal);
    }

    size_t out_size = tests[0].size();
    cfloat *outData= new cfloat[out_size];
    output.host((void*)outData);

    vector<cfloat> goldBar(tests[0].begin(), tests[0].end());

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
            (output_scale*outData[elIter]) << " at: " << elIter<< std::endl;
    }
    // cleanup
    delete[] outData;
}


TEST(fft3, CPP)
{
    cppFFTTest<cfloat, cfloat, false>(string(TEST_DIR"/signal/fft3_c2c.test"));
}

TEST(ifft3, CPP)
{
    cppFFTTest<cfloat, cfloat, true>(string(TEST_DIR"/signal/ifft3_c2c.test"));
}

TEST(fft3, RandomData)
{
    af::array a = af::randu(31, 31, 31);
    af::array b = af::fft3(a, 64, 64, 64);
    af::array c = af::ifft3(b);

    af::dim4 aDims = a.dims();
    af::dim4 cDims = c.dims();
    af::dim4 aStrides(1, aDims[0], aDims[0]*aDims[1], aDims[0]*aDims[1]*aDims[2]);
    af::dim4 cStrides(1, cDims[0], cDims[0]*cDims[1], cDims[0]*cDims[1]*cDims[2]);

    float* gold = new float[a.elements()];
    float* out  = new float[2*c.elements()];

    a.host((void*)gold);
    c.host((void*)out);

    for (int k=0; k<(int)aDims[2]; ++k) {
        int gkOff = k*aStrides[2];
        int okOff = k*cStrides[2];
        for (int j=0; j<(int)aDims[1]; ++j) {
            int gjOff = j*aStrides[1];
            int ojOff = j*cStrides[1];
            for (int i=0; i<(int)aDims[0]; ++i) {
                int giOff = i*aStrides[0];
                int oiOff = i*cStrides[0];

                int gi = gkOff + gjOff + giOff;
                int oi = okOff + ojOff + oiOff;

                bool isUnderTolerance = std::abs(gold[gi]-out[2*oi])<0.001;
                ASSERT_EQ(true, isUnderTolerance)<< "Expected value="<<
                    gold[gi] <<"\t Actual Value="<< out[2*oi] << " at: " <<gi<< std::endl;
            }
        }
    }

    delete[] gold;
    delete[] out;
}

TEST(dft, CPP)
{
    cppDFTTest<cfloat, cfloat, false>(string(TEST_DIR"/signal/fft_c2c.test"));
}

TEST(idft, CPP)
{
    cppDFTTest<cfloat, cfloat, true>(string(TEST_DIR"/signal/ifft_c2c.test"));
}

TEST(dft2, CPP)
{
    cppDFTTest<cfloat, cfloat, false>(string(TEST_DIR"/signal/fft2_c2c.test"));
}

TEST(idft2, CPP)
{
    cppDFTTest<cfloat, cfloat, true>(string(TEST_DIR"/signal/ifft2_c2c.test"));
}

TEST(dft3, CPP)
{
    cppDFTTest<cfloat, cfloat, false>(string(TEST_DIR"/signal/fft3_c2c.test"));
}

TEST(idft3, CPP)
{
    cppDFTTest<cfloat, cfloat, true>(string(TEST_DIR"/signal/ifft3_c2c.test"));
}

TEST(fft, CPP_4D)
{
    af::array a = af::randu(1024, 1024);
    af::array b = af::fft(a);

    af::array A = af::moddims(a, 1024, 32, 16, 2);
    af::array B = af::fft(A);

    af::cfloat *h_b = b.host<af::cfloat>();
    af::cfloat *h_B = B.host<af::cfloat>();

    for (int i = 0; i < (int)a.elements(); i++) {
        ASSERT_EQ(h_b[i], h_B[i]) << "at: " << i << std::endl;
    }

    delete[] h_b;
    delete[] h_B;
}

TEST(ifft, CPP_4D)
{
    af::array a = af::randu(1024, 1024, c32);
    af::array b = af::ifft(a);

    af::array A = af::moddims(a, 1024, 32, 16, 2);
    af::array B = af::ifft(A);

    af::cfloat *h_b = b.host<af::cfloat>();
    af::cfloat *h_B = B.host<af::cfloat>();

    for (int i = 0; i < (int)a.elements(); i++) {
        ASSERT_EQ(h_b[i], h_B[i]) << "at: " << i << std::endl;
    }

    delete[] h_b;
    delete[] h_B;
}

TEST(fft, GFOR)
{
    af::array a = af::randu(1024, 1024);
    af::array b = af::constant(0, 1024, 1024, c32);
    af::array c = af::fft(a);

    gfor(af::seq ii, a.dims(1)) {
        b(af::span, ii) = af::fft(a(af::span, ii));
    }

    af::cfloat *h_b = b.host<af::cfloat>();
    af::cfloat *h_c = c.host<af::cfloat>();

    for (int i = 0; i < (int)a.elements(); i++) {
        ASSERT_EQ(h_b[i], h_c[i]) << "at: " << i << std::endl;
    }

    delete[] h_b;
    delete[] h_c;
}

TEST(fft2, GFOR)
{
    af::array a = af::randu(1024, 1024, 4);
    af::array b = af::constant(0, 1024, 1024, 4, c32);
    af::array c = af::fft2(a);

    gfor(af::seq ii, a.dims(2)) {
        b(af::span, af::span, ii) = af::fft2(a(af::span, af::span, ii));
    }

    af::cfloat *h_b = b.host<af::cfloat>();
    af::cfloat *h_c = c.host<af::cfloat>();

    for (int i = 0; i < (int)a.elements(); i++) {
        ASSERT_EQ(h_b[i], h_c[i]) << "at: " << i << std::endl;
    }

    delete[] h_b;
    delete[] h_c;
}

TEST(fft3, GFOR)
{
    af::array a = af::randu(32, 32, 32, 4);
    af::array b = af::constant(0, 32, 32, 32, 4, c32);
    af::array c = af::fft3(a);

    gfor(af::seq ii, a.dims(3)) {
        b(af::span, af::span, af::span, ii) = af::fft3(a(af::span, af::span, af::span, ii));
    }

    af::cfloat *h_b = b.host<af::cfloat>();
    af::cfloat *h_c = c.host<af::cfloat>();

    for (int i = 0; i < (int)a.elements(); i++) {
        ASSERT_EQ(h_b[i], h_c[i]) << "at: " << i << std::endl;
    }

    delete[] h_b;
    delete[] h_c;
}
