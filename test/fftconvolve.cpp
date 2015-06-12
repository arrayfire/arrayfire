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
#include <testHelpers.hpp>

using std::vector;
using std::string;
using af::cfloat;
using af::cdouble;

template<typename T>
class FFTConvolve : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

template<typename T>
class FFTConvolveLarge : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<cfloat, cdouble, float, double, int, uint, char, uchar> TestTypes;
typedef ::testing::Types<float, double> TestTypesLarge;

// register the type list
TYPED_TEST_CASE(FFTConvolve, TestTypes);
TYPED_TEST_CASE(FFTConvolveLarge, TestTypesLarge);

template<typename T, int baseDim>
void fftconvolveTest(string pTestFile, bool expand)
{
    if (noDoubleTests<T>()) return;

    using af::dim4;

    vector<dim4>      numDims;
    vector<vector<T> >      in;
    vector<vector<T> >   tests;

    readTests<T, T, int>(pTestFile, numDims, in, tests);

    dim4 sDims        = numDims[0];
    dim4 fDims        = numDims[1];
    af_array signal   = 0;
    af_array filter   = 0;
    af_array outArray = 0;
    af_dtype in_type =(af_dtype)af::dtype_traits<T>::af_type;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&signal, &(in[0].front()),
                                          sDims.ndims(), sDims.get(), in_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&filter, &(in[1].front()),
                                          fDims.ndims(), fDims.get(), in_type));

    af_conv_mode mode = expand ? AF_CONV_EXPAND : AF_CONV_DEFAULT;
    switch(baseDim) {
        case 1: ASSERT_EQ(AF_SUCCESS, af_fft_convolve1(&outArray, signal, filter, mode)); break;
        case 2: ASSERT_EQ(AF_SUCCESS, af_fft_convolve2(&outArray, signal, filter, mode)); break;
        case 3: ASSERT_EQ(AF_SUCCESS, af_fft_convolve3(&outArray, signal, filter, mode)); break;
    }

    vector<T> currGoldBar = tests[0];
    size_t nElems         = currGoldBar.size();

    dim_t out_elems = 0;
    ASSERT_EQ(AF_SUCCESS, af_get_elements(&out_elems, outArray));
    ASSERT_EQ(nElems, (size_t)out_elems);

    T *outData            = new T[nElems];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_NEAR(
            real(currGoldBar[elIter]),
            real(outData[elIter])
            , 1e-2)<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(signal));
    ASSERT_EQ(AF_SUCCESS, af_release_array(filter));
}

template<typename T, int baseDim>
void fftconvolveTestLarge(int sDim, int fDim, int sBatch, int fBatch, bool expand)
{
    if (noDoubleTests<T>()) return;

    using af::dim4;
    using af::seq;
    using af::array;

    int outDim = sDim + fDim - 1;
    int fftDim = (int)pow(2, ceil(log2(outDim)));

    int sd[4], fd[4];
    for (int k = 0; k < 4; k++) {
        if (k < baseDim) {
            sd[k] = sDim;
            fd[k] = fDim;
        }
        else if (k == baseDim) {
            sd[k] = sBatch;
            fd[k] = fBatch;
        }
        else {
            sd[k] = 1;
            fd[k] = 1;
        }
    }

    const dim4 signalDims(sd[0], sd[1], sd[2], sd[3]);
    const dim4 filterDims(fd[0], fd[1], fd[2], fd[3]);

    array signal = randu(signalDims, (af_dtype) af::dtype_traits<T>::af_type);
    array filter = randu(filterDims, (af_dtype) af::dtype_traits<T>::af_type);

    array out = fftConvolve(signal, filter, expand ? AF_CONV_EXPAND : AF_CONV_DEFAULT);

    array gold;
    switch(baseDim) {
    case 1:
        gold = real(af::ifft(af::fft(signal, fftDim) * af::fft(filter, fftDim)));
        break;
    case 2:
        gold = real(af::ifft2(af::fft2(signal, fftDim, fftDim) * af::fft2(filter, fftDim, fftDim)));
        break;
    case 3:
        gold = real(af::ifft3(af::fft3(signal, fftDim, fftDim, fftDim) * af::fft3(filter, fftDim, fftDim, fftDim)));
        break;
    default:
        ASSERT_LT(baseDim, 4);
    }

    int cropMin = 0, cropMax = 0;
    if (expand) {
        cropMin = 0;
        cropMax = outDim - 1;
    }
    else {
        cropMin = fDim/2;
        cropMax = outDim - fDim/2 - 1;
    }

    switch(baseDim) {
    case 1:
        gold = gold(seq(cropMin, cropMax));
        break;
    case 2:
        gold = gold(seq(cropMin, cropMax), seq(cropMin, cropMax));
        break;
    case 3:
        gold = gold(seq(cropMin, cropMax), seq(cropMin, cropMax), seq(cropMin, cropMax));
        break;
    }

    size_t outElems  = out.elements();
    size_t goldElems = gold.elements();

    ASSERT_EQ(goldElems, outElems);

    T *goldData = new T[goldElems];
    gold.host(goldData);

    T *outData = new T[outElems];
    out.host(outData);

    for (size_t elIter=0; elIter<outElems; ++elIter) {
        ASSERT_NEAR(goldData[elIter], outData[elIter], 5e-2) << "at: " << elIter << std::endl;
    }

    delete[] goldData;
    delete[] outData;
}

TYPED_TEST(FFTConvolveLarge, VectorLargeSignalSmallFilter)
{
    fftconvolveTestLarge<TypeParam, 1>(32768, 25, 1, 1, true);
}

TYPED_TEST(FFTConvolveLarge, VectorLargeSignalLargeFilter)
{
    fftconvolveTestLarge<TypeParam, 1>(32768, 4095, 1, 1, true);
}

TYPED_TEST(FFTConvolveLarge, SameVectorLargeSignalSmallFilter)
{
    fftconvolveTestLarge<TypeParam, 1>(32768, 25, 1, 1, false);
}

TYPED_TEST(FFTConvolveLarge, SameVectorLargeSignalLargeFilter)
{
    fftconvolveTestLarge<TypeParam, 1>(32768, 4095, 1, 1, false);
}

TYPED_TEST(FFTConvolveLarge, RectangleLargeSignalSmallFilter)
{
    fftconvolveTestLarge<TypeParam, 2>(1024, 5, 1, 1, true);
}

TYPED_TEST(FFTConvolveLarge, RectangleLargeSignalLargeFilter)
{
    fftconvolveTestLarge<TypeParam, 2>(1024, 511, 1, 1, true);
}

TYPED_TEST(FFTConvolveLarge, SameRectangleLargeSignalSmallFilter)
{
    fftconvolveTestLarge<TypeParam, 2>(1024, 5, 1, 1, false);
}

TYPED_TEST(FFTConvolveLarge, SameRectangleLargeSignalLargeFilter)
{
    fftconvolveTestLarge<TypeParam, 2>(1024, 511, 1, 1, false);
}

TYPED_TEST(FFTConvolveLarge, CuboidLargeSignalSmallFilter)
{
    fftconvolveTestLarge<TypeParam, 3>(64, 5, 1, 1, true);
}

TYPED_TEST(FFTConvolveLarge, CuboidLargeSignalLargeFilter)
{
    fftconvolveTestLarge<TypeParam, 3>(64, 31, 1, 1, true);
}

TYPED_TEST(FFTConvolveLarge, SameCuboidLargeSignalSmallFilter)
{
    fftconvolveTestLarge<TypeParam, 3>(64, 5, 1, 1, false);
}

TYPED_TEST(FFTConvolveLarge, SameCuboidLargeSignalLargeFilter)
{
    fftconvolveTestLarge<TypeParam, 2>(64, 31, 1, 1, false);
}

TYPED_TEST(FFTConvolve, Vector)
{
    fftconvolveTest<TypeParam, 1>(string(TEST_DIR"/convolve/vector.test"), true);
}

TYPED_TEST(FFTConvolve, Rectangle)
{
    fftconvolveTest<TypeParam, 2>(string(TEST_DIR"/convolve/rectangle.test"), true);
}

TYPED_TEST(FFTConvolve, Cuboid)
{
    fftconvolveTest<TypeParam, 3>(string(TEST_DIR"/convolve/cuboid.test"), true);
}

TYPED_TEST(FFTConvolve, Vector_Many2One)
{
    fftconvolveTest<TypeParam, 1>(string(TEST_DIR"/convolve/vector_many2one.test"), true);
}

TYPED_TEST(FFTConvolve, Rectangle_Many2One)
{
    fftconvolveTest<TypeParam, 2>(string(TEST_DIR"/convolve/rectangle_many2one.test"), true);
}

TYPED_TEST(FFTConvolve, Cuboid_Many2One)
{
    fftconvolveTest<TypeParam, 3>(string(TEST_DIR"/convolve/cuboid_many2one.test"), true);
}

TYPED_TEST(FFTConvolve, Vector_Many2Many)
{
    fftconvolveTest<TypeParam, 1>(string(TEST_DIR"/convolve/vector_many2many.test"), true);
}

TYPED_TEST(FFTConvolve, Rectangle_Many2Many)
{
    fftconvolveTest<TypeParam, 2>(string(TEST_DIR"/convolve/rectangle_many2many.test"), true);
}

TYPED_TEST(FFTConvolve, Cuboid_Many2Many)
{
    fftconvolveTest<TypeParam, 3>(string(TEST_DIR"/convolve/cuboid_many2many.test"), true);
}

TYPED_TEST(FFTConvolve, Vector_One2Many)
{
    fftconvolveTest<TypeParam, 1>(string(TEST_DIR"/convolve/vector_one2many.test"), true);
}

TYPED_TEST(FFTConvolve, Rectangle_One2Many)
{
    fftconvolveTest<TypeParam, 2>(string(TEST_DIR"/convolve/rectangle_one2many.test"), true);
}

TYPED_TEST(FFTConvolve, Cuboid_One2Many)
{
    fftconvolveTest<TypeParam, 3>(string(TEST_DIR"/convolve/cuboid_one2many.test"), true);
}

TYPED_TEST(FFTConvolve, Same_Vector)
{
    fftconvolveTest<TypeParam, 1>(string(TEST_DIR"/convolve/vector_same.test"), false);
}

TYPED_TEST(FFTConvolve, Same_Rectangle)
{
    fftconvolveTest<TypeParam, 2>(string(TEST_DIR"/convolve/rectangle_same.test"), false);
}

TYPED_TEST(FFTConvolve, Same_Cuboid)
{
    fftconvolveTest<TypeParam, 3>(string(TEST_DIR"/convolve/cuboid_same.test"), false);
}

TYPED_TEST(FFTConvolve, Same_Vector_Many2One)
{
    fftconvolveTest<TypeParam, 1>(string(TEST_DIR"/convolve/vector_same_many2one.test"), false);
}

TYPED_TEST(FFTConvolve, Same_Rectangle_Many2One)
{
    fftconvolveTest<TypeParam, 2>(string(TEST_DIR"/convolve/rectangle_same_many2one.test"), false);
}

TYPED_TEST(FFTConvolve, Same_Cuboid_Many2One)
{
    fftconvolveTest<TypeParam, 3>(string(TEST_DIR"/convolve/cuboid_same_many2one.test"), false);
}

TYPED_TEST(FFTConvolve, Same_Vector_Many2Many)
{
    fftconvolveTest<TypeParam, 1>(string(TEST_DIR"/convolve/vector_same_many2many.test"), false);
}

TYPED_TEST(FFTConvolve, Same_Rectangle_Many2Many)
{
    fftconvolveTest<TypeParam, 2>(string(TEST_DIR"/convolve/rectangle_same_many2many.test"), false);
}

TYPED_TEST(FFTConvolve, Same_Cuboid_Many2Many)
{
    fftconvolveTest<TypeParam, 3>(string(TEST_DIR"/convolve/cuboid_same_many2many.test"), false);
}

TYPED_TEST(FFTConvolve, Same_Vector_One2Many)
{
    fftconvolveTest<TypeParam, 1>(string(TEST_DIR"/convolve/vector_same_one2many.test"), false);
}

TYPED_TEST(FFTConvolve, Same_Rectangle_One2Many)
{
    fftconvolveTest<TypeParam, 2>(string(TEST_DIR"/convolve/rectangle_same_one2many.test"), false);
}
TYPED_TEST(FFTConvolve, Same_Cuboid_One2Many)
{
    fftconvolveTest<TypeParam, 3>(string(TEST_DIR"/convolve/cuboid_same_one2many.test"), false);
}

TEST(FFTConvolve1, CPP)
{
    if (noDoubleTests<float>()) return;

    using af::dim4;

    vector<dim4>      numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTests<float, float, int>(string(TEST_DIR"/convolve/vector.test"), numDims, in, tests);

    //![ex_image_convolve1]
    //vector<dim4> numDims;
    //vector<vector<float> > in;
    af::array signal(numDims[0], &(in[0].front()));
    //signal dims = [32 1 1 1]
    af::array filter(numDims[1], &(in[1].front()));
    //filter dims = [4 1 1 1]

    af::array output = fftConvolve1(signal, filter, AF_CONV_EXPAND);
    //output dims = [32 1 1 1] - same as input since expand(3rd argument is false)
    //None of the dimensions > 1 has lenght > 1, so no batch mode is activated.
    //![ex_image_convolve1]

    vector<float> currGoldBar = tests[0];
    size_t nElems  = output.elements();
    float *outData = new float[nElems];
    output.host(outData);

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_NEAR(currGoldBar[elIter], outData[elIter], 1e-2)<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
}

TEST(FFTConvolve2, CPP)
{
    if (noDoubleTests<float>()) return;

    using af::dim4;

    vector<dim4>      numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTests<float, float, int>(string(TEST_DIR"/convolve/rectangle_one2many.test"), numDims, in, tests);

    //![ex_image_convolve2]
    //vector<dim4> numDims;
    //vector<vector<float> > in;
    af::array signal(numDims[0], &(in[0].front()));
    //signal dims = [15 17 1 1]
    af::array filter(numDims[1], &(in[1].front()));
    //filter dims = [5 5 2 1]

    af::array output = fftConvolve2(signal, filter, AF_CONV_EXPAND);
    //output dims = [15 17 1 1] - same as input since expand(3rd argument is false)
    //however, notice that the 3rd dimension of filter is > 1.
    //So, one to many batch mode will be activated automatically
    //where the 2d input signal is convolved with each 2d filter
    //and the result will written corresponding slice in the output 3d array
    //![ex_image_convolve2]

    vector<float> currGoldBar = tests[0];
    size_t nElems  = output.elements();
    float *outData = new float[nElems];
    output.host(outData);

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_NEAR(currGoldBar[elIter], outData[elIter], 1e-2)<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
}

TEST(FFTConvolve3, CPP)
{
    if (noDoubleTests<float>()) return;

    using af::dim4;

    vector<dim4>      numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTests<float, float, int>(string(TEST_DIR"/convolve/cuboid_many2many.test"), numDims, in, tests);

    //![ex_image_convolve3]
    //vector<dim4> numDims;
    //vector<vector<float> > in;
    af::array signal(numDims[0], &(in[0].front()));
    //signal dims = [10 11 2 2]
    af::array filter(numDims[1], &(in[1].front()));
    //filter dims = [4 2 3 2]

    af::array output = fftConvolve3(signal, filter, AF_CONV_EXPAND);
    //output dims = [10 11 2 2] - same as input since expand(3rd argument is false)
    //however, notice that the 4th dimension is > 1 for both signal
    //and the filter, therefore many to many batch mode will be
    //activated where each 3d signal is convolved with the corresponding 3d filter
    //![ex_image_convolve3]

    vector<float> currGoldBar = tests[0];
    size_t nElems  = output.elements();
    float *outData = new float[nElems];
    output.host(outData);

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_NEAR(currGoldBar[elIter], outData[elIter], 1e-2)<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
}

TEST(FFTConvolve, Docs_Unified_Wrapper)
{
    // This unit test doesn't necessarily need to function
    // accuracy as af::convolve is merely a wrapper to
    // af::convolve[1|2|3]
    using af::array;
    using af::dim4;
    using af::randu;
    using af::constant;
    using af::convolve;

    //![ex_image_convolve_1d]
    array a = randu(10);
    //af_print(a);
    //a [10 1 1 1] = 0.0000 0.1315 0.7556 0.4587 0.5328 0.2190 0.0470 0.6789 0.6793 0.9347
    array b = randu(4);
    //af_print(b);
    //b [4 1 1 1]  = 0.3835 0.5194 0.8310 0.0346
    array c = convolve(a, b);
    //af_print(c);
    //c [10 1 1 1] = 0.3581 0.6777 1.0750 0.7679 0.5903 0.4851 0.6598 1.2770 1.0734 0.8002
    //![ex_image_convolve_1d]

    //![ex_image_convolve_2d]
    array d = constant(0.5, 5, 5);
    //af_print(d);
    //d [5 5 1 1]
    //    0.5000     0.5000     0.5000     0.5000     0.5000
    //    0.5000     0.5000     0.5000     0.5000     0.5000
    //    0.5000     0.5000     0.5000     0.5000     0.5000
    //    0.5000     0.5000     0.5000     0.5000     0.5000
    //    0.5000     0.5000     0.5000     0.5000     0.5000
    array e = constant(1, 2, 2);
    //af_print(e);
    //e [2 2 1 1]
    //     1.0000     1.0000
    //     1.0000     1.0000
    array f = fftConvolve(d, e);
    //af_print(f);
    //f [5 5 1 1]
    //     2.0000     2.0000     2.0000     2.0000     1.0000
    //     2.0000     2.0000     2.0000     2.0000     1.0000
    //     2.0000     2.0000     2.0000     2.0000     1.0000
    //     2.0000     2.0000     2.0000     2.0000     1.0000
    //     1.0000     1.0000     1.0000     1.0000     0.5000
    //![ex_image_convolve_2d]

    //![ex_image_convolve_3d]
    array g = constant(1, 4, 4, 4);
    //af_print(g);
    //g [4 4 4 1]
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000

    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000

    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000

    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    //    1.0000     1.0000     1.0000     1.0000
    array h = constant(0.5, 2, 2, 2);
    //af_print(h);
    //h [2 2 2 1]
    //    0.5000     0.5000
    //    0.5000     0.5000

    //    0.5000     0.5000
    //    0.5000     0.5000

    array i = fftConvolve(g, h);
    //af_print(i);
    //i [4 4 4 1]
    //    4.0000     4.0000     4.0000     2.0000
    //    4.0000     4.0000     4.0000     2.0000
    //    4.0000     4.0000     4.0000     2.0000
    //    2.0000     2.0000     2.0000     1.0000

    //    4.0000     4.0000     4.0000     2.0000
    //    4.0000     4.0000     4.0000     2.0000
    //    4.0000     4.0000     4.0000     2.0000
    //    2.0000     2.0000     2.0000     1.0000

    //    4.0000     4.0000     4.0000     2.0000
    //    4.0000     4.0000     4.0000     2.0000
    //    4.0000     4.0000     4.0000     2.0000
    //    2.0000     2.0000     2.0000     1.0000

    //    2.0000     2.0000     2.0000     1.0000
    //    2.0000     2.0000     2.0000     1.0000
    //    2.0000     2.0000     2.0000     1.0000
    //    1.0000     1.0000     1.0000     0.5000
    //![ex_image_convolve_3d]
}
using namespace af;

TEST(GFOR, fftConvolve2_MO)
{
    array A = randu(5, 5, 3);
    array B = randu(5, 5, 3);
    array K = randu(3, 3);

    gfor(seq ii, 3) {
        B(span, span, ii) = fftConvolve2(A(span, span, ii), K);
    }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = fftConvolve2(A(span, span, ii), K);
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}

TEST(GFOR, fftConvolve2_OM)
{
    array A = randu(5, 5);
    array B = randu(5, 5, 3);
    array K = randu(3, 3, 3);

    gfor(seq ii, 3) {
        B(span, span, ii) = fftConvolve2(A, K(span, span, ii));
    }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = fftConvolve2(A, K(span, span, ii));
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}

TEST(GFOR, fftConvolve2_MM)
{
    array A = randu(5, 5, 3);
    array B = randu(5, 5, 3);
    array K = randu(3, 3, 3);

    gfor(seq ii, 3) {
        B(span, span, ii) = fftConvolve2(A(span, span, ii), K(span, span, ii));
    }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = fftConvolve2(A(span, span, ii), K(span, span, ii));
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}

TEST(Padding, fftConvolve2)
{
    for (int n = 5; n < 32; n++) {
        array a = randu(n, n);
        array b = randu(5, 5);
        array c = fftConvolve2(a, b);
        array d = convolve2(a, b, AF_CONV_DEFAULT, AF_CONV_SPATIAL);
        ASSERT_EQ(max<double>(abs(c - d)) < 1E-5, true);
    }
}
