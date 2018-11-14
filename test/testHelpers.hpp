/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#include <af/array.h>
#include <af/dim4.hpp>
#include <af/internal.h>
#include <af/traits.hpp>
#include <arrayfire.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#define UNUSED(expr) do { (void)(expr); } while (0)

namespace aft {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
typedef intl intl;
typedef uintl uintl;
#pragma GCC diagnostic pop
}

using aft::uintl;
using aft::intl;

namespace {

typedef unsigned char  uchar;
typedef unsigned int   uint;
typedef unsigned short ushort;

enum TestOutputArrayType {
    NULL_ARRAY,
    FULL_ARRAY,
    SUB_ARRAY,
    REORDERED_ARRAY
};

struct TestOutputArrayInfo {
    af::array out_arr;
    af::array out_arr_cpy;
    af_array out_arr_ptr;
    af::index subarr_s0;
    af::index subarr_s1;
    af::index subarr_s2;
    af::index subarr_s3;
    TestOutputArrayType arr_type;
};

std::string readNextNonEmptyLine(std::ifstream &file)
{
    std::string result = "";
    // Using a for loop to read the next non empty line
    for (std::string line; std::getline(file, line);) {
        result += line;
        if (result != "") break;
    }
    // If no file has been found, throw an exception
    if (result == "") {
        throw std::runtime_error("Non empty lines not found in the file");
    }
    return result;
}

template<typename inType, typename outType, typename FileElementType>
void readTests(const std::string &FileName, std::vector<af::dim4> &inputDims,
                std::vector<std::vector<inType> >   &testInputs,
                std::vector<std::vector<outType> > &testOutputs)
{
    using std::vector;

    std::ifstream testFile(FileName.c_str());
    if(testFile.good()) {
        unsigned inputCount;
        testFile >> inputCount;
        inputDims.resize(inputCount);
        for(unsigned i=0; i<inputCount; i++) {
            testFile >> inputDims[i];
        }

        unsigned testCount;
        testFile >> testCount;
        testOutputs.resize(testCount);

        vector<unsigned> testSizes(testCount);
        for(unsigned i = 0; i < testCount; i++) {
            testFile >> testSizes[i];
        }

        testInputs.resize(inputCount,vector<inType>(0));
        for(unsigned k=0; k<inputCount; k++) {
            dim_t nElems = inputDims[k].elements();
            testInputs[k].resize(nElems);
            FileElementType tmp;
            for(unsigned i = 0; i < nElems; i++) {
                testFile >> tmp;
                testInputs[k][i] = static_cast<inType>(tmp);
            }
        }

        testOutputs.resize(testCount, vector<outType>(0));
        for(unsigned i = 0; i < testCount; i++) {
            testOutputs[i].resize(testSizes[i]);
            FileElementType tmp;
            for(unsigned j = 0; j < testSizes[i]; j++) {
                testFile >> tmp;
                testOutputs[i][j] = static_cast<outType>(tmp);
            }
        }
    }
    else {
        FAIL() << "TEST FILE NOT FOUND";
    }
}

template<typename inType, typename outType>
void readTestsFromFile(const std::string &FileName, std::vector<af::dim4> &inputDims,
                std::vector<std::vector<inType> >  &testInputs,
                std::vector<std::vector<outType> > &testOutputs)
{
    using std::vector;

    std::ifstream testFile(FileName.c_str());
    if(testFile.good()) {
        unsigned inputCount;
        testFile >> inputCount;
        for(unsigned i=0; i<inputCount; i++) {
            af::dim4 temp(1);
            testFile >> temp;
            inputDims.push_back(temp);
        }

        unsigned testCount;
        testFile >> testCount;
        testOutputs.resize(testCount);

        vector<unsigned> testSizes(testCount);
        for(unsigned i = 0; i < testCount; i++) {
            testFile >> testSizes[i];
        }

        testInputs.resize(inputCount,vector<inType>(0));
        for(unsigned k=0; k<inputCount; k++) {
            dim_t nElems = inputDims[k].elements();
            testInputs[k].resize(nElems);
            inType tmp;
            for(unsigned i = 0; i < nElems; i++) {
                testFile >> tmp;
                testInputs[k][i] = tmp;
            }
        }

        testOutputs.resize(testCount, vector<outType>(0));
        for(unsigned i = 0; i < testCount; i++) {
            testOutputs[i].resize(testSizes[i]);
            outType tmp;
            for(unsigned j = 0; j < testSizes[i]; j++) {
                testFile >> tmp;
                testOutputs[i][j] = tmp;
            }
        }
    }
    else {
        FAIL() << "TEST FILE NOT FOUND";
    }
}

inline void readImageTests(const std::string        &pFileName,
                           std::vector<af::dim4>    &pInputDims,
                           std::vector<std::string> &pTestInputs,
                           std::vector<dim_t>    &pTestOutSizes,
                           std::vector<std::string> &pTestOutputs)
{
    using std::vector;

    std::ifstream testFile(pFileName.c_str());
    if(testFile.good()) {
        unsigned inputCount;
        testFile >> inputCount;
        for(unsigned i=0; i<inputCount; i++) {
            af::dim4 temp(1);
            testFile >> temp;
            pInputDims.push_back(temp);
        }

        unsigned testCount;
        testFile >> testCount;
        pTestOutputs.resize(testCount);

        pTestOutSizes.resize(testCount);
        for(unsigned i = 0; i < testCount; i++) {
            testFile >> pTestOutSizes[i];
        }

        pTestInputs.resize(inputCount, "");
        for(unsigned k=0; k<inputCount; k++) {
            pTestInputs[k] = readNextNonEmptyLine(testFile);
        }

        pTestOutputs.resize(testCount, "");
        for(unsigned i = 0; i < testCount; i++) {
            pTestOutputs[i] = readNextNonEmptyLine(testFile);
        }
    }
    else {
        FAIL() << "TEST FILE NOT FOUND";
    }
}

template<typename outType>
void readImageTests(const std::string                 &pFileName,
                    std::vector<af::dim4>             &pInputDims,
                    std::vector<std::string>          &pTestInputs,
                    std::vector<std::vector<outType> > &pTestOutputs)
{
    using std::vector;

    std::ifstream testFile(pFileName.c_str());
    if(testFile.good()) {
        unsigned inputCount;
        testFile >> inputCount;
        for(unsigned i=0; i<inputCount; i++) {
            af::dim4 temp(1);
            testFile >> temp;
            pInputDims.push_back(temp);
        }

        unsigned testCount;
        testFile >> testCount;
        pTestOutputs.resize(testCount);

        vector<unsigned> testSizes(testCount);
        for(unsigned i = 0; i < testCount; i++) {
            testFile >> testSizes[i];
        }

        pTestInputs.resize(inputCount, "");
        for(unsigned k=0; k<inputCount; k++) {
            pTestInputs[k] = readNextNonEmptyLine(testFile);
        }

        pTestOutputs.resize(testCount, vector<outType>(0));
        for(unsigned i = 0; i < testCount; i++) {
            pTestOutputs[i].resize(testSizes[i]);
            outType tmp;
            for(unsigned j = 0; j < testSizes[i]; j++) {
                testFile >> tmp;
                pTestOutputs[i][j] = tmp;
            }
        }
    }
    else {
        FAIL() << "TEST FILE NOT FOUND";
    }
}

template<typename descType>
void readImageFeaturesDescriptors(const std::string                  &pFileName,
                                  std::vector<af::dim4>              &pInputDims,
                                  std::vector<std::string>           &pTestInputs,
                                  std::vector<std::vector<float> >    &pTestFeats,
                                  std::vector<std::vector<descType> > &pTestDescs)
{
    using std::vector;

    std::ifstream testFile(pFileName.c_str());
    if(testFile.good()) {
        unsigned inputCount;
        testFile >> inputCount;
        for(unsigned i=0; i<inputCount; i++) {
            af::dim4 temp(1);
            testFile >> temp;
            pInputDims.push_back(temp);
        }

        unsigned attrCount, featCount, descLen;
        testFile >> featCount;
        testFile >> attrCount;
        testFile >> descLen;
        pTestFeats.resize(attrCount);

        pTestInputs.resize(inputCount, "");
        for(unsigned k=0; k<inputCount; k++) {
            pTestInputs[k] = readNextNonEmptyLine(testFile);
        }

        pTestFeats.resize(attrCount, vector<float>(0));
        for(unsigned i = 0; i < attrCount; i++) {
            pTestFeats[i].resize(featCount);
            float tmp;
            for(unsigned j = 0; j < featCount; j++) {
                testFile >> tmp;
                pTestFeats[i][j] = tmp;
            }
        }

        pTestDescs.resize(featCount, vector<descType>(0));
        for(unsigned i = 0; i < featCount; i++) {
            pTestDescs[i].resize(descLen);
            descType tmp;
            for(unsigned j = 0; j < descLen; j++) {
                testFile >> tmp;
                pTestDescs[i][j] = tmp;
            }
        }
    }
    else {
        FAIL() << "TEST FILE NOT FOUND";
    }
}

/**
 * Below is not a pair wise comparition method, rather
 * it computes the accumulated error of the computed
 * output and gold output.
 *
 * The cut off is decided based on root mean square
 * deviation from cpu result
 *
 * For images, the maximum possible error will happen if all
 * the observed values are zeros and all the predicted values
 * are 255's. In such case, the value of NRMSD will be 1.0
 * Similarly, we can deduce that 0.0 will be the minimum
 * value of NRMSD. Hence, the range of RMSD is [0,255] for image inputs.
 */
template<typename T>
bool compareArraysRMSD(dim_t data_size, T *gold, T *data, double tolerance)
{
    double accum  = 0.0;
    double maxion = -FLT_MAX;//(double)std::numeric_limits<T>::lowest();
    double minion = FLT_MAX;//(double)std::numeric_limits<T>::max();

    for(dim_t i=0;i<data_size;i++)
    {
        double dTemp = (double)data[i];
        double gTemp = (double)gold[i];
        double diff  = gTemp-dTemp;
        double err   = (std::isfinite(diff) && (std::abs(diff) > 1.0e-4)) ? diff : 0.0f;
        accum  += std::pow(err, 2.0);
        maxion  = std::max(maxion, dTemp);
        minion  = std::min(minion, dTemp);
    }
    accum /= data_size;
    double NRMSD = std::sqrt(accum)/(maxion-minion);

    if (std::isnan(NRMSD) || NRMSD > tolerance) {
#ifndef NDEBUG
        printf("Comparison failed, NRMSD value: %lf\n", NRMSD);
#endif
        return false;
    }

    return true;
}

template<typename T, typename Other>
struct is_same_type {
    static const bool value = false;
};

template<typename T>
struct is_same_type<T, T> {
    static const bool value = true;
};

template<bool, typename T, typename O>
struct cond_type;

template<typename T, typename Other>
struct cond_type<true, T, Other> {
    typedef T type;
};

template<typename T, typename Other>
struct cond_type<false, T, Other> {
    typedef Other type;
};

template<bool B, class T = void>
struct enable_if {};

template<class T>
struct enable_if<true, T> { typedef T type; };

template<typename T>
inline double real(T val) { return (double)val; }
template<>
inline double real<af::cdouble>(af::cdouble val) { return real(val); }
template<>
inline double real<af::cfloat> (af::cfloat val) { return real(val); }

template<typename T>
inline double imag(T val) { return (double)val; }
template<>
inline double imag<af::cdouble>(af::cdouble val) { return imag(val); }
template<>
inline double imag<af::cfloat> (af::cfloat val) { return imag(val); }


template<class T>
struct IsFloatingPoint {
    static const bool value = is_same_type<float, T>::value  ||
        is_same_type<double, T>::value  ||
        is_same_type<long double, T>::value;
};

template<typename T>
bool noDoubleTests()
{
    af::dtype ty = (af::dtype)af::dtype_traits<T>::af_type;
    bool isTypeDouble = (ty == f64) || (ty == c64);
    int dev = af::getDevice();
    bool isDoubleSupported = af::isDoubleAvailable(dev);

    return ((isTypeDouble && !isDoubleSupported) ? true : false);
}

inline bool noImageIOTests()
{
    bool ret = !af::isImageIOAvailable();
    if(ret) printf("Image IO Not Configured. Test will exit\n");
    return ret;
}

inline bool noLAPACKTests()
{
    bool ret = !af::isLAPACKAvailable();
    if(ret) printf("LAPACK Not Configured. Test will exit\n");
    return ret;
}

// TODO: perform conversion on device for CUDA and OpenCL
template<typename T>
af_err conv_image(af_array *out, af_array in)
{
    af_array outArray;

    dim_t d0, d1, d2, d3;
    af_get_dims(&d0, &d1, &d2, &d3, in);
    af::dim4 idims(d0, d1, d2, d3);

    dim_t nElems = 0;
    af_get_elements(&nElems, in);

    float *in_data = new float[nElems];
    af_get_data_ptr(in_data, in);

    T *out_data = new T[nElems];

    for (int i = 0; i < (int)nElems; i++)
        out_data[i] = (T)in_data[i];

    af_create_array(&outArray, out_data, idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type);

    std::swap(*out, outArray);

    delete [] in_data;
    delete [] out_data;

    return AF_SUCCESS;
}

template<typename T>
af::array cpu_randu(const af::dim4 dims)
{
    typedef typename af::dtype_traits<T>::base_type BT;

    bool isTypeCplx = is_same_type<T, af::cfloat>::value || is_same_type<T, af::cdouble>::value;
    bool isTypeFloat = is_same_type<BT, float>::value || is_same_type<BT, double>::value;

    size_t elements = (isTypeCplx ? 2 : 1) * dims.elements();

    std::vector<BT> out(elements);
    for(size_t i = 0; i < elements; i++) {
        out[i] = isTypeFloat ? (BT)(rand())/RAND_MAX : rand() % 100;
    }

    return af::array(dims, (T *)&out[0]);
}

void cleanSlate()
{
  const size_t step_bytes = 1024;

  size_t alloc_bytes, alloc_buffers;
  size_t lock_bytes, lock_buffers;

  af::deviceGC();

  af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                    &lock_bytes, &lock_buffers);

  ASSERT_EQ(0u, alloc_buffers);
  ASSERT_EQ(0u, lock_buffers);
  ASSERT_EQ(0u, alloc_bytes);
  ASSERT_EQ(0u, lock_bytes);

  af::setMemStepSize(step_bytes);

  ASSERT_EQ(af::getMemStepSize(), step_bytes);
}

//********** arrayfire custom test asserts ***********

std::ostream& operator<<(std::ostream& os, af_err e) {
    return os << af_err_to_string(e);
}

std::ostream& operator<<(std::ostream& os, af::dtype type) {
    std::string name;
    switch (type) {
    case f32: name = "f32"; break;
    case c32: name = "c32"; break;
    case f64: name = "f64"; break;
    case c64: name = "c64"; break;
    case b8:  name = "b8";  break;
    case s32: name = "s32"; break;
    case u32: name = "u32"; break;
    case u8:  name = "u8";  break;
    case s64: name = "s64"; break;
    case u64: name = "u64"; break;
    case s16: name = "s16"; break;
    case u16: name = "u16"; break;
    default: assert(false && "Invalid type");
    }
    return os << name;
}

// Overloading unary + op is needed to make unsigned char values printable
//  as numbers

const af::cfloat& operator+(const af::cfloat& val) {
    return val;
}

const af::cdouble& operator+(const af::cdouble& val) {
    return val;
}

// Calculate a multi-dimensional coordinates' linearized index
dim_t ravelIdx(af::dim4 coords, af::dim4 strides) {
    return std::inner_product(coords.get(), coords.get()+4, strides.get(), 0LL);
}

// Calculate a linearized index's multi-dimensonal coordinates in an af::array,
//  given its dimension sizes and strides
af::dim4 unravelIdx(dim_t idx, af::dim4 dims, af::dim4 strides) {
    af::dim4 coords;
    coords[3] = idx / (strides[3]);
    coords[2] = idx / (strides[2]) % dims[2];
    coords[1] = idx / (strides[1]) % dims[1];
    coords[0] = idx % dims[0];

    return coords;
}

af::dim4 unravelIdx(dim_t idx, af::array arr) {
    af::dim4 dims = arr.dims();
    af::dim4 st = af::getStrides(arr);
    return unravelIdx(idx, dims, st);
}

af::dim4 calcStrides(const af::dim4 &parentDim)
{
    af::dim4 out(1, 1, 1, 1);
    dim_t *out_dims = out.get();
    const dim_t *parent_dims =  parentDim.get();

    for (dim_t i=1; i < 4; i++) {
        out_dims[i] = out_dims[i - 1] * parent_dims[i-1];
    }

    return out;
}

std::string minimalDim4(af::dim4 coords, af::dim4 dims) {
    std::ostringstream os;
    os << "(" << coords[0];
    if (dims[1] > 1 || dims[2] > 1 || dims[3] > 1) {
        os << ", " << coords[1];
    }
    if (dims[2] > 1 || dims[3] > 1) {
        os << ", " << coords[2];
    }
    if (dims[3] > 1) {
        os << ", " << coords[3];
    }
    os << ")";

    return os.str();
}

template<typename T>
std::string printContext(const std::vector<T>& hGold, std::string goldName,
                         const std::vector<T>& hOut, std::string outName,
                         af::dim4 arrDims,
                         af::dim4 arrStrides,
                         dim_t idx) {
    std::ostringstream os;

    af::dim4 coords = unravelIdx(idx, arrDims, arrStrides);
    dim_t ctxWidth = 5;

    // Coordinates that span dim0
    af::dim4 coordsMinBound = coords;
    coordsMinBound[0] = 0;
    af::dim4 coordsMaxBound = coords;
    coordsMaxBound[0] = arrDims[0] - 1;

    // dim0 positions that can be displayed
    dim_t dim0Start = std::max<dim_t>(0LL, coords[0] - ctxWidth);
    dim_t dim0End = std::min<dim_t>(coords[0] + ctxWidth + 1LL, arrDims[0]);

    // Linearized indices of values in vectors that can be displayed
    dim_t vecStartIdx = std::max<dim_t>(ravelIdx(coordsMinBound, arrStrides),
                                        idx - ctxWidth);

    // Display as minimal coordinates as needed
    // First value is the range of dim0 positions that will be displayed
    os << "Viewing slice (" << dim0Start << ":" << dim0End - 1;
    if (arrDims[1] > 1 || arrDims[2] > 1 || arrDims[3] > 1)
        os << ", " << coords[1];
    if (arrDims[2] > 1 || arrDims[3] > 1)
        os << ", " << coords[2];
    if (arrDims[3] > 1)
        os << ", " << coords[3];
    os << "), dims are (" << arrDims << ") strides: (" << arrStrides << ")\n";

    dim_t ctxElems = dim0End - dim0Start;
    std::vector<int> valFieldWidths(ctxElems);
    std::vector<std::string> ctxDim0(ctxElems);
    std::vector<std::string> ctxOutVals(ctxElems);
    std::vector<std::string> ctxGoldVals(ctxElems);

    // Get dim0 positions and out/reference values for the context window
    //
    // Also get the max string length between the position and out/ref values
    // per item so that it can be used later as the field width for
    // displaying each item in the context window
    for (dim_t i = 0; i < ctxElems; ++i) {
        std::ostringstream tmpOs;

        dim_t dim0 = dim0Start + i;
        if (dim0 == coords[0])
            tmpOs << "[" << dim0 << "]";
        else
            tmpOs << dim0;
        ctxDim0[i] = tmpOs.str();
        size_t dim0Len = tmpOs.str().length();
        tmpOs.str(std::string());

        dim_t valIdx = vecStartIdx + i;

        if (valIdx == idx) {
            tmpOs << "[" << +hOut[valIdx] << "]";
        }
        else {
            tmpOs << +hOut[valIdx];
        }
        ctxOutVals[i] = tmpOs.str();
        size_t outLen = tmpOs.str().length();
        tmpOs.str(std::string());

        if (valIdx == idx) {
            tmpOs << "[" << +hGold[valIdx] << "]";
        }
        else {
            tmpOs << +hGold[valIdx];
        }
        ctxGoldVals[i] = tmpOs.str();
        size_t goldLen = tmpOs.str().length();
        tmpOs.str(std::string());

        int maxWidth = std::max<int>(dim0Len, outLen);
        maxWidth = std::max<int>(maxWidth, goldLen);
        valFieldWidths[i] = maxWidth;
    }

    size_t varNameWidth = std::max<size_t>(goldName.length(), outName.length());

    // Display dim0 positions, output values, and reference values
    os << std::right << std::setw(varNameWidth) << "" << "   ";
    for (uint i = 0; i < (dim0End - dim0Start); ++i) {
        os << std::setw(valFieldWidths[i] + 1) << std::right << ctxDim0[i];
    }
    os << "\n";

    os << std::right << std::setw(varNameWidth) << outName << ": {";
    for (uint i = 0; i < (dim0End - dim0Start); ++i) {
        os << std::setw(valFieldWidths[i] + 1) << std::right << ctxOutVals[i];
    }
    os << " }\n";

    os << std::right << std::setw(varNameWidth) << goldName << ": {";
    for (uint i = 0; i < (dim0End - dim0Start); ++i) {
        os << std::setw(valFieldWidths[i] + 1) << std::right << ctxGoldVals[i];
    }
    os << " }";

    return os.str();
}

struct FloatTag {};
struct IntegerTag {};

template<typename T>
::testing::AssertionResult elemWiseEq(std::string aName, std::string bName,
                                      const std::vector<T>& a, af::dim4 aDims,
                                      const std::vector<T>& b, af::dim4 bDims,
                                      float maxAbsDiff, IntegerTag) {
    UNUSED(maxAbsDiff);
    typedef typename std::vector<T>::const_iterator iter;
    std::pair<iter, iter> mismatches = std::mismatch(a.begin(), a.end(), b.begin());
    iter bItr = mismatches.second;

    if (bItr == b.end()) {
        return ::testing::AssertionSuccess();
    } else {
        dim_t idx = std::distance(b.begin(), bItr);
        af::dim4 aStrides = calcStrides(aDims);
        af::dim4 bStrides = calcStrides(bDims);
        af::dim4 coords = unravelIdx(idx, bDims, bStrides);

        return ::testing::AssertionFailure()
            << "VALUE DIFFERS at "
            << minimalDim4(coords, aDims) << ":\n"
            << printContext(a, aName, b, bName, aDims, aStrides, idx);
    }
}

struct absMatch{
    float diff_;
    absMatch(float diff) : diff_(diff) {}

    template<typename T>
    bool operator() (T lhs, T rhs) {
        using std::abs;
        using af::abs;
        return abs(rhs - lhs) <= diff_;
    }
};

template<typename T>
::testing::AssertionResult elemWiseEq(std::string aName, std::string bName,
                                      const std::vector<T>& a, af::dim4 aDims,
                                      const std::vector<T>& b, af::dim4 bDims,
                                      float maxAbsDiff, FloatTag) {
    typedef typename std::vector<T>::const_iterator iter;
    // TODO(mark): Modify equality for float
    std::pair<iter, iter> mismatches = std::mismatch(a.begin(), a.end(),
                                                     b.begin(),
                                                     absMatch(maxAbsDiff));

    iter aItr = mismatches.first;
    iter bItr = mismatches.second;

    if (aItr == a.end()) {
        return ::testing::AssertionSuccess();
    } else {
        dim_t idx = std::distance(b.begin(), bItr);
        af::dim4 coords = unravelIdx(idx, bDims, calcStrides(bDims));

        af::dim4 aStrides = calcStrides(aDims);

        ::testing::AssertionResult result =
        ::testing::AssertionFailure()
              << "VALUE DIFFERS at "
              << minimalDim4(coords, aDims) << ":\n"
              << printContext(a, aName, b, bName, aDims, aStrides, idx);

        if(maxAbsDiff > 0) {
            using std::abs;
            using af::abs;
            double absdiff = abs(*aItr - *bItr);
            result << "\n  Actual diff: " << absdiff << "\n"
                   << "Expected diff: " << maxAbsDiff;
        }

        return result;
    }
}

template<typename T>
::testing::AssertionResult elemWiseEq(std::string aName, std::string bName,
                                      const af::array& a, const af::array& b,
                                      float maxAbsDiff) {
    typedef typename cond_type<
        IsFloatingPoint<typename af::dtype_traits<T>::base_type>::value,
        FloatTag, IntegerTag>::type TagType;
    TagType tag;

    std::vector<T> hA(static_cast<size_t>(a.elements()));
    a.host(hA.data());

    std::vector<T> hB(static_cast<size_t>(b.elements()));
    b.host(hB.data());
    return elemWiseEq<T>(aName, bName, hA, a.dims(), hB, b.dims(), maxAbsDiff, tag);
}

// Declaration used by testWriteToOutputArray
::testing::AssertionResult assertArrayEq(std::string aName, std::string bName,
                                         const af::array& a, const af::array& b,
                                         float maxAbsDiff = 0.f);

::testing::AssertionResult
testWriteToOutputArray(std::string gold_name, std::string result_name,
                       af::array gold, TestOutputArrayInfo& metadata) {
    if (metadata.arr_type == SUB_ARRAY) {
        // "Paste" the gold subarray to the large array
        af::copy(metadata.out_arr_cpy, gold,
                 metadata.subarr_s0,
                 metadata.subarr_s1,
                 metadata.subarr_s2,
                 metadata.subarr_s3);

        // Perform the element-wise check on the two large arrays
        return assertArrayEq(gold_name, result_name,
                             metadata.out_arr_cpy, metadata.out_arr);
    }
    else {
        return assertArrayEq(gold_name, result_name, gold, metadata.out_arr);
    }
}

// Partner function of genTestOutputArray. This uses the same "special"
// array that genTestOutputArray generates, and checks whether the
// af_* function wrote to that array correctly
::testing::AssertionResult
testWriteToOutputArray(std::string gold_name, std::string result_name,
                       af_array gold, af_array out,
                       TestOutputArrayInfo *metadata) {
    // In the case of NULL_ARRAY, metadata->out_arr starts as null
    // The af function generates a new output array, hence
    // metadata->out_arr needs to contain that new output array
    if (metadata->arr_type == NULL_ARRAY) {
        af_array out_retain = 0;
        af_retain_array(&out_retain, out);
        metadata->out_arr = af::array(out_retain);
    }
    // For every other case, must check if the af_array provided by gen*Array
    // was used by the af function as its output array
    else {
        if (metadata->out_arr_ptr != out) {
            return ::testing::AssertionFailure()
                << "af_array POINTER MISMATCH:\n"
                << "  Actual: " << out << "\n"
                << "Expected: " << metadata->out_arr_ptr;
        }
    }

    af_array gold_retain = 0;
    af_retain_array(&gold_retain, gold);
    af::array gold_cpp(gold_retain);

    return testWriteToOutputArray(gold_name, result_name, gold_cpp, *metadata);
}

// Called by ASSERT_SPECIAL_ARRAYS_EQ
::testing::AssertionResult assertArrayEq(std::string aName, std::string bName,
                                         std::string metadataName,
                                         const af_array a, const af_array b,
                                         TestOutputArrayInfo *metadata) {
    // b is only used to check if the output af_array's pointer value matches
    // what's expected. The actual element-wise check uses metadata->out_arr
    // See testWriteToOutputArray for more details
    return testWriteToOutputArray(aName, bName, a, b, metadata);
}

// Unused for now since use of existing arrays as outputs is not currently
// supported in the C++ API
::testing::AssertionResult assertArrayEq(std::string aName, std::string bName,
                                         std::string metadataName,
                                         const af::array& a, const af::array& b,
                                         TestOutputArrayInfo &metadata) {
    return testWriteToOutputArray(aName, bName, a, metadata);
}

// Called by ASSERT_ARRAYS_EQ
::testing::AssertionResult assertArrayEq(std::string aName, std::string bName,
                                         const af::array& a, const af::array& b,
                                         float maxAbsDiff) {
    af::dtype aType = a.type();
    af::dtype bType = b.type();
    if (aType != bType)
        return ::testing::AssertionFailure()
            << "TYPE MISMATCH: \n"
            << "  Actual: " << bName << "(" << b.type() << ")\n"
            << "Expected: " << aName << "(" << a.type() << ")";

    af::dtype arrDtype = aType;
    if (a.dims() != b.dims())
        return ::testing::AssertionFailure()
            << "SIZE MISMATCH: \n"
            << "  Actual: " << bName << "([" << b.dims() << "])\n"
            << "Expected: " << aName << "([" << a.dims() << "])";

    switch (arrDtype) {
    case f32: return elemWiseEq<float>              (aName, bName, a, b, maxAbsDiff); break;
    case c32: return elemWiseEq<af::cfloat>         (aName, bName, a, b, maxAbsDiff); break;
    case f64: return elemWiseEq<double>             (aName, bName, a, b, maxAbsDiff); break;
    case c64: return elemWiseEq<af::cdouble>        (aName, bName, a, b, maxAbsDiff); break;
    case b8:  return elemWiseEq<char>               (aName, bName, a, b, maxAbsDiff); break;
    case s32: return elemWiseEq<int>                (aName, bName, a, b, maxAbsDiff); break;
    case u32: return elemWiseEq<uint>               (aName, bName, a, b, maxAbsDiff); break;
    case u8:  return elemWiseEq<uchar>              (aName, bName, a, b, maxAbsDiff); break;
    case s64: return elemWiseEq<long long>          (aName, bName, a, b, maxAbsDiff); break;
    case u64: return elemWiseEq<unsigned long long> (aName, bName, a, b, maxAbsDiff); break;
    case s16: return elemWiseEq<short>              (aName, bName, a, b, maxAbsDiff); break;
    case u16: return elemWiseEq<unsigned short>     (aName, bName, a, b, maxAbsDiff); break;
    default:  return ::testing::AssertionFailure()
            << "INVALID TYPE, see enum numbers: "
            << bName << "(" << b.type() << ") and "
            << aName << "(" << a.type() << ")";
    }

    return ::testing::AssertionSuccess();
}

// Called by ASSERT_VEC_ARRAY_EQ
template<typename T>
::testing::AssertionResult assertArrayEq(std::string aName, std::string aDimsName,
                                         std::string bName,
                                         const std::vector<T>& hA, af::dim4 aDims,
                                         const af::array& b,
                                         float maxAbsDiff = 0.0f) {
    af::dtype aDtype = (af::dtype) af::dtype_traits<T>::af_type;
    if (aDtype != b.type()) {
        return ::testing::AssertionFailure()
            << "TYPE MISMATCH:\n"
            << "  Actual: " << bName << "(" << b.type() << ")\n"
            << "Expected: " << aName << "(" << aDtype << ")";
    }

    if(aDims != b.dims()) {
        return ::testing::AssertionFailure()
            << "SIZE MISMATCH:\n"
            << "  Actual: " << bName << "([" << b.dims() << "])\n"
            << "Expected: " << aDimsName << "([" << aDims << "])";
    }

    // In case vector<T> a.size() != aDims.elements()
    if (hA.size() != static_cast<size_t>(aDims.elements()))
        return ::testing::AssertionFailure()
            << "SIZE MISMATCH:\n"
            << "  Actual: " << aDimsName << "([" << aDims << "] => "
            << aDims.elements() << ")\n"
            << "Expected: " << aName << ".size()(" << hA.size() << ")";

    typedef typename cond_type<
        IsFloatingPoint<typename af::dtype_traits<T>::base_type>::value,
        FloatTag, IntegerTag>::type TagType;
    TagType tag;

    std::vector<T> hB(b.elements());
    b.host(&hB.front());
    return elemWiseEq<T>(aName, bName, hA, aDims, hB, b.dims(), maxAbsDiff, tag);
}

// To support C API
::testing::AssertionResult assertArrayEq(std::string aName, std::string bName,
                                         const af_array a, const af_array b) {
    af_array aa = 0, bb = 0;
    af_retain_array(&aa, a);
    af_retain_array(&bb, b);
    af::array aaa(aa);
    af::array bbb(bb);
    return assertArrayEq(aName, bName, aaa, bbb, 0.0f);
}

// To support C API
template<typename T>
::testing::AssertionResult assertArrayEq(std::string hA_name, std::string aDimsName,
                                         std::string bName,
                                         const std::vector<T>& hA, af::dim4 aDims,
                                         const af_array b) {
    af_array bb = 0;
    af_retain_array(&bb, b);
    af::array bbb(bb);
    return assertArrayEq(hA_name, aDimsName, bName, hA, aDims, bbb);
}

// Called by ASSERT_ARRAYS_NEAR
::testing::AssertionResult assertArrayNear(std::string aName, std::string bName,
                                           std::string maxAbsDiffName,
                                           const af::array& a, const af::array& b,
                                           float maxAbsDiff) {
    UNUSED(maxAbsDiffName);
    return assertArrayEq(aName, bName, a, b, maxAbsDiff);
}

// Called by ASSERT_VEC_ARRAY_NEAR
template<typename T>
::testing::AssertionResult assertArrayNear(std::string hA_name, std::string aDimsName,
                                           std::string bName,
                                           std::string maxAbsDiffName,
                                           const std::vector<T>& hA, af::dim4 aDims,
                                           const af::array& b,
                                           float maxAbsDiff) {
    UNUSED(maxAbsDiffName);
    return assertArrayEq(hA_name, aDimsName, bName, hA, aDims, b, maxAbsDiff);
}

// To support C API
::testing::AssertionResult assertArrayNear(std::string aName, std::string bName,
                                           std::string maxAbsDiffName,
                                           const af_array a, const af_array b,
                                           float maxAbsDiff) {
    af_array aa = 0, bb = 0;
    af_retain_array(&aa, a);
    af_retain_array(&bb, b);
    af::array aaa(aa);
    af::array bbb(bb);
    return assertArrayNear(aName, bName, maxAbsDiffName, aaa, bbb, maxAbsDiff);
}

// To support C API
template<typename T>
::testing::AssertionResult assertArrayNear(std::string hA_name, std::string aDimsName,
                                           std::string bName,
                                           std::string maxAbsDiffName,
                                           const std::vector<T>& hA, af::dim4 aDims,
                                           const af_array b,
                                           float maxAbsDiff) {
    af_array bb = 0;
    af_retain_array(&bb, b);
    af::array bbb(bb);
    return assertArrayNear(hA_name, aDimsName, maxAbsDiffName, bName, hA, aDims,
                         bbb, maxAbsDiff);
}

/// Checks if the C-API arrayfire function returns successfully
///
/// \param[in] CALL This is the arrayfire C function
#define ASSERT_SUCCESS(CALL)                    \
    ASSERT_EQ(AF_SUCCESS, CALL)

/// Compares two af::array or af_arrays for their types, dims, and values (strict equality).
///
/// \param[in] EXPECTED The expected array of the assertion
/// \param[in] ACTUAL The actual resulting array from the calculation
#define ASSERT_ARRAYS_EQ(EXPECTED, ACTUAL) \
    EXPECT_PRED_FORMAT2(assertArrayEq, EXPECTED, ACTUAL)

/// Same as ASSERT_ARRAYS_EQ, but for cases when a "special" output array is
/// given to the function.
/// The special array can be null, a full-sized array, a subarray, or reordered
/// Can only be used for testing C-API functions currently
///
/// \param[in] EXPECTED The expected array of the assertion
/// \param[in] ACTUAL The actual resulting array from the calculation
#define ASSERT_SPECIAL_ARRAYS_EQ(EXPECTED, ACTUAL, META)               \
    EXPECT_PRED_FORMAT3(assertArrayEq, EXPECTED, ACTUAL, META)

/// Compares a std::vector with an af::/af_array for their types, dims, and values (strict equality).
///
/// \param[in] EXPECTED_VEC The vector that represents the expected array
/// \param[in] EXPECTED_ARR_DIMS The dimensions of the expected array
/// \param[in] ACTUAL_ARR The actual resulting array from the calculation
#define ASSERT_VEC_ARRAY_EQ(EXPECTED_VEC, EXPECTED_ARR_DIMS, ACTUAL_ARR) \
    EXPECT_PRED_FORMAT3(assertArrayEq, EXPECTED_VEC, EXPECTED_ARR_DIMS, ACTUAL_ARR)

/// Compares two af::array or af_arrays for their type, dims, and values (with a given tolerance).
///
/// \param[in] EXPECTED Expected value of the assertion
/// \param[in] ACTUAL Actual value of the calculation
/// \param[in] MAX_ABSDIFF Expected maximum absolute difference between
///            elements of EXPECTED and ACTUAL
///
/// \NOTE: This macro will deallocate the af_arrays after the call
#define ASSERT_ARRAYS_NEAR(EXPECTED, ACTUAL, MAX_ABSDIFF) \
    EXPECT_PRED_FORMAT3(assertArrayNear, EXPECTED, ACTUAL, MAX_ABSDIFF)

/// Compares a std::vector with an af::array for their dims and values (with a given tolerance).
///
/// \param[in] EXPECTED_VEC The vector that represents the expected array
/// \param[in] EXPECTED_ARR_DIMS The dimensions of the expected array
/// \param[in] ACTUAL_ARR The actual array from the calculation
/// \param[in] MAX_ABSDIFF Expected maximum absolute difference between
///            elements of EXPECTED and ACTUAL
#define ASSERT_VEC_ARRAY_NEAR(EXPECTED_VEC, EXPECTED_ARR_DIMS, ACTUAL_ARR, MAX_ABSDIFF) \
    EXPECT_PRED_FORMAT4(assertArrayNear, EXPECTED_VEC, EXPECTED_ARR_DIMS, ACTUAL_ARR, \
                            MAX_ABSDIFF)

}

// Generates a null af_array. testWriteToOutputArray expects that the array that
// it receives after the af_* function is a valid, allocated af_array
af::array genNullArray(const af::dim4& dims, const af::dtype ty,
                       TestOutputArrayInfo& metadata) {
    af::array out;
    metadata.out_arr = out;
    metadata.subarr_s0 = af::span;
    metadata.subarr_s1 = af::span;
    metadata.subarr_s2 = af::span;
    metadata.subarr_s3 = af::span;
    return metadata.out_arr;
}

// Generates a random array. testWriteToOutputArray expects that it will receive
// the same af_array that this generates after the af_* function is called
af::array genRegularArray(const af::dim4& dims, const af::dtype ty,
                          TestOutputArrayInfo& metadata) {
    metadata.out_arr = af::randu(dims, ty);
    metadata.out_arr_cpy = metadata.out_arr.copy();
    metadata.subarr_s0 = af::span;
    metadata.subarr_s1 = af::span;
    metadata.subarr_s2 = af::span;
    metadata.subarr_s3 = af::span;
    return metadata.out_arr;
}

// Generates a large, random array, and returns a subarray for the af_* function
// to use. testWriteToOutputArray expects that the large array that it receives is
// equal to the same large array with the gold array pasted on the same subarray location
af::array genSubArray(const af::dim4& dims, const af::dtype ty,
                      TestOutputArrayInfo& metadata) {
    const dim_t pad_size = 2;

    // Generate a large array that's padded on both sides of each dimension
    // Padding only applied if the dimension is used, i.e. if dims[i] > 1
    af::dim4 full_arr_dims(dims[0] > 1 ? dims[0] + 2*pad_size : dims[0],
                           dims[1] > 1 ? dims[1] + 2*pad_size : dims[1],
                           dims[2] > 1 ? dims[2] + 2*pad_size : dims[2],
                           dims[3] > 1 ? dims[3] + 2*pad_size : dims[3]);
    af::array out_arr = af::randu(full_arr_dims, ty);

    // Calculate index of sub-array. These will be used also by
    // testWriteToOutputArray so that the gold sub array will be placed in the
    // same location
    af::seq subarr_s0 =
        dims[0] > 1 ? af::seq(pad_size, pad_size + dims[0] - 1) : af::span;
    af::seq subarr_s1 =
        dims[1] > 1 ? af::seq(pad_size, pad_size + dims[1] - 1) : af::span;
    af::seq subarr_s2 =
        dims[2] > 1 ? af::seq(pad_size, pad_size + dims[2] - 1) : af::span;
    af::seq subarr_s3 =
        dims[3] > 1 ? af::seq(pad_size, pad_size + dims[3] - 1) : af::span;
    af::array subarr = out_arr(subarr_s0, subarr_s1, subarr_s2, subarr_s3);

    metadata.out_arr = out_arr;
    metadata.out_arr_cpy = out_arr.copy();
    metadata.subarr_s0 = subarr_s0;
    metadata.subarr_s1 = subarr_s1;
    metadata.subarr_s2 = subarr_s2;
    metadata.subarr_s3 = subarr_s3;

    return subarr;
}

// Generates a reordered array. testWriteToOutputArray expects that this array
// will still have the correct output values from the af_* function, even though
// the array was initially reordered.
af::array genReorderedArray(const af::dim4& dims, const af::dtype ty,
                            TestOutputArrayInfo& metadata) {
    // This reorder combination will not move data around, but will simply
    // call modDims and modStrides (see src/api/c/reorder.cpp).
    // Thus this will test if the output is still correct even with the
    // modified dims and strides "hack"
    unsigned reorder_0 = 0;
    unsigned reorder_1 = 2;
    unsigned reorder_2 = 1;

    // Shape the output array such that the reordered output array will have
    // the correct dimensions that the test asks for (i.e. will match dims)
    af::dim4 out_dims(dims[reorder_0], dims[reorder_1], dims[reorder_2]);
    metadata.out_arr = af::randu(out_dims, ty);
    metadata.out_arr = af::reorder(metadata.out_arr,
                                   reorder_0, reorder_1, reorder_2);
    metadata.out_arr_cpy = metadata.out_arr.copy();
    metadata.subarr_s0 = af::span;
    metadata.subarr_s1 = af::span;
    metadata.subarr_s2 = af::span;
    metadata.subarr_s3 = af::span;
    return metadata.out_arr;
}

af::array genTestOutputArray(const af::dim4& dims, const af::dtype ty,
                             TestOutputArrayInfo& metadata,
                             TestOutputArrayType arr_type) {
    metadata.arr_type = arr_type;
    switch (arr_type) {
    case NULL_ARRAY:
        return genNullArray(dims, ty, metadata);
        break;
    case FULL_ARRAY:
        return genRegularArray(dims, ty, metadata);
        break;
    case SUB_ARRAY:
        return genSubArray(dims, ty, metadata);
        break;
    case REORDERED_ARRAY:
        return genReorderedArray(dims, ty, metadata);
        break;
    }
}

// Partner function of testWriteToOutputArray. This generates the "special"
// array that testWriteToOutputArray will use to check if the af_* function
// correctly uses an existing array as its output
void genTestOutputArray(af_array *out, const unsigned ndims, const dim_t *dims,
                        const af::dtype ty, TestOutputArrayInfo* metadata,
                        TestOutputArrayType arr_type) {
    af::dim4 arr_dims(ndims, dims);
    af::array test_output_array = genTestOutputArray(arr_dims, ty, *metadata,
                                                     arr_type);
    af_retain_array(out, test_output_array.get());

    // Empty af::array will still have a non-null af_array inside. Thus need to
    // force out to be 0 for NULL_ARRAY case
    if (arr_type == NULL_ARRAY) {
        *out = 0;
    }

    metadata->out_arr_ptr = *out;
}

#pragma GCC diagnostic pop
