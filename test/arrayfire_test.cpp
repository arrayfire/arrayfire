/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define EXTERN_TEMPLATE
#include <testHelpers.hpp>

#include <arrayfire.h>
#include <af/algorithm.h>
#include <af/compatible.h>
#include <af/internal.h>

#include <gtest/gtest.h>
#include <half.hpp>
#include <relative_difference.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

using af::af_cdouble;
using af::af_cfloat;
using std::vector;

bool operator==(const af_half &lhs, const af_half &rhs) {
    return lhs.data_ == rhs.data_;
}

std::ostream &operator<<(std::ostream &os, const af_half &val) {
    float out = *reinterpret_cast<const half_float::half *>(&val);
    os << out;
    return os;
}

std::ostream &operator<<(std::ostream &os, af::Backend bk) {
    switch (bk) {
        case AF_BACKEND_CPU: os << "AF_BACKEND_CPU"; break;
        case AF_BACKEND_CUDA: os << "AF_BACKEND_CUDA"; break;
        case AF_BACKEND_OPENCL: os << "AF_BACKEND_OPENCL"; break;
        case AF_BACKEND_DEFAULT: os << "AF_BACKEND_DEFAULT"; break;
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, af_err e) {
    return os << af_err_to_string(e);
}

std::ostream &operator<<(std::ostream &os, af::dtype type) {
    std::string name;
    switch (type) {
        case f32: name = "f32"; break;
        case c32: name = "c32"; break;
        case f64: name = "f64"; break;
        case c64: name = "c64"; break;
        case b8: name = "b8"; break;
        case s32: name = "s32"; break;
        case u32: name = "u32"; break;
        case u8: name = "u8"; break;
        case s64: name = "s64"; break;
        case u64: name = "u64"; break;
        case s16: name = "s16"; break;
        case u16: name = "u16"; break;
        case f16: name = "f16"; break;
        default: assert(false && "Invalid type");
    }
    return os << name;
}

std::string readNextNonEmptyLine(std::ifstream &file) {
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

std::string getBackendName() {
    af::Backend backend = af::getActiveBackend();
    if (backend == AF_BACKEND_OPENCL)
        return std::string("opencl");
    else if (backend == AF_BACKEND_CUDA)
        return std::string("cuda");

    return std::string("cpu");
}

std::string getTestName() {
    std::string testname =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    return testname;
}

namespace half_float {
std::ostream &operator<<(std::ostream &os, half_float::half val) {
    os << (float)val;
    return os;
}
}  // namespace half_float

// Called by ASSERT_ARRAYS_EQ
::testing::AssertionResult assertArrayEq(std::string aName, std::string bName,
                                         const af::array &a, const af::array &b,
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
        case f32:
            return elemWiseEq<float>(aName, bName, a, b, maxAbsDiff);
            break;
        case c32:
            return elemWiseEq<af::cfloat>(aName, bName, a, b, maxAbsDiff);
            break;
        case f64:
            return elemWiseEq<double>(aName, bName, a, b, maxAbsDiff);
            break;
        case c64:
            return elemWiseEq<af::cdouble>(aName, bName, a, b, maxAbsDiff);
            break;
        case b8: return elemWiseEq<char>(aName, bName, a, b, maxAbsDiff); break;
        case s32: return elemWiseEq<int>(aName, bName, a, b, maxAbsDiff); break;
        case u32:
            return elemWiseEq<uint>(aName, bName, a, b, maxAbsDiff);
            break;
        case u8:
            return elemWiseEq<uchar>(aName, bName, a, b, maxAbsDiff);
            break;
        case s64:
            return elemWiseEq<long long>(aName, bName, a, b, maxAbsDiff);
            break;
        case u64:
            return elemWiseEq<unsigned long long>(aName, bName, a, b,
                                                  maxAbsDiff);
            break;
        case s16:
            return elemWiseEq<short>(aName, bName, a, b, maxAbsDiff);
            break;
        case u16:
            return elemWiseEq<unsigned short>(aName, bName, a, b, maxAbsDiff);
            break;
        case f16:
            return elemWiseEq<half_float::half>(aName, bName, a, b, maxAbsDiff);
            break;
        default:
            return ::testing::AssertionFailure()
                   << "INVALID TYPE, see enum numbers: " << bName << "("
                   << b.type() << ") and " << aName << "(" << a.type() << ")";
    }

    return ::testing::AssertionSuccess();
}

template<typename T>
::testing::AssertionResult imageEq(std::string aName, std::string bName,
                                   const af::array &a, const af::array &b,
                                   float maxAbsDiff) {
    std::vector<T> avec(a.elements());
    a.host(avec.data());
    std::vector<T> bvec(b.elements());
    b.host(bvec.data());
    double NRMSD = computeArraysRMSD(a.elements(), avec.data(), bvec.data());

    if (NRMSD < maxAbsDiff) {
        return ::testing::AssertionSuccess();
    } else {
        std::string test_name =
            ::testing::UnitTest::GetInstance()->current_test_info()->name();

        std::string valid_path =
            std::string(TEST_RESULT_IMAGE_DIR) + test_name + "ValidImage.png";
        std::string result_path =
            std::string(TEST_RESULT_IMAGE_DIR) + test_name + "ResultImage.png";
        std::string diff_path =
            std::string(TEST_RESULT_IMAGE_DIR) + test_name + "DiffImage.png";

        // af::array img = af::join(1, a, b);
        // af::Window win;
        // while (!win.close()) { win.image(img); }
        af::saveImage(valid_path.c_str(), a.as(f32));
        af::saveImage(result_path.c_str(), b.as(f32));
        af::saveImage(diff_path.c_str(), abs(a.as(f32) - b.as(f32)));

        std::cout
            << "<DartMeasurementFile type=\"image/png\" name=\"ValidImage\">"
            << valid_path << "</DartMeasurementFile>\n";
        std::cout
            << "<DartMeasurementFile type=\"image/png\" name=\"TestImage\">"
            << result_path << "</DartMeasurementFile>\n";

        std::cout << "<DartMeasurementFile "
                  << "type=\"image/png\" name=\"DifferenceImage2\">"
                  << diff_path << "</DartMeasurementFile>\n";

        return ::testing::AssertionFailure()
               << "RMSD Error(" << NRMSD << ") exceeds threshold(" << maxAbsDiff
               << "): " << bName << "(" << b.type() << ") and " << aName << "("
               << a.type() << ")";
    }
}

// Called by ASSERT_ARRAYS_EQ
::testing::AssertionResult assertImageEq(std::string aName, std::string bName,
                                         const af::array &a, const af::array &b,
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
        case u8: return imageEq<unsigned char>(aName, bName, a, b, maxAbsDiff);
        case b8: return imageEq<char>(aName, bName, a, b, maxAbsDiff);
        case s32: return imageEq<int>(aName, bName, a, b, maxAbsDiff);
        case u32: return imageEq<unsigned int>(aName, bName, a, b, maxAbsDiff);
        case f32: return imageEq<float>(aName, bName, a, b, maxAbsDiff);
        case f64: return imageEq<double>(aName, bName, a, b, maxAbsDiff);
        case s16: return imageEq<short>(aName, bName, a, b, maxAbsDiff);
        case u16:
            return imageEq<unsigned short>(aName, bName, a, b, maxAbsDiff);
        case u64:
            return imageEq<unsigned long long>(aName, bName, a, b, maxAbsDiff);
        case s64: return imageEq<long long>(aName, bName, a, b, maxAbsDiff);
        default: throw(AF_ERR_NOT_SUPPORTED);
    }
    return ::testing::AssertionSuccess();
}

template<>
float convert(af::half in) {
    return static_cast<float>(half_float::half(in.data_));
}

template<>
af_half convert(int in) {
    half_float::half h = half_float::half(in);
    af_half out;
    memcpy(&out, &h, sizeof(af_half));
    return out;
}

template<typename inType, typename outType, typename FileElementType>
void readTests(const std::string &FileName, std::vector<af::dim4> &inputDims,
               std::vector<std::vector<inType>> &testInputs,
               std::vector<std::vector<outType>> &testOutputs) {
    using std::vector;

    std::ifstream testFile(FileName.c_str());
    if (testFile.good()) {
        unsigned inputCount;
        testFile >> inputCount;
        inputDims.resize(inputCount);
        for (unsigned i = 0; i < inputCount; i++) { testFile >> inputDims[i]; }

        unsigned testCount;
        testFile >> testCount;
        testOutputs.resize(testCount);

        vector<unsigned> testSizes(testCount);
        for (unsigned i = 0; i < testCount; i++) { testFile >> testSizes[i]; }

        testInputs.resize(inputCount, vector<inType>(0));
        for (unsigned k = 0; k < inputCount; k++) {
            dim_t nElems = inputDims[k].elements();
            testInputs[k].resize(nElems);
            FileElementType tmp;
            for (unsigned i = 0; i < nElems; i++) {
                testFile >> tmp;
                testInputs[k][i] = convert<inType, FileElementType>(tmp);
            }
        }

        testOutputs.resize(testCount, vector<outType>(0));
        for (unsigned i = 0; i < testCount; i++) {
            testOutputs[i].resize(testSizes[i]);
            FileElementType tmp;
            for (unsigned j = 0; j < testSizes[i]; j++) {
                testFile >> tmp;
                testOutputs[i][j] = convert<outType, FileElementType>(tmp);
            }
        }
    } else {
        FAIL() << "TEST FILE NOT FOUND";
    }
}

#define INSTANTIATE(Tin, Tout, Tfile)                                  \
    template void readTests<Tin, Tout, Tfile>(                         \
        const std::string &FileName, std::vector<af::dim4> &inputDims, \
        std::vector<std::vector<Tin>> &testInputs,                     \
        std::vector<std::vector<Tout>> &testOutputs)

INSTANTIATE(float, float, int);
INSTANTIATE(double, float, int);
INSTANTIATE(int, float, int);
INSTANTIATE(unsigned int, float, int);
INSTANTIATE(char, float, int);
INSTANTIATE(unsigned char, float, int);
INSTANTIATE(short, float, int);
INSTANTIATE(unsigned short, float, int);
INSTANTIATE(long long, float, int);
INSTANTIATE(unsigned long long, float, int);
INSTANTIATE(af_cfloat, af_cfloat, int);
INSTANTIATE(double, double, int);
INSTANTIATE(af_cdouble, af_cdouble, int);
INSTANTIATE(int, int, int);
INSTANTIATE(unsigned int, unsigned int, int);
INSTANTIATE(unsigned int, unsigned int, unsigned int);
INSTANTIATE(long long, long long, int);
INSTANTIATE(unsigned long long, unsigned long long, int);
INSTANTIATE(char, char, int);
INSTANTIATE(unsigned char, unsigned char, int);
INSTANTIATE(short, short, int);
INSTANTIATE(unsigned short, unsigned short, int);
INSTANTIATE(half_float::half, half_float::half, int);
INSTANTIATE(af_half, af_half, int);
INSTANTIATE(float, int, int);
INSTANTIATE(unsigned int, int, int);
INSTANTIATE(char, int, int);
INSTANTIATE(unsigned char, int, int);
INSTANTIATE(short, int, int);
INSTANTIATE(unsigned short, int, int);

INSTANTIATE(unsigned char, unsigned short, int);
INSTANTIATE(unsigned char, short, int);
INSTANTIATE(unsigned char, double, int);

INSTANTIATE(long long, unsigned int, unsigned int);
INSTANTIATE(unsigned long long, unsigned int, unsigned int);
INSTANTIATE(int, unsigned int, unsigned int);
INSTANTIATE(short, unsigned int, unsigned int);
INSTANTIATE(unsigned short, unsigned int, unsigned int);
INSTANTIATE(char, unsigned int, unsigned int);
INSTANTIATE(unsigned char, unsigned int, unsigned int);
INSTANTIATE(float, unsigned int, unsigned int);
INSTANTIATE(double, unsigned int, unsigned int);

INSTANTIATE(float, unsigned int, int);
INSTANTIATE(double, unsigned int, int);
INSTANTIATE(int, unsigned int, int);
INSTANTIATE(long long, unsigned int, int);
INSTANTIATE(unsigned long long, unsigned int, int);
INSTANTIATE(char, unsigned int, int);
INSTANTIATE(unsigned char, unsigned int, int);
INSTANTIATE(short, unsigned int, int);
INSTANTIATE(unsigned short, unsigned int, int);

INSTANTIATE(float, char, int);
INSTANTIATE(double, char, int);
INSTANTIATE(unsigned char, char, int);
INSTANTIATE(short, char, int);
INSTANTIATE(unsigned short, char, int);
INSTANTIATE(int, char, int);
INSTANTIATE(unsigned int, char, int);

INSTANTIATE(char, float, float);
INSTANTIATE(int, float, float);
INSTANTIATE(unsigned int, float, float);
INSTANTIATE(short, float, float);
INSTANTIATE(unsigned char, float, float);
INSTANTIATE(unsigned short, float, float);
INSTANTIATE(double, float, float);
INSTANTIATE(af::af_cfloat, float, float);
INSTANTIATE(af::af_cdouble, float, float);
INSTANTIATE(long long, float, float);
INSTANTIATE(long long, double, float);
INSTANTIATE(unsigned long long, double, float);
INSTANTIATE(float, float, float);
INSTANTIATE(af_cfloat, af_cfloat, float);
INSTANTIATE(af_cfloat, af_cfloat, af_cfloat);
INSTANTIATE(af_cdouble, af_cdouble, af_cdouble);
INSTANTIATE(double, double, float);
INSTANTIATE(double, double, double);
INSTANTIATE(af_cdouble, af_cdouble, float);
INSTANTIATE(int, int, float);
INSTANTIATE(unsigned int, unsigned int, float);
INSTANTIATE(long long, long long, float);
INSTANTIATE(unsigned long long, unsigned long long, float);
INSTANTIATE(char, char, float);
INSTANTIATE(unsigned char, unsigned char, float);
INSTANTIATE(short, short, float);
INSTANTIATE(unsigned short, unsigned short, float);
INSTANTIATE(half_float::half, half_float::half, float);
INSTANTIATE(half_float::half, half_float::half, double);

INSTANTIATE(af_cdouble, af_cdouble, double);
INSTANTIATE(double, af_cdouble, float);
INSTANTIATE(float, af_cfloat, float);
INSTANTIATE(half_float::half, uint, uint);
INSTANTIATE(float, float, double);
INSTANTIATE(int, float, double);
INSTANTIATE(unsigned int, float, double);
INSTANTIATE(short, float, double);
INSTANTIATE(unsigned short, float, double);
INSTANTIATE(char, float, double);
INSTANTIATE(unsigned char, float, double);
INSTANTIATE(long long, double, double);
INSTANTIATE(unsigned long long, double, double);
INSTANTIATE(af_cfloat, af_cfloat, double);
INSTANTIATE(half_float::half, float, double);

#undef INSTANTIATE

bool noDoubleTests(af::dtype ty) {
    bool isTypeDouble      = (ty == f64) || (ty == c64);
    int dev                = af::getDevice();
    bool isDoubleSupported = af::isDoubleAvailable(dev);

    return ((isTypeDouble && !isDoubleSupported) ? true : false);
}

bool noHalfTests(af::dtype ty) {
    bool isTypeHalf      = (ty == f16);
    int dev              = af::getDevice();
    bool isHalfSupported = af::isHalfAvailable(dev);

    return ((isTypeHalf && !isHalfSupported) ? true : false);
}

af_half abs(af_half in) {
    half_float::half in_;
    // casting to void* to avoid class-memaccess warnings on windows
    memcpy(static_cast<void *>(&in_), &in, sizeof(af_half));
    half_float::half out_ = abs(in_);
    af_half out;
    memcpy(&out, &out_, sizeof(af_half));
    return out;
}

af_half operator-(af_half lhs, af_half rhs) {
    half_float::half lhs_;
    half_float::half rhs_;

    // casting to void* to avoid class-memaccess warnings on windows
    memcpy(static_cast<void *>(&lhs_), &lhs, sizeof(af_half));
    memcpy(static_cast<void *>(&rhs_), &rhs, sizeof(af_half));
    half_float::half out = lhs_ - rhs_;
    af_half o;
    memcpy(&o, &out, sizeof(af_half));
    return o;
}

const af::cfloat &operator+(const af::cfloat &val) { return val; }

const af::cdouble &operator+(const af::cdouble &val) { return val; }

const af_half &operator+(const af_half &val) { return val; }

// Calculate a multi-dimensional coordinates' linearized index
dim_t ravelIdx(af::dim4 coords, af::dim4 strides) {
    return std::inner_product(coords.get(), coords.get() + 4, strides.get(),
                              0LL);
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
    af::dim4 st   = af::getStrides(arr);
    return unravelIdx(idx, dims, st);
}

af::dim4 calcStrides(const af::dim4 &parentDim) {
    af::dim4 out(1, 1, 1, 1);
    dim_t *out_dims          = out.get();
    const dim_t *parent_dims = parentDim.get();

    for (dim_t i = 1; i < 4; i++) {
        out_dims[i] = out_dims[i - 1] * parent_dims[i - 1];
    }

    return out;
}

std::string minimalDim4(af::dim4 coords, af::dim4 dims) {
    std::ostringstream os;
    os << "(" << coords[0];
    if (dims[1] > 1 || dims[2] > 1 || dims[3] > 1) { os << ", " << coords[1]; }
    if (dims[2] > 1 || dims[3] > 1) { os << ", " << coords[2]; }
    if (dims[3] > 1) { os << ", " << coords[3]; }
    os << ")";

    return os.str();
}

// Generates a random array. testWriteToOutputArray expects that it will receive
// the same af_array that this generates after the af_* function is called
void genRegularArray(TestOutputArrayInfo *metadata, const unsigned ndims,
                     const dim_t *const dims, const af_dtype ty) {
    metadata->init(ndims, dims, ty);
}

void genRegularArray(TestOutputArrayInfo *metadata, double val,
                     const unsigned ndims, const dim_t *const dims,
                     const af_dtype ty) {
    metadata->init(val, ndims, dims, ty);
}

// Generates a large, random array, and extracts a subarray for the af_*
// function to use. testWriteToOutputArray expects that the large array that it
// receives is equal to the same large array with the gold array injected on the
// same subarray location
void genSubArray(TestOutputArrayInfo *metadata, const unsigned ndims,
                 const dim_t *const dims, const af_dtype ty) {
    const dim_t pad_size = 2;

    // The large array is padded on both sides of each dimension
    // Padding is only applied if the dimension is used, i.e. if dims[i] > 1
    dim_t full_arr_dims[4] = {dims[0], dims[1], dims[2], dims[3]};
    for (uint i = 0; i < ndims; ++i) {
        full_arr_dims[i] = dims[i] + 2 * pad_size;
    }

    // Calculate index of sub-array. These will be used also by
    // testWriteToOutputArray so that the gold sub array will be placed in the
    // same location. Currently, this location is the center of the large array
    af_seq subarr_idxs[4] = {af_span, af_span, af_span, af_span};
    for (uint i = 0; i < ndims; ++i) {
        af_seq idx     = {pad_size, pad_size + dims[i] - 1.0, 1.0};
        subarr_idxs[i] = idx;
    }

    metadata->init(ndims, full_arr_dims, ty, &subarr_idxs[0]);
}

void genSubArray(TestOutputArrayInfo *metadata, double val,
                 const unsigned ndims, const dim_t *const dims,
                 const af_dtype ty) {
    const dim_t pad_size = 2;

    // The large array is padded on both sides of each dimension
    // Padding is only applied if the dimension is used, i.e. if dims[i] > 1
    dim_t full_arr_dims[4] = {dims[0], dims[1], dims[2], dims[3]};
    for (uint i = 0; i < ndims; ++i) {
        full_arr_dims[i] = dims[i] + 2 * pad_size;
    }

    // Calculate index of sub-array. These will be used also by
    // testWriteToOutputArray so that the gold sub array will be placed in the
    // same location. Currently, this location is the center of the large array
    af_seq subarr_idxs[4] = {af_span, af_span, af_span, af_span};
    for (uint i = 0; i < ndims; ++i) {
        af_seq idx     = {pad_size, pad_size + dims[i] - 1.0, 1.0};
        subarr_idxs[i] = idx;
    }

    metadata->init(val, ndims, full_arr_dims, ty, &subarr_idxs[0]);
}

// Generates a reordered array. testWriteToOutputArray expects that this array
// will still have the correct output values from the af_* function, even though
// the array was initially reordered.
void genReorderedArray(TestOutputArrayInfo *metadata, const unsigned ndims,
                       const dim_t *const dims, const af_dtype ty) {
    // The rest of this function assumes that dims has 4 elements. Just in case
    // dims has < 4 elements, use another dims array that is filled with 1s
    dim_t all_dims[4] = {1, 1, 1, 1};
    for (uint i = 0; i < ndims; ++i) { all_dims[i] = dims[i]; }

    // This reorder combination will not move data around, but will simply
    // call modDims and modStrides (see src/api/c/reorder.cpp).
    // The output will be checked if it is still correct even with the
    // modified dims and strides "hack" with no data movement
    uint reorder_idxs[4] = {0, 2, 1, 3};

    // Shape the output array such that the reordered output array will have
    // the correct dimensions that the test asks for (i.e. must match dims arg)
    dim_t init_dims[4] = {all_dims[0], all_dims[1], all_dims[2], all_dims[3]};
    for (uint i = 0; i < 4; ++i) { init_dims[i] = all_dims[reorder_idxs[i]]; }
    metadata->init(4, init_dims, ty);

    af_array reordered = 0;
    ASSERT_SUCCESS(af_reorder(&reordered, metadata->getOutput(),
                              reorder_idxs[0], reorder_idxs[1], reorder_idxs[2],
                              reorder_idxs[3]));
    metadata->setOutput(reordered);
}

void genReorderedArray(TestOutputArrayInfo *metadata, double val,
                       const unsigned ndims, const dim_t *const dims,
                       const af_dtype ty) {
    // The rest of this function assumes that dims has 4 elements. Just in case
    // dims has < 4 elements, use another dims array that is filled with 1s
    dim_t all_dims[4] = {1, 1, 1, 1};
    for (uint i = 0; i < ndims; ++i) { all_dims[i] = dims[i]; }

    // This reorder combination will not move data around, but will simply
    // call modDims and modStrides (see src/api/c/reorder.cpp).
    // The output will be checked if it is still correct even with the
    // modified dims and strides "hack" with no data movement
    uint reorder_idxs[4] = {0, 2, 1, 3};

    // Shape the output array such that the reordered output array will have
    // the correct dimensions that the test asks for (i.e. must match dims arg)
    dim_t init_dims[4] = {all_dims[0], all_dims[1], all_dims[2], all_dims[3]};
    for (uint i = 0; i < 4; ++i) { init_dims[i] = all_dims[reorder_idxs[i]]; }
    metadata->init(val, 4, init_dims, ty);

    af_array reordered = 0;
    ASSERT_SUCCESS(af_reorder(&reordered, metadata->getOutput(),
                              reorder_idxs[0], reorder_idxs[1], reorder_idxs[2],
                              reorder_idxs[3]));
    metadata->setOutput(reordered);
}
// Partner function of testWriteToOutputArray. This generates the "special"
// array that testWriteToOutputArray will use to check if the af_* function
// correctly uses an existing array as its output
void genTestOutputArray(af_array *out_ptr, const unsigned ndims,
                        const dim_t *const dims, const af_dtype ty,
                        TestOutputArrayInfo *metadata) {
    switch (metadata->getOutputArrayType()) {
        case FULL_ARRAY: genRegularArray(metadata, ndims, dims, ty); break;
        case SUB_ARRAY: genSubArray(metadata, ndims, dims, ty); break;
        case REORDERED_ARRAY:
            genReorderedArray(metadata, ndims, dims, ty);
            break;
        default: break;
    }
    *out_ptr = metadata->getOutput();
}

void genTestOutputArray(af_array *out_ptr, double val, const unsigned ndims,
                        const dim_t *const dims, const af_dtype ty,
                        TestOutputArrayInfo *metadata) {
    switch (metadata->getOutputArrayType()) {
        case FULL_ARRAY: genRegularArray(metadata, val, ndims, dims, ty); break;
        case SUB_ARRAY: genSubArray(metadata, val, ndims, dims, ty); break;
        case REORDERED_ARRAY:
            genReorderedArray(metadata, val, ndims, dims, ty);
            break;
        default: break;
    }
    *out_ptr = metadata->getOutput();
}

// Partner function of genTestOutputArray. This uses the same "special"
// array that genTestOutputArray generates, and checks whether the
// af_* function wrote to that array correctly
::testing::AssertionResult testWriteToOutputArray(
    std::string gold_name, std::string result_name, const af_array gold,
    const af_array out, TestOutputArrayInfo *metadata) {
    // In the case of NULL_ARRAY, the output array starts out as null.
    // After the af_* function is called, it shouldn't be null anymore
    if (metadata->getOutputArrayType() == NULL_ARRAY) {
        if (out == 0) {
            return ::testing::AssertionFailure()
                   << "Output af_array " << result_name << " is null";
        }
        metadata->setOutput(out);
    }
    // For every other case, must check if the af_array generated by
    // genTestOutputArray was used by the af_* function as its output array
    else {
        if (metadata->getOutput() != out) {
            return ::testing::AssertionFailure()
                   << "af_array POINTER MISMATCH:\n"
                   << "  Actual: " << out << "\n"
                   << "Expected: " << metadata->getOutput();
        }
    }

    if (metadata->getOutputArrayType() == SUB_ARRAY) {
        // There are two full arrays. One will be injected with the gold
        // subarray, the other should have already been injected with the af_*
        // function's output. Then we compare the two full arrays
        af_array gold_full_array = metadata->getFullOutputCopy();
        af_assign_seq(&gold_full_array, gold_full_array,
                      metadata->getSubArrayNumDims(),
                      metadata->getSubArrayIdxs(), gold);

        return assertArrayEq(gold_name, result_name,
                             metadata->getFullOutputCopy(),
                             metadata->getFullOutput());
    } else {
        return assertArrayEq(gold_name, result_name, gold, out);
    }
}

// Called by ASSERT_SPECIAL_ARRAYS_EQ
::testing::AssertionResult assertArrayEq(std::string aName, std::string bName,
                                         std::string metadataName,
                                         const af_array a, const af_array b,
                                         TestOutputArrayInfo *metadata) {
    UNUSED(metadataName);
    return testWriteToOutputArray(aName, bName, a, b, metadata);
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

// Called by ASSERT_ARRAYS_NEAR
::testing::AssertionResult assertArrayNear(std::string aName, std::string bName,
                                           std::string maxAbsDiffName,
                                           const af::array &a,
                                           const af::array &b,
                                           float maxAbsDiff) {
    UNUSED(maxAbsDiffName);
    return assertArrayEq(aName, bName, a, b, maxAbsDiff);
}

// Called by ASSERT_IMAGES_NEAR
::testing::AssertionResult assertImageNear(std::string aName, std::string bName,
                                           std::string maxAbsDiffName,
                                           const af_array &a, const af_array &b,
                                           float maxAbsDiff) {
    UNUSED(maxAbsDiffName);
    af_array aa = 0, bb = 0;
    af_retain_array(&aa, a);
    af_retain_array(&bb, b);
    af::array aaa(aa);
    af::array bbb(bb);
    return assertImageEq(aName, bName, aaa, bbb, maxAbsDiff);
}

// Called by ASSERT_IMAGES_NEAR
::testing::AssertionResult assertImageNear(std::string aName, std::string bName,
                                           std::string maxAbsDiffName,
                                           const af::array &a,
                                           const af::array &b,
                                           float maxAbsDiff) {
    UNUSED(maxAbsDiffName);
    return assertImageEq(aName, bName, a, b, maxAbsDiff);
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

void cleanSlate() {
    const size_t step_bytes = 1024;

    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    af::deviceGC();

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(0u, alloc_buffers);
    ASSERT_EQ(0u, lock_buffers);
    ASSERT_EQ(0u, alloc_bytes);
    ASSERT_EQ(0u, lock_bytes);

    af::setMemStepSize(step_bytes);

    ASSERT_EQ(af::getMemStepSize(), step_bytes);
}

bool noImageIOTests() {
    bool ret = !af::isImageIOAvailable();
    if (ret) printf("Image IO Not Configured. Test will exit\n");
    return ret;
}

bool noLAPACKTests() {
    bool ret = !af::isLAPACKAvailable();
    if (ret) printf("LAPACK Not Configured. Test will exit\n");
    return ret;
}

template<typename inType, typename outType>
void readTestsFromFile(const std::string &FileName,
                       std::vector<af::dim4> &inputDims,
                       std::vector<std::vector<inType>> &testInputs,
                       std::vector<std::vector<outType>> &testOutputs) {
    using std::vector;

    std::ifstream testFile(FileName.c_str());
    if (testFile.good()) {
        unsigned inputCount;
        testFile >> inputCount;
        for (unsigned i = 0; i < inputCount; i++) {
            af::dim4 temp(1);
            testFile >> temp;
            inputDims.push_back(temp);
        }

        unsigned testCount;
        testFile >> testCount;
        testOutputs.resize(testCount);

        vector<unsigned> testSizes(testCount);
        for (unsigned i = 0; i < testCount; i++) { testFile >> testSizes[i]; }

        testInputs.resize(inputCount, vector<inType>(0));
        for (unsigned k = 0; k < inputCount; k++) {
            dim_t nElems = inputDims[k].elements();
            testInputs[k].resize(nElems);
            inType tmp;
            for (unsigned i = 0; i < nElems; i++) {
                testFile >> tmp;
                testInputs[k][i] = tmp;
            }
        }

        testOutputs.resize(testCount, vector<outType>(0));
        for (unsigned i = 0; i < testCount; i++) {
            testOutputs[i].resize(testSizes[i]);
            outType tmp;
            for (unsigned j = 0; j < testSizes[i]; j++) {
                testFile >> tmp;
                testOutputs[i][j] = tmp;
            }
        }
    } else {
        FAIL() << "TEST FILE NOT FOUND";
    }
}

#define INSTANTIATE(Ti, To)                                            \
    template void readTestsFromFile<Ti, To>(                           \
        const std::string &FileName, std::vector<af::dim4> &inputDims, \
        std::vector<std::vector<Ti>> &testInputs,                      \
        std::vector<std::vector<To>> &testOutputs)

INSTANTIATE(float, float);
INSTANTIATE(float, af_cfloat);
INSTANTIATE(af_cfloat, af_cfloat);
INSTANTIATE(double, double);
INSTANTIATE(double, af_cdouble);
INSTANTIATE(af_cdouble, af_cdouble);
INSTANTIATE(int, float);

#undef INSTANTIATE

template<typename outType>
void readImageTests(const std::string &pFileName,
                    std::vector<af::dim4> &pInputDims,
                    std::vector<std::string> &pTestInputs,
                    std::vector<std::vector<outType>> &pTestOutputs) {
    using std::vector;

    std::ifstream testFile(pFileName.c_str());
    if (testFile.good()) {
        unsigned inputCount;
        testFile >> inputCount;
        for (unsigned i = 0; i < inputCount; i++) {
            af::dim4 temp(1);
            testFile >> temp;
            pInputDims.push_back(temp);
        }

        unsigned testCount;
        testFile >> testCount;
        pTestOutputs.resize(testCount);

        vector<unsigned> testSizes(testCount);
        for (unsigned i = 0; i < testCount; i++) { testFile >> testSizes[i]; }

        pTestInputs.resize(inputCount, "");
        for (unsigned k = 0; k < inputCount; k++) {
            pTestInputs[k] = readNextNonEmptyLine(testFile);
        }

        pTestOutputs.resize(testCount, vector<outType>(0));
        for (unsigned i = 0; i < testCount; i++) {
            pTestOutputs[i].resize(testSizes[i]);
            outType tmp;
            for (unsigned j = 0; j < testSizes[i]; j++) {
                testFile >> tmp;
                pTestOutputs[i][j] = tmp;
            }
        }
    } else {
        FAIL() << "TEST FILE NOT FOUND";
    }
}

#define INSTANTIATE(To)                                                  \
    template void readImageTests<To>(                                    \
        const std::string &pFileName, std::vector<af::dim4> &pInputDims, \
        std::vector<std::string> &pTestInputs,                           \
        std::vector<std::vector<To>> &pTestOutputs)

INSTANTIATE(float);
#undef INSTANTIATE

void readImageTests(const std::string &pFileName,
                    std::vector<af::dim4> &pInputDims,
                    std::vector<std::string> &pTestInputs,
                    std::vector<dim_t> &pTestOutSizes,
                    std::vector<std::string> &pTestOutputs) {
    using std::vector;

    std::ifstream testFile(pFileName.c_str());
    if (testFile.good()) {
        unsigned inputCount;
        testFile >> inputCount;
        for (unsigned i = 0; i < inputCount; i++) {
            af::dim4 temp(1);
            testFile >> temp;
            pInputDims.push_back(temp);
        }

        unsigned testCount;
        testFile >> testCount;
        pTestOutputs.resize(testCount);

        pTestOutSizes.resize(testCount);
        for (unsigned i = 0; i < testCount; i++) {
            testFile >> pTestOutSizes[i];
        }

        pTestInputs.resize(inputCount, "");
        for (unsigned k = 0; k < inputCount; k++) {
            pTestInputs[k] = readNextNonEmptyLine(testFile);
        }

        pTestOutputs.resize(testCount, "");
        for (unsigned i = 0; i < testCount; i++) {
            pTestOutputs[i] = readNextNonEmptyLine(testFile);
        }
    } else {
        FAIL() << "TEST FILE NOT FOUND";
    }
}

template<typename descType>
void readImageFeaturesDescriptors(
    const std::string &pFileName, std::vector<af::dim4> &pInputDims,
    std::vector<std::string> &pTestInputs,
    std::vector<std::vector<float>> &pTestFeats,
    std::vector<std::vector<descType>> &pTestDescs) {
    using std::vector;

    std::ifstream testFile(pFileName.c_str());
    if (testFile.good()) {
        unsigned inputCount;
        testFile >> inputCount;
        for (unsigned i = 0; i < inputCount; i++) {
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
        for (unsigned k = 0; k < inputCount; k++) {
            pTestInputs[k] = readNextNonEmptyLine(testFile);
        }

        pTestFeats.resize(attrCount, vector<float>(0));
        for (unsigned i = 0; i < attrCount; i++) {
            pTestFeats[i].resize(featCount);
            float tmp;
            for (unsigned j = 0; j < featCount; j++) {
                testFile >> tmp;
                pTestFeats[i][j] = tmp;
            }
        }

        pTestDescs.resize(featCount, vector<descType>(0));
        for (unsigned i = 0; i < featCount; i++) {
            pTestDescs[i].resize(descLen);
            descType tmp;
            for (unsigned j = 0; j < descLen; j++) {
                testFile >> tmp;
                pTestDescs[i][j] = tmp;
            }
        }
    } else {
        FAIL() << "TEST FILE NOT FOUND";
    }
}

#define INSTANTIATE(TYPE)                                                \
    template void readImageFeaturesDescriptors<TYPE>(                    \
        const std::string &pFileName, std::vector<af::dim4> &pInputDims, \
        std::vector<std::string> &pTestInputs,                           \
        std::vector<std::vector<float>> &pTestFeats,                     \
        std::vector<std::vector<TYPE>> &pTestDescs)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(unsigned int);
#undef INSTANTIATE

template<typename T>
double computeArraysRMSD(dim_t data_size, T *gold, T *data) {
    double accum  = 0.0;
    double maxion = -FLT_MAX;  //(double)std::numeric_limits<T>::lowest();
    double minion = FLT_MAX;   //(double)std::numeric_limits<T>::max();

    for (dim_t i = 0; i < data_size; i++) {
        double dTemp = (double)data[i];
        double gTemp = (double)gold[i];
        double diff  = gTemp - dTemp;
        if (diff > 1.e-4) {
            // printf("%d: diff: %f %f %f\n", i, diff, data[i], gold[i]);
        }
        double err =
            (std::isfinite(diff) && (std::abs(diff) > 1.0e-4)) ? diff : 0.0f;
        accum += std::pow(err, 2.0);
        maxion = std::max(maxion, dTemp);
        minion = std::min(minion, dTemp);
    }
    accum /= data_size;
    double NRMSD = std::sqrt(accum) / (maxion - minion);

    return NRMSD;
}

template<>
double computeArraysRMSD<unsigned char>(dim_t data_size, unsigned char *gold,
                                        unsigned char *data) {
    double accum = 0.0;
    int maxion   = 0;    //(double)std::numeric_limits<T>::lowest();
    int minion   = 255;  //(double)std::numeric_limits<T>::max();

    for (dim_t i = 0; i < data_size; i++) {
        int dTemp  = data[i];
        int gTemp  = gold[i];
        int diff   = abs(gTemp - dTemp);
        double err = (diff > 1) ? diff : 0.0f;
        accum += std::pow(err, 2.0);
        maxion = std::max(maxion, dTemp);
        minion = std::min(minion, dTemp);
    }
    accum /= data_size;
    double NRMSD = std::sqrt(accum) / (maxion - minion);

    return NRMSD;
}

template<typename T>
bool compareArraysRMSD(dim_t data_size, T *gold, T *data, double tolerance) {
    double accum  = 0.0;
    double maxion = -FLT_MAX;  //(double)std::numeric_limits<T>::lowest();
    double minion = FLT_MAX;   //(double)std::numeric_limits<T>::max();

    for (dim_t i = 0; i < data_size; i++) {
        double dTemp = (double)data[i];
        double gTemp = (double)gold[i];
        double diff  = gTemp - dTemp;
        double err =
            (std::isfinite(diff) && (std::abs(diff) > 1.0e-4)) ? diff : 0.0f;
        accum += std::pow(err, 2.0);
        maxion = std::max(maxion, dTemp);
        minion = std::min(minion, dTemp);
    }
    accum /= data_size;
    double NRMSD = std::sqrt(accum) / (maxion - minion);

    if (std::isnan(NRMSD) || NRMSD > tolerance) {
#ifndef NDEBUG
        printf("Comparison failed, NRMSD value: %lf\n", NRMSD);
#endif
        return false;
    }

    return true;
}

#define INSTANTIATE(TYPE)                                                 \
    template double computeArraysRMSD<TYPE>(dim_t data_size, TYPE * gold, \
                                            TYPE * data);                 \
    template bool compareArraysRMSD<TYPE>(dim_t data_size, TYPE * gold,   \
                                          TYPE * data, double tolerance)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(char);
#undef INSTANTIATE

TestOutputArrayInfo::TestOutputArrayInfo()
    : out_arr(0)
    , out_arr_cpy(0)
    , out_subarr(0)
    , out_subarr_ndims(0)
    , out_arr_type(NULL_ARRAY) {
    for (uint i = 0; i < 4; ++i) { out_subarr_idxs[i] = af_span; }
}

TestOutputArrayInfo::TestOutputArrayInfo(TestOutputArrayType arr_type)
    : out_arr(0)
    , out_arr_cpy(0)
    , out_subarr(0)
    , out_subarr_ndims(0)
    , out_arr_type(arr_type) {
    for (uint i = 0; i < 4; ++i) { out_subarr_idxs[i] = af_span; }
}

TestOutputArrayInfo::~TestOutputArrayInfo() {
    if (out_subarr) af_release_array(out_subarr);
    if (out_arr_cpy) af_release_array(out_arr_cpy);
    if (out_arr) af_release_array(out_arr);
}

void TestOutputArrayInfo::init(const unsigned ndims, const dim_t *const dims,
                               const af_dtype ty) {
    ASSERT_SUCCESS(af_randu(&out_arr, ndims, dims, ty));
}

void TestOutputArrayInfo::init(const unsigned ndims, const dim_t *const dims,
                               const af_dtype ty,
                               const af_seq *const subarr_idxs) {
    init(ndims, dims, ty);

    ASSERT_SUCCESS(af_copy_array(&out_arr_cpy, out_arr));
    for (uint i = 0; i < ndims; ++i) { out_subarr_idxs[i] = subarr_idxs[i]; }
    out_subarr_ndims = ndims;

    ASSERT_SUCCESS(af_index(&out_subarr, out_arr, ndims, subarr_idxs));
}

void TestOutputArrayInfo::init(double val, const unsigned ndims,
                               const dim_t *const dims, const af_dtype ty) {
    switch (ty) {
        case c32:
        case c64:
            af_constant_complex(&out_arr, val, 0.0, ndims, dims, ty);
            break;
        case s64:
            af_constant_long(&out_arr, static_cast<intl>(val), ndims, dims);
            break;
        case u64:
            af_constant_ulong(&out_arr, static_cast<uintl>(val), ndims, dims);
            break;
        default: af_constant(&out_arr, val, ndims, dims, ty); break;
    }
}

void TestOutputArrayInfo::init(double val, const unsigned ndims,
                               const dim_t *const dims, const af_dtype ty,
                               const af_seq *const subarr_idxs) {
    init(val, ndims, dims, ty);

    ASSERT_SUCCESS(af_copy_array(&out_arr_cpy, out_arr));
    for (uint i = 0; i < ndims; ++i) { out_subarr_idxs[i] = subarr_idxs[i]; }
    out_subarr_ndims = ndims;

    ASSERT_SUCCESS(af_index(&out_subarr, out_arr, ndims, subarr_idxs));
}

af_array TestOutputArrayInfo::getOutput() {
    if (out_arr_type == SUB_ARRAY) {
        return out_subarr;
    } else {
        return out_arr;
    }
}

void TestOutputArrayInfo::setOutput(af_array array) {
    if (out_arr != 0) { ASSERT_SUCCESS(af_release_array(out_arr)); }
    out_arr = array;
}

af_array TestOutputArrayInfo::getFullOutput() { return out_arr; }
af_array TestOutputArrayInfo::getFullOutputCopy() { return out_arr_cpy; }
af_seq *TestOutputArrayInfo::getSubArrayIdxs() { return &out_subarr_idxs[0]; }
dim_t TestOutputArrayInfo::getSubArrayNumDims() { return out_subarr_ndims; }
TestOutputArrayType TestOutputArrayInfo::getOutputArrayType() {
    return out_arr_type;
}

#if defined(USE_MTX)
::testing::AssertionResult mtxReadSparseMatrix(af::array &out,
                                               const char *fileName) {
    FILE *fileHandle;

    if ((fileHandle = fopen(fileName, "r")) == NULL) {
        return ::testing::AssertionFailure()
               << "Failed to open mtx file: " << fileName << "\n";
    }

    MM_typecode matcode;
    if (mm_read_banner(fileHandle, &matcode)) {
        return ::testing::AssertionFailure()
               << "Could not process Matrix Market banner.\n";
    }

    if (!(mm_is_matrix(matcode) && mm_is_sparse(matcode))) {
        return ::testing::AssertionFailure()
               << "Input mtx doesn't have a sparse matrix.\n";
    }

    if (mm_is_integer(matcode)) {
        return ::testing::AssertionFailure() << "MTX file has integer data. \
                Integer sparse matrices are not supported in ArrayFire yet.\n";
    }

    int M = 0, N = 0, nz = 0;
    if (mm_read_mtx_crd_size(fileHandle, &M, &N, &nz)) {
        return ::testing::AssertionFailure()
               << "Failed to read matrix dimensions.\n";
    }

    if (mm_is_real(matcode)) {
        std::vector<int> I(nz);
        std::vector<int> J(nz);
        std::vector<float> V(nz);

        for (int i = 0; i < nz; ++i) {
            int c, r;
            double v;
            int readCount = fscanf(fileHandle, "%d %d %lg\n", &r, &c, &v);
            if (readCount != 3) {
                fclose(fileHandle);
                return ::testing::AssertionFailure()
                       << "\nEnd of file reached, expected more data, "
                       << "following are some reasons this happens.\n"
                       << "\t - use of template type that doesn't match data "
                          "type\n"
                       << "\t - the mtx file itself doesn't have enough data\n";
            }
            I[i] = r - 1;
            J[i] = c - 1;
            V[i] = (float)v;
        }

        out = af::sparse(M, N, nz, V.data(), I.data(), J.data(), f32,
                         AF_STORAGE_COO);
    } else if (mm_is_complex(matcode)) {
        std::vector<int> I(nz);
        std::vector<int> J(nz);
        std::vector<af::cfloat> V(nz);

        for (int i = 0; i < nz; ++i) {
            int c, r;
            double real, imag;
            int readCount =
                fscanf(fileHandle, "%d %d %lg %lg\n", &r, &c, &real, &imag);
            if (readCount != 4) {
                fclose(fileHandle);
                return ::testing::AssertionFailure()
                       << "\nEnd of file reached, expected more data, "
                       << "following are some reasons this happens.\n"
                       << "\t - use of template type that doesn't match data "
                          "type\n"
                       << "\t - the mtx file itself doesn't have enough data\n";
            }
            I[i] = r - 1;
            J[i] = c - 1;
            V[i] = af::cfloat(float(real), float(imag));
        }

        out = af::sparse(M, N, nz, V.data(), I.data(), J.data(), c32,
                         AF_STORAGE_COO);
    } else {
        return ::testing::AssertionFailure()
               << "Unknown matcode from MTX FILE\n";
    }

    fclose(fileHandle);
    return ::testing::AssertionSuccess();
}
#endif  // USE_MTX

// TODO: perform conversion on device for CUDA and OpenCL
template<typename T>
af_err conv_image(af_array *out, af_array in) {
    af_array outArray;

    dim_t d0, d1, d2, d3;
    af_get_dims(&d0, &d1, &d2, &d3, in);
    af::dim4 idims(d0, d1, d2, d3);

    dim_t nElems = 0;
    af_get_elements(&nElems, in);

    float *in_data = new float[nElems];
    af_get_data_ptr(in_data, in);

    T *out_data = new T[nElems];

    for (int i = 0; i < (int)nElems; i++) out_data[i] = (T)in_data[i];

    af_create_array(&outArray, out_data, idims.ndims(), idims.get(),
                    (af_dtype)af::dtype_traits<T>::af_type);

    std::swap(*out, outArray);

    delete[] in_data;
    delete[] out_data;

    return AF_SUCCESS;
}

#define INSTANTIATE(To) \
    template af_err conv_image<To>(af_array * out, af_array in)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(unsigned char);
INSTANTIATE(half_float::half);
INSTANTIATE(unsigned int);
INSTANTIATE(unsigned short);
INSTANTIATE(int);
INSTANTIATE(char);
INSTANTIATE(short);
INSTANTIATE(af_cdouble);
INSTANTIATE(af_cfloat);
INSTANTIATE(long long);
INSTANTIATE(unsigned long long);
#undef INSTANTIATE

template<typename T>
af::array cpu_randu(const af::dim4 dims) {
    typedef typename af::dtype_traits<T>::base_type BT;

    bool isTypeCplx = is_same_type<T, af::cfloat>::value ||
                      is_same_type<T, af::cdouble>::value;
    bool isTypeFloat = is_same_type<BT, float>::value ||
                       is_same_type<BT, double>::value ||
                       is_same_type<BT, half_float::half>::value;

    size_t elements = (isTypeCplx ? 2 : 1) * dims.elements();

    std::vector<BT> out(elements);
    for (size_t i = 0; i < elements; i++) {
        out[i] = isTypeFloat ? (BT)(rand()) / static_cast<double>(RAND_MAX)
                             : rand() % 100;
    }

    return af::array(dims, (T *)&out[0]);
}

#define INSTANTIATE(To) template af::array cpu_randu<To>(const af::dim4 dims)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(unsigned char);
INSTANTIATE(half_float::half);
INSTANTIATE(unsigned int);
INSTANTIATE(unsigned short);
INSTANTIATE(int);
INSTANTIATE(char);
INSTANTIATE(short);
INSTANTIATE(af_cdouble);
INSTANTIATE(af_cfloat);
INSTANTIATE(long long);
INSTANTIATE(unsigned long long);
#undef INSTANTIATE

template<typename T>
struct sparseCooValue {
    int row = 0;
    int col = 0;
    T value = 0;
    sparseCooValue(int r, int c, T v) : row(r), col(c), value(v) {}
};

template<typename T>
void swap(sparseCooValue<T> &lhs, sparseCooValue<T> &rhs) {
    std::swap(lhs.row, rhs.row);
    std::swap(lhs.col, rhs.col);
    std::swap(lhs.value, rhs.value);
}

template<typename T>
bool operator<(const sparseCooValue<T> &lhs, const sparseCooValue<T> &rhs) {
    if (lhs.row < rhs.row) {
        return true;
    } else if (lhs.row == rhs.row && lhs.col < rhs.col) {
        return true;
    } else {
        return false;
    }
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const sparseCooValue<T> &val) {
    os << "(" << val.row << ", " << val.col << "): " << val.value;
    return os;
}

template<typename T>
bool isZero(const sparseCooValue<T> &val) {
    return real(val.value) == 0.;
}

template<typename T>
vector<sparseCooValue<T>> toCooVector(const af::array &arr) {
    vector<sparseCooValue<T>> out;
    if (arr.issparse()) {
        switch (sparseGetStorage(arr)) {
            case AF_STORAGE_COO: {
                dim_t nnz = sparseGetNNZ(arr);
                vector<int> row(nnz), col(nnz);
                vector<T> values(nnz);
                sparseGetValues(arr).host(values.data());
                sparseGetRowIdx(arr).host(row.data());
                sparseGetColIdx(arr).host(col.data());
                out.reserve(nnz);
                for (int i = 0; i < nnz; i++) {
                    out.emplace_back(row[i], col[i], values[i]);
                }
            } break;
            case AF_STORAGE_CSR: {
                dim_t nnz = sparseGetNNZ(arr);
                vector<int> row(arr.dims(0) + 1), col(nnz);
                vector<T> values(nnz);
                sparseGetValues(arr).host(values.data());
                sparseGetRowIdx(arr).host(row.data());
                sparseGetColIdx(arr).host(col.data());
                out.reserve(nnz);
                for (int i = 0; i < row.size() - 1; i++) {
                    for (int r = row[i]; r < row[i + 1]; r++) {
                        out.emplace_back(i, col[r], values[r]);
                    }
                }
            } break;
            case AF_STORAGE_CSC: {
                dim_t nnz = sparseGetNNZ(arr);
                vector<int> row(nnz), col(arr.dims(1) + 1);
                vector<T> values(nnz);
                sparseGetValues(arr).host(values.data());
                sparseGetRowIdx(arr).host(row.data());
                sparseGetColIdx(arr).host(col.data());
                out.reserve(nnz);
                for (int i = 0; i < col.size() - 1; i++) {
                    for (int c = col[i]; c < col[i + 1]; c++) {
                        out.emplace_back(row[c], i, values[c]);
                    }
                }
            } break;
            default: throw std::logic_error("NOT SUPPORTED");
        }
    } else {
        vector<T> values(arr.elements());
        arr.host(values.data());
        int M = arr.dims(0), N = arr.dims(1);
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < M; i++) {
                if (std::fpclassify(real(values[j * M + i])) == FP_ZERO) {
                    out.emplace_back(i, j, values[j * M + i]);
                }
            }
        }
    }

    // Remove zero elements from result to ensure that only non-zero elements
    // are compared
    out.erase(std::remove_if(out.begin(), out.end(), isZero<T>), out.end());
    std::sort(begin(out), end(out));
    return out;
}

template<typename T>
bool operator==(const sparseCooValue<T> &lhs, sparseCooValue<T> &rhs) {
    return lhs.row == rhs.row && lhs.col == rhs.col &&
           cmp(lhs.value, rhs.value);
}

template<typename T>
std::string printContext(const std::vector<T> &hGold, std::string goldName,
                         const std::vector<T> &hOut, std::string outName,
                         af::dim4 arrDims, af::dim4 arrStrides, dim_t idx) {
    std::ostringstream os;

    af::dim4 coords = unravelIdx(idx, arrDims, arrStrides);
    dim_t ctxWidth  = 5;

    // Coordinates that span dim0
    af::dim4 coordsMinBound = coords;
    coordsMinBound[0]       = 0;
    af::dim4 coordsMaxBound = coords;
    coordsMaxBound[0]       = arrDims[0] - 1;

    // dim0 positions that can be displayed
    dim_t dim0Start = std::max<dim_t>(0LL, coords[0] - ctxWidth);
    dim_t dim0End   = std::min<dim_t>(coords[0] + ctxWidth + 1LL, arrDims[0]);

    // Linearized indices of values in vectors that can be displayed
    dim_t vecStartIdx =
        std::max<dim_t>(ravelIdx(coordsMinBound, arrStrides), idx - ctxWidth);

    // Display as minimal coordinates as needed
    // First value is the range of dim0 positions that will be displayed
    os << "Viewing slice (" << dim0Start << ":" << dim0End - 1;
    if (arrDims[1] > 1 || arrDims[2] > 1 || arrDims[3] > 1)
        os << ", " << coords[1];
    if (arrDims[2] > 1 || arrDims[3] > 1) os << ", " << coords[2];
    if (arrDims[3] > 1) os << ", " << coords[3];
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
        ctxDim0[i]     = tmpOs.str();
        size_t dim0Len = tmpOs.str().length();
        tmpOs.str(std::string());

        dim_t valIdx = vecStartIdx + i;

        if (valIdx == idx) {
            tmpOs << "[" << +hOut[valIdx] << "]";
        } else {
            tmpOs << +hOut[valIdx];
        }
        ctxOutVals[i] = tmpOs.str();
        size_t outLen = tmpOs.str().length();
        tmpOs.str(std::string());

        if (valIdx == idx) {
            tmpOs << "[" << +hGold[valIdx] << "]";
        } else {
            tmpOs << +hGold[valIdx];
        }
        ctxGoldVals[i] = tmpOs.str();
        size_t goldLen = tmpOs.str().length();
        tmpOs.str(std::string());

        int maxWidth      = std::max<int>(dim0Len, outLen);
        maxWidth          = std::max<int>(maxWidth, goldLen);
        valFieldWidths[i] = maxWidth;
    }

    size_t varNameWidth = std::max<size_t>(goldName.length(), outName.length());

    // Display dim0 positions, output values, and reference values
    os << std::right << std::setw(varNameWidth) << ""
       << "   ";
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

template<typename T>
std::string printContext(const std::vector<sparseCooValue<T>> &hGold,
                         std::string goldName,
                         const std::vector<sparseCooValue<T>> &hOut,
                         std::string outName, af::dim4 arrDims,
                         af::dim4 arrStrides, dim_t idx) {
    std::ostringstream os;

    af::dim4 coords = unravelIdx(idx, arrDims, arrStrides);
    dim_t ctxWidth  = 5;

    // Coordinates that span dim0
    af::dim4 coordsMinBound = coords;
    coordsMinBound[0]       = 0;
    af::dim4 coordsMaxBound = coords;
    coordsMaxBound[0]       = arrDims[0] - 1;

    // dim0 positions that can be displayed
    dim_t dim0Start = std::max<dim_t>(0LL, idx - ctxWidth);
    dim_t dim0End   = std::min<dim_t>(idx + ctxWidth + 1LL, hGold.size());

    int setwval = 9;
    // Linearized indices of values in vectors that can be displayed
    dim_t vecStartIdx =
        std::max<dim_t>(ravelIdx(coordsMinBound, arrStrides), idx - ctxWidth);
    os << "Idx: ";
    for (int elem = dim0Start; elem < dim0End; elem++) {
        if (elem == idx) {
            os << std::setw(setwval - 2) << "[" << elem << "]";
        } else {
            os << std::setw(setwval) << elem;
        }
    }
    os << "\nRow: ";
    for (int elem = dim0Start; elem < dim0End; elem++) {
        if (elem == idx) {
            os << std::setw(setwval - 2) << "[" << hGold[elem].row << "]";
        } else {
            os << std::setw(setwval) << hGold[elem].row;
        }
    }
    os << "\n     ";
    for (int elem = dim0Start; elem < dim0End; elem++) {
        if (elem == idx) {
            os << std::setw(setwval - 2) << "[" << hOut[elem].row << "]";
        } else {
            os << std::setw(setwval) << hOut[elem].row;
        }
    }
    os << "\nCol: ";
    for (int elem = dim0Start; elem < dim0End; elem++) {
        if (elem == idx) {
            os << std::setw(setwval - 2) << "[" << hGold[elem].col << "]";
        } else {
            os << std::setw(setwval) << hGold[elem].col;
        }
    }
    os << "\n     ";
    for (int elem = dim0Start; elem < dim0End; elem++) {
        if (elem == idx) {
            os << std::setw(setwval - 2) << "[" << hOut[elem].col << "]";
        } else {
            os << std::setw(setwval) << hOut[elem].col;
        }
    }

    os << "\nValue: ";
    for (int elem = dim0Start; elem < dim0End; elem++) {
        if (elem == idx) {
            os << std::setw(setwval - 2) << "[" << hGold[elem].value << "]";
        } else {
            os << std::setw(setwval) << hGold[elem].value;
        }
    }
    os << "\n       ";
    for (int elem = dim0Start; elem < dim0End; elem++) {
        if (elem == idx) {
            os << std::setw(setwval - 2) << "[" << hOut[elem].value << "]";
        } else {
            os << std::setw(setwval) << hOut[elem].value;
        }
    }

    return os.str();
}

template<typename T>
::testing::AssertionResult elemWiseEq(std::string aName, std::string bName,
                                      const std::vector<T> &a, af::dim4 aDims,
                                      const std::vector<T> &b, af::dim4 bDims,
                                      float maxAbsDiff, IntegerTag) {
    UNUSED(maxAbsDiff);
    typedef typename std::vector<T>::const_iterator iter;

    std::pair<iter, iter> mismatches =
        std::mismatch(a.begin(), a.end(), b.begin());
    iter bItr = mismatches.second;

    if (bItr == b.end()) {
        return ::testing::AssertionSuccess();
    } else {
        dim_t idx         = std::distance(b.begin(), bItr);
        af::dim4 aStrides = calcStrides(aDims);
        af::dim4 bStrides = calcStrides(bDims);
        af::dim4 coords   = unravelIdx(idx, bDims, bStrides);

        return ::testing::AssertionFailure()
               << "VALUE DIFFERS at " << minimalDim4(coords, aDims) << ":\n"
               << printContext(a, aName, b, bName, aDims, aStrides, idx);
    }
}

struct absMatch {
    float diff_;
    absMatch(float diff) : diff_(diff) {}

    template<typename T>
    bool operator()(const T &lhs, const T &rhs) const {
        if (diff_ > 0) {
            using half_float::abs;
            using std::abs;
            return abs(rhs - lhs) <= diff_;
        } else {
            return boost::math::epsilon_difference(lhs, rhs) < T(1.f);
        }
    }
};

template<>
bool absMatch::operator()<af::af_cfloat>(const af::af_cfloat &lhs,
                                         const af::af_cfloat &rhs) const {
    return af::abs(rhs - lhs) <= diff_;
}

template<>
bool absMatch::operator()<af::af_cdouble>(const af::af_cdouble &lhs,
                                          const af::af_cdouble &rhs) const {
    return af::abs(rhs - lhs) <= diff_;
}

template<>
bool absMatch::operator()<std::complex<float>>(
    const std::complex<float> &lhs, const std::complex<float> &rhs) const {
    return std::abs(rhs - lhs) <= diff_;
}

template<>
bool absMatch::operator()<std::complex<double>>(
    const std::complex<double> &lhs, const std::complex<double> &rhs) const {
    return std::abs(rhs - lhs) <= diff_;
}

template<typename T>
::testing::AssertionResult elemWiseEq(std::string aName, std::string bName,
                                      const std::vector<T> &a, af::dim4 aDims,
                                      const std::vector<T> &b, af::dim4 bDims,
                                      float maxAbsDiff, FloatTag) {
    typedef typename std::vector<T>::const_iterator iter;
    // TODO(mark): Modify equality for float
    std::pair<iter, iter> mismatches =
        std::mismatch(a.begin(), a.end(), b.begin(), absMatch(maxAbsDiff));

    iter aItr = mismatches.first;
    iter bItr = mismatches.second;

    if (aItr == a.end()) {
        return ::testing::AssertionSuccess();
    } else {
        dim_t idx       = std::distance(b.begin(), bItr);
        af::dim4 coords = unravelIdx(idx, bDims, calcStrides(bDims));

        af::dim4 aStrides = calcStrides(aDims);

        ::testing::AssertionResult result =
            ::testing::AssertionFailure()
            << "VALUE DIFFERS at " << minimalDim4(coords, aDims) << ":\n"
            << printContext(a, aName, b, bName, aDims, aStrides, idx);

        if (maxAbsDiff > 0) {
            using af::abs;
            using std::abs;
            double absdiff = abs(*aItr - *bItr);
            result << "\n  Actual diff: " << absdiff << "\n"
                   << "Expected diff: " << maxAbsDiff;
        }

        return result;
    }
}

template<typename T>
::testing::AssertionResult elemWiseEq(std::string aName, std::string bName,
                                      const std::vector<sparseCooValue<T>> &a,
                                      af::dim4 aDims,
                                      const std::vector<sparseCooValue<T>> &b,
                                      af::dim4 bDims, float maxAbsDiff,
                                      IntegerTag) {
    return ::testing::AssertionFailure() << "Unsupported sparse type\n";
}
template<typename T>
::testing::AssertionResult elemWiseEq(std::string aName, std::string bName,
                                      const std::vector<sparseCooValue<T>> &a,
                                      af::dim4 aDims,
                                      const std::vector<sparseCooValue<T>> &b,
                                      af::dim4 bDims, float maxAbsDiff,
                                      FloatTag) {
    typedef typename std::vector<sparseCooValue<T>>::const_iterator iter;
    // TODO(mark): Modify equality for float

    const absMatch diff(maxAbsDiff);
    std::pair<iter, iter> mismatches = std::mismatch(
        a.begin(), a.end(), b.begin(),
        [&diff](const sparseCooValue<T> &lhs, const sparseCooValue<T> &rhs) {
            return lhs.row == rhs.row && lhs.col == rhs.col &&
                   diff(lhs.value, rhs.value);
        });

    iter aItr = mismatches.first;
    iter bItr = mismatches.second;

    if (aItr == a.end()) {
        return ::testing::AssertionSuccess();
    } else {
        dim_t idx       = std::distance(b.begin(), bItr);
        af::dim4 coords = unravelIdx(idx, bDims, calcStrides(bDims));

        af::dim4 aStrides = calcStrides(aDims);

        ::testing::AssertionResult result =
            ::testing::AssertionFailure()
            << "VALUE DIFFERS at " << idx << ":\n"
            << printContext(a, aName, b, bName, aDims, aStrides, idx);

        return result;
    }
}

template<typename T>
::testing::AssertionResult elemWiseEq(std::string aName, std::string bName,
                                      const af::array &a, const af::array &b,
                                      float maxAbsDiff) {
    typedef typename cond_type<
        IsFloatingPoint<typename af::dtype_traits<T>::base_type>::value,
        FloatTag, IntegerTag>::type TagType;
    TagType tag;

    if (a.issparse() || b.issparse()) {
        vector<sparseCooValue<T>> hA = toCooVector<T>(a);
        vector<sparseCooValue<T>> hB = toCooVector<T>(b);

        return elemWiseEq<T>(aName, bName, hA, a.dims(), hB, b.dims(),
                             maxAbsDiff, tag);
    } else {
        std::vector<T> hA(static_cast<size_t>(a.elements()));
        a.host(hA.data());

        std::vector<T> hB(static_cast<size_t>(b.elements()));
        b.host(hB.data());
        return elemWiseEq<T>(aName, bName, hA, a.dims(), hB, b.dims(),
                             maxAbsDiff, tag);
    }
}

template<typename T>
::testing::AssertionResult assertArrayEq(std::string aName,
                                         std::string aDimsName,
                                         std::string bName,
                                         const std::vector<T> &hA,
                                         af::dim4 aDims, const af::array &b,
                                         float maxAbsDiff) {
    af::dtype aDtype = (af::dtype)af::dtype_traits<T>::af_type;
    if (aDtype != b.type()) {
        return ::testing::AssertionFailure()
               << "TYPE MISMATCH:\n"
               << "  Actual: " << bName << "(" << b.type() << ")\n"
               << "Expected: " << aName << "(" << aDtype << ")";
    }

    if (aDims != b.dims()) {
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
    return elemWiseEq<T>(aName, bName, hA, aDims, hB, b.dims(), maxAbsDiff,
                         tag);
}

// To support C API
template<typename T>
::testing::AssertionResult assertArrayEq(std::string hA_name,
                                         std::string aDimsName,
                                         std::string bName,
                                         const std::vector<T> &hA,
                                         af::dim4 aDims, const af_array b) {
    af_array bb = 0;
    af_retain_array(&bb, b);
    af::array bbb(bb);
    return assertArrayEq(hA_name, aDimsName, bName, hA, aDims, bbb);
}

// Called by ASSERT_VEC_ARRAY_NEAR
template<typename T>
::testing::AssertionResult assertArrayNear(
    std::string hA_name, std::string aDimsName, std::string bName,
    std::string maxAbsDiffName, const std::vector<T> &hA, af::dim4 aDims,
    const af::array &b, float maxAbsDiff) {
    UNUSED(maxAbsDiffName);
    return assertArrayEq(hA_name, aDimsName, bName, hA, aDims, b, maxAbsDiff);
}

// To support C API
template<typename T>
::testing::AssertionResult assertArrayNear(
    std::string hA_name, std::string aDimsName, std::string bName,
    std::string maxAbsDiffName, const std::vector<T> &hA, af::dim4 aDims,
    const af_array b, float maxAbsDiff) {
    af_array bb = 0;
    af_retain_array(&bb, b);
    af::array bbb(bb);
    return assertArrayNear(hA_name, aDimsName, bName, maxAbsDiffName, hA, aDims,
                           bbb, maxAbsDiff);
}

::testing::AssertionResult assertRefEq(std::string hA_name,
                                       std::string expected_name,
                                       const af::array &a, int expected) {
    int count = 0;
    af_get_data_ref_count(&count, a.get());
    if (count != expected) {
        std::stringstream ss;
        ss << "Incorrect reference count:\nExpected: " << expected << "\n"
           << std::setw(8) << hA_name << ": " << count;

        return ::testing::AssertionFailure() << ss.str();

    } else {
        return ::testing::AssertionSuccess();
    }
}

#define INSTANTIATE(To)                                                        \
    template std::string printContext(                                         \
        const std::vector<To> &hGold, std::string goldName,                    \
        const std::vector<To> &hOut, std::string outName, af::dim4 arrDims,    \
        af::dim4 arrStrides, dim_t idx);                                       \
    template ::testing::AssertionResult assertArrayEq<To>(                     \
        std::string aName, std::string aDimsName, std::string bName,           \
        const std::vector<To> &hA, af::dim4 aDims, const af::array &b,         \
        float maxAbsDiff);                                                     \
    template ::testing::AssertionResult assertArrayEq<To>(                     \
        std::string hA_name, std::string aDimsName, std::string bName,         \
        const std::vector<To> &hA, af::dim4 aDims, const af_array b);          \
    template ::testing::AssertionResult assertArrayNear<To>(                   \
        std::string hA_name, std::string aDimsName, std::string bName,         \
        std::string maxAbsDiffName, const std::vector<To> &hA, af::dim4 aDims, \
        const af_array b, float maxAbsDiff);                                   \
    template ::testing::AssertionResult assertArrayNear<To>(                   \
        std::string hA_name, std::string aDimsName, std::string bName,         \
        std::string maxAbsDiffName, const std::vector<To> &hA, af::dim4 aDims, \
        const af::array &b, float maxAbsDiff)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(unsigned char);
INSTANTIATE(half_float::half);
INSTANTIATE(unsigned int);
INSTANTIATE(unsigned short);
INSTANTIATE(int);
INSTANTIATE(char);
INSTANTIATE(short);
INSTANTIATE(af_cdouble);
INSTANTIATE(af_cfloat);
INSTANTIATE(long long);
INSTANTIATE(unsigned long long);
INSTANTIATE(std::complex<float>);
INSTANTIATE(std::complex<double>);
#undef INSTANTIATE

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
