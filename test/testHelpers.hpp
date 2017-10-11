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

#include <string>
#include <fstream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cfloat>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/array.h>

typedef unsigned char  uchar;
typedef unsigned int   uint;
typedef unsigned short ushort;

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
            unsigned nElems = inputDims[k].elements();
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
            unsigned nElems = inputDims[k].elements();
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

    std::cout<<"NRMSD = "<<NRMSD<<std::endl;
    if (std::isnan(NRMSD) || NRMSD > tolerance)
        return false;

    return true;
}

template<typename T, typename Other>
struct is_same_type{
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

    dim_t elements = (isTypeCplx ? 2 : 1) * dims.elements();

    std::vector<BT> out(elements);
    for(int i = 0; i < (int)elements; i++) {
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

#pragma GCC diagnostic pop
