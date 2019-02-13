/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define GTEST_LINKED_AS_SHARED_LIBRARY 1
#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>

#include <sstream>
#include <vector>

using af::array;
using af::dim4;
using af::randomEngine;
using af::randu;
using af::setBackend;
using af::setSeed;
using std::stringstream;
using std::vector;

struct rng_params {
    af::randomEngineType engine;
    af::Backend backends[2];
    af::dim4 size;
    int seed;
    af_dtype type;
};

class RNGMatch : public ::testing::TestWithParam<rng_params> {
  protected:
    void SetUp() {

      //backends_available = getAvailableBackends()
      //;


        setBackend(GetParam().backends[0]);
        randomEngine(GetParam().engine);
        setSeed(GetParam().seed);
        array tmp = randu(GetParam().size);
        void* data = malloc(tmp.bytes());
        tmp.host(data);

        setBackend(GetParam().backends[1]);
        values[0] = array(GetParam().size);
        values[0].write(data, values[0].bytes());
        free(data);
        randomEngine(GetParam().engine);
        setSeed(GetParam().seed);
        values[1] = randu(GetParam().size);
    }

    array values[2];
    bool backends_available;

};

std::string engine_name(af::randomEngineType engine) {
  switch(engine) {
      case AF_RANDOM_ENGINE_PHILOX:  return "PHILOX";
      case AF_RANDOM_ENGINE_THREEFRY:  return "THREEFRY";
      case AF_RANDOM_ENGINE_MERSENNE:  return "MERSENNE";
  }
}

std::string backend_name(af::Backend backend) {
  switch(backend) {
  case AF_BACKEND_DEFAULT: return "DEFAULT";
  case AF_BACKEND_CPU: return "CPU";
  case AF_BACKEND_CUDA: return "CUDA";
  case AF_BACKEND_OPENCL: return "OPENCL";
  }
}


std::string rngmatch_info(const ::testing::TestParamInfo<RNGMatch::ParamType> info) {
                            stringstream ss;
                            ss << "size_" << info.param.size[0] << "_" << info.param.size[1]
                               << "_" << info.param.size[2] << "_" << info.param.size[3]
                               << "_seed_" << info.param.seed
                               << "_type_" << info.param.type;
                            return ss.str();
}

INSTANTIATE_TEST_CASE_P(PhiloxCPU_CUDA,
                        RNGMatch,
                        ::testing::Values(
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10000), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10, 10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100, 10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100, 10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100, 10, 10), 12, f32},

                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10000), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10, 10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100, 10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100, 10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100, 10, 10), 12, u8},

                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10000), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10, 10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100, 10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100, 10, 10), 12, c64},
                                            rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100, 10, 10), 12, c64}),
                                            rngmatch_info);

INSTANTIATE_TEST_CASE_P(MersenneCPU_CUDA,
                        RNGMatch,
                        ::testing::Values(
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10000), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10, 10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100, 10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100, 10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100, 10, 10), 12, f32},

                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10000), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10, 10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100, 10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100, 10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100, 10, 10), 12, u8},

                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10000), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 10, 10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(10, 100, 10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(100, 100, 10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_CUDA}, dim4(1000, 100, 10, 10), 12, c64}),
                        rngmatch_info);

INSTANTIATE_TEST_CASE_P(PhiloxCPU_OpenCL,
                        RNGMatch,
                        ::testing::Values(
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10000), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10, 10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100, 10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100, 10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100, 10, 10), 12, f32},

                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10000), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10, 10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100, 10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100, 10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100, 10, 10), 12, u8},

                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10000), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10, 10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100, 10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100, 10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_PHILOX, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100, 10, 10), 12, c64}
                                          ),
                        rngmatch_info);

INSTANTIATE_TEST_CASE_P(MersenneCPU_OPENCL,
                        RNGMatch,
                        ::testing::Values(
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10000), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10, 10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100, 10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100, 10, 10), 12, f32},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100, 10, 10), 12, f32},

                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10000), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10, 10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100, 10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100, 10, 10), 12, u8},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100, 10, 10), 12, u8},

                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10000), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 10, 10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(10, 100, 10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(100, 100, 10, 10), 12, c64},
                                          rng_params{AF_RANDOM_ENGINE_MERSENNE, {AF_BACKEND_CPU, AF_BACKEND_OPENCL}, dim4(1000, 100, 10, 10), 12, c64}),
                        rngmatch_info);


TEST_P(RNGMatch, BackendEquals) {
  ASSERT_ARRAYS_EQ(values[0], values[1]);
}
