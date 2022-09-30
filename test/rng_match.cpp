/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>

#include <sstream>
#include <vector>

using af::array;
using af::dim4;
using af::getAvailableBackends;
using af::randomEngine;
using af::randu;
using af::setBackend;
using af::setSeed;
using std::get;
using std::make_pair;
using std::stringstream;
using std::vector;

enum param { engine, backend, size, seed, type };

using rng_params =
    std::tuple<af::randomEngineType, std::pair<af::Backend, af::Backend>,
               af::dim4, int, af_dtype>;

class RNGMatch : public ::testing::TestWithParam<rng_params> {
   protected:
    void SetUp() {
        backends_available =
            getAvailableBackends() & get<backend>(GetParam()).first;
        backends_available =
            backends_available &&
            (getAvailableBackends() & get<backend>(GetParam()).second);

        if (backends_available) {
            setBackend(get<backend>(GetParam()).first);
            randomEngine(get<engine>(GetParam()));
            setSeed(get<seed>(GetParam()));
            array tmp  = randu(get<size>(GetParam()), get<type>(GetParam()));
            void* data = malloc(tmp.bytes());
            tmp.host(data);

            setBackend(get<backend>(GetParam()).second);
            values[0] = array(get<size>(GetParam()), get<type>(GetParam()));
            values[0].write(data, values[0].bytes());
            free(data);
            randomEngine(get<engine>(GetParam()));
            setSeed(get<seed>(GetParam()));
            values[1] = randu(get<size>(GetParam()), get<type>(GetParam()));
        }
    }

    array values[2];
    bool backends_available;
};

std::string engine_name(af::randomEngineType engine) {
    switch (engine) {
        case AF_RANDOM_ENGINE_PHILOX: return "PHILOX";
        case AF_RANDOM_ENGINE_THREEFRY: return "THREEFRY";
        case AF_RANDOM_ENGINE_MERSENNE: return "MERSENNE";
        default: return "UNKNOWN ENGINE";
    }
}

std::string backend_name(af::Backend backend) {
    switch (backend) {
        case AF_BACKEND_DEFAULT: return "DEFAULT";
        case AF_BACKEND_CPU: return "CPU";
        case AF_BACKEND_CUDA: return "CUDA";
        case AF_BACKEND_OPENCL: return "OPENCL";
        default: return "UNKNOWN BACKEND";
    }
}

std::string rngmatch_info(
    const ::testing::TestParamInfo<RNGMatch::ParamType> info) {
    stringstream ss;
    ss << "size_" << get<size>(info.param)[0] << "_"
       << backend_name(get<backend>(info.param).first) << "_"
       << backend_name(get<backend>(info.param).second) << "_"
       << get<size>(info.param)[1] << "_" << get<size>(info.param)[2] << "_"
       << get<size>(info.param)[3] << "_seed_" << get<seed>(info.param)
       << "_type_" << get<type>(info.param);
    return ss.str();
}

INSTANTIATE_TEST_SUITE_P(
    PhiloxCPU_CUDA, RNGMatch,
    ::testing::Combine(
        ::testing::Values(AF_RANDOM_ENGINE_PHILOX),
        ::testing::Values(make_pair(AF_BACKEND_CPU, AF_BACKEND_CUDA),
                          make_pair(AF_BACKEND_CPU, AF_BACKEND_OPENCL)),
        ::testing::Values(dim4(10), dim4(100), dim4(1000), dim4(10000),
                          dim4(1E5), dim4(10, 10), dim4(10, 100),
                          dim4(100, 100), dim4(1000, 100), dim4(10, 10, 10),
                          dim4(10, 100, 10), dim4(100, 100, 10),
                          dim4(1000, 100, 10), dim4(10, 10, 10, 10),
                          dim4(10, 100, 10, 10), dim4(100, 100, 10, 10),
                          dim4(1000, 100, 10, 10)),
        ::testing::Values(12), ::testing::Values(f32, f64, c32, c64, u8)),
    rngmatch_info);

TEST_P(RNGMatch, BackendEquals) {
    if (backends_available) {
        array actual   = values[0];
        array expected = values[1];
        ASSERT_ARRAYS_EQ(actual, expected);
    } else {
        printf("SKIPPED\n");
    }
}
