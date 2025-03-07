/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <kernel/random_engine.hpp>
#include <af/dim4.hpp>
#include <cassert>

using arrayfire::common::half;

namespace arrayfire {
namespace cuda {
void initMersenneState(Array<uint> &state, const uintl seed,
                       const Array<uint> &tbl) {
    kernel::initMersenneState(state.get(), tbl.get(), seed);
}

template<typename T>
Array<T> uniformDistribution(const af::dim4 &dims,
                             const af_random_engine_type type,
                             const uintl &seed, uintl &counter) {
    Array<T> out = createEmptyArray<T>(dims);
    kernel::uniformDistributionCBRNG<T>(out.get(), out.elements(), type, seed,
                                        counter);
    return out;
}

template<typename T>
Array<T> normalDistribution(const af::dim4 &dims,
                            const af_random_engine_type type, const uintl &seed,
                            uintl &counter) {
    Array<T> out = createEmptyArray<T>(dims);
    kernel::normalDistributionCBRNG<T>(out.get(), out.elements(), type, seed,
                                       counter);
    return out;
}

template<typename T>
Array<T> uniformDistribution(const af::dim4 &dims, Array<uint> pos,
                             Array<uint> sh1, Array<uint> sh2, uint mask,
                             Array<uint> recursion_table,
                             Array<uint> temper_table, Array<uint> state) {
    Array<T> out = createEmptyArray<T>(dims);
    kernel::uniformDistributionMT<T>(out.get(), out.elements(), state.get(),
                                     pos.get(), sh1.get(), sh2.get(), mask,
                                     recursion_table.get(), temper_table.get());
    return out;
}

template<typename T>
Array<T> normalDistribution(const af::dim4 &dims, Array<uint> pos,
                            Array<uint> sh1, Array<uint> sh2, uint mask,
                            Array<uint> recursion_table,
                            Array<uint> temper_table, Array<uint> state) {
    Array<T> out = createEmptyArray<T>(dims);
    kernel::normalDistributionMT<T>(out.get(), out.elements(), state.get(),
                                    pos.get(), sh1.get(), sh2.get(), mask,
                                    recursion_table.get(), temper_table.get());
    return out;
}

#define INSTANTIATE_UNIFORM(T)                                   \
    template Array<T> uniformDistribution<T>(                    \
        const af::dim4 &dims, const af_random_engine_type type,  \
        const uintl &seed, uintl &counter);                      \
    template Array<T> uniformDistribution<T>(                    \
        const af::dim4 &dims, Array<uint> pos, Array<uint> sh1,  \
        Array<uint> sh2, uint mask, Array<uint> recursion_table, \
        Array<uint> temper_table, Array<uint> state);

#define INSTANTIATE_NORMAL(T)                                    \
    template Array<T> normalDistribution<T>(                     \
        const af::dim4 &dims, const af_random_engine_type type,  \
        const uintl &seed, uintl &counter);                      \
    template Array<T> normalDistribution<T>(                     \
        const af::dim4 &dims, Array<uint> pos, Array<uint> sh1,  \
        Array<uint> sh2, uint mask, Array<uint> recursion_table, \
        Array<uint> temper_table, Array<uint> state);

#define COMPLEX_UNIFORM_DISTRIBUTION(T, TR)                                 \
    template<>                                                              \
    Array<T> uniformDistribution<T>(const af::dim4 &dims,                   \
                                    const af_random_engine_type type,       \
                                    const uintl &seed, uintl &counter) {    \
        Array<T> out    = createEmptyArray<T>(dims);                        \
        TR *outPtr      = (TR *)out.get();                                  \
        size_t elements = out.elements() * 2;                               \
        kernel::uniformDistributionCBRNG<TR>(outPtr, elements, type, seed,  \
                                             counter);                      \
        return out;                                                         \
    }                                                                       \
    template<>                                                              \
    Array<T> uniformDistribution<T>(                                        \
        const af::dim4 &dims, Array<uint> pos, Array<uint> sh1,             \
        Array<uint> sh2, uint mask, Array<uint> recursion_table,            \
        Array<uint> temper_table, Array<uint> state) {                      \
        Array<T> out    = createEmptyArray<T>(dims);                        \
        TR *outPtr      = (TR *)out.get();                                  \
        size_t elements = out.elements() * 2;                               \
        kernel::uniformDistributionMT<TR>(                                  \
            outPtr, elements, state.get(), pos.get(), sh1.get(), sh2.get(), \
            mask, recursion_table.get(), temper_table.get());               \
        return out;                                                         \
    }

#define COMPLEX_NORMAL_DISTRIBUTION(T, TR)                                  \
    template<>                                                              \
    Array<T> normalDistribution<T>(const af::dim4 &dims,                    \
                                   const af_random_engine_type type,        \
                                   const uintl &seed, uintl &counter) {     \
        Array<T> out    = createEmptyArray<T>(dims);                        \
        TR *outPtr      = (TR *)out.get();                                  \
        size_t elements = out.elements() * 2;                               \
        kernel::normalDistributionCBRNG<TR>(outPtr, elements, type, seed,   \
                                            counter);                       \
        return out;                                                         \
    }                                                                       \
    template<>                                                              \
    Array<T> normalDistribution<T>(                                         \
        const af::dim4 &dims, Array<uint> pos, Array<uint> sh1,             \
        Array<uint> sh2, uint mask, Array<uint> recursion_table,            \
        Array<uint> temper_table, Array<uint> state) {                      \
        Array<T> out    = createEmptyArray<T>(dims);                        \
        TR *outPtr      = (TR *)out.get();                                  \
        size_t elements = out.elements() * 2;                               \
        kernel::normalDistributionMT<TR>(                                   \
            outPtr, elements, state.get(), pos.get(), sh1.get(), sh2.get(), \
            mask, recursion_table.get(), temper_table.get());               \
        return out;                                                         \
    }

INSTANTIATE_UNIFORM(float)
INSTANTIATE_UNIFORM(double)
INSTANTIATE_UNIFORM(int)
INSTANTIATE_UNIFORM(uint)
INSTANTIATE_UNIFORM(intl)
INSTANTIATE_UNIFORM(uintl)
INSTANTIATE_UNIFORM(char)
INSTANTIATE_UNIFORM(uchar)
INSTANTIATE_UNIFORM(short)
INSTANTIATE_UNIFORM(ushort)
INSTANTIATE_UNIFORM(half)

INSTANTIATE_NORMAL(float)
INSTANTIATE_NORMAL(double)
INSTANTIATE_NORMAL(half)

COMPLEX_UNIFORM_DISTRIBUTION(cdouble, double)
COMPLEX_UNIFORM_DISTRIBUTION(cfloat, float)

COMPLEX_NORMAL_DISTRIBUTION(cdouble, double)
COMPLEX_NORMAL_DISTRIBUTION(cfloat, float)

}  // namespace cuda
}  // namespace arrayfire
