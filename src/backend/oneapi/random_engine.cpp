/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreemengt can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <err_oneapi.hpp>
#include <kernel/random_engine.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>

using arrayfire::common::half;

namespace arrayfire {
namespace oneapi {
void initMersenneState(Array<uint> &state, const uintl seed,
                       const Array<uint> &tbl) {
    kernel::initMersenneState(state, tbl, seed);
}

template<typename T>
Array<T> uniformDistribution(const af::dim4 &dims,
                             const af_random_engine_type type,
                             const uintl &seed, uintl &counter) {
    Array<T> out = createEmptyArray<T>(dims);
    kernel::uniformDistributionCBRNG<T>(out, out.elements(), type, seed,
                                        counter);
    return out;
}

template<typename T>
Array<T> normalDistribution(const af::dim4 &dims,
                            const af_random_engine_type type, const uintl &seed,
                            uintl &counter) {
    Array<T> out = createEmptyArray<T>(dims);
    kernel::normalDistributionCBRNG<T>(out, out.elements(), type, seed,
                                       counter);
    return out;
}

template<typename T>
Array<T> uniformDistribution(const af::dim4 &dims, Array<uint> pos,
                             Array<uint> sh1, Array<uint> sh2, uint mask,
                             Array<uint> recursion_table,
                             Array<uint> temper_table, Array<uint> state) {
    Array<T> out = createEmptyArray<T>(dims);
    kernel::uniformDistributionMT<T>(out, out.elements(), state, pos, sh1, sh2,
                                     mask, recursion_table, temper_table);
    return out;
}

template<typename T>
Array<T> normalDistribution(const af::dim4 &dims, Array<uint> pos,
                            Array<uint> sh1, Array<uint> sh2, uint mask,
                            Array<uint> recursion_table,
                            Array<uint> temper_table, Array<uint> state) {
    Array<T> out = createEmptyArray<T>(dims);
    kernel::normalDistributionMT<T>(out, out.elements(), state, pos, sh1, sh2,
                                    mask, recursion_table, temper_table);
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

INSTANTIATE_UNIFORM(float)
INSTANTIATE_UNIFORM(double)
INSTANTIATE_UNIFORM(cfloat)
INSTANTIATE_UNIFORM(cdouble)
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
INSTANTIATE_NORMAL(cdouble)
INSTANTIATE_NORMAL(cfloat)
INSTANTIATE_NORMAL(half)

}  // namespace oneapi
}  // namespace arrayfire
