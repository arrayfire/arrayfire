/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <kernel/lookup.hpp>
#include <lookup.hpp>
#include <err_opencl.hpp>

namespace opencl
{
template<typename in_t, typename idx_t>
Array<in_t> lookup(const Array<in_t> &input,
                   const Array<idx_t> &indices, const unsigned dim)
{
    const dim4 iDims = input.dims();

    dim4 oDims(1);
    for (int d=0; d<4; ++d)
        oDims[d] = (d==int(dim) ? indices.elements() : iDims[d]);

    Array<in_t> out = createEmptyArray<in_t>(oDims);

    dim_t nDims = iDims.ndims();

    switch(dim) {
        case 0: kernel::lookup<in_t, idx_t, 0>(out, input, indices, nDims); break;
        case 1: kernel::lookup<in_t, idx_t, 1>(out, input, indices, nDims); break;
        case 2: kernel::lookup<in_t, idx_t, 2>(out, input, indices, nDims); break;
        case 3: kernel::lookup<in_t, idx_t, 3>(out, input, indices, nDims); break;
    }

    return out;
}

#define INSTANTIATE(T)  \
template Array<T> lookup<T, float   >(const Array<T>&, const Array<float   >&, const unsigned); \
template Array<T> lookup<T, double  >(const Array<T>&, const Array<double  >&, const unsigned); \
template Array<T> lookup<T, int     >(const Array<T>&, const Array<int     >&, const unsigned); \
template Array<T> lookup<T, unsigned>(const Array<T>&, const Array<unsigned>&, const unsigned); \
template Array<T> lookup<T, short   >(const Array<T>&, const Array<short   >&, const unsigned); \
template Array<T> lookup<T, ushort  >(const Array<T>&, const Array<ushort  >&, const unsigned); \
template Array<T> lookup<T, intl    >(const Array<T>&, const Array<intl    >&, const unsigned); \
template Array<T> lookup<T, uintl   >(const Array<T>&, const Array<uintl   >&, const unsigned); \
template Array<T> lookup<T, uchar   >(const Array<T>&, const Array<uchar   >&, const unsigned);

INSTANTIATE(float   );
INSTANTIATE(cfloat  );
INSTANTIATE(double  );
INSTANTIATE(cdouble );
INSTANTIATE(int     );
INSTANTIATE(unsigned);
INSTANTIATE(intl    );
INSTANTIATE(uintl   );
INSTANTIATE(uchar   );
INSTANTIATE(char    );
INSTANTIATE(ushort  );
INSTANTIATE(short   );
}
