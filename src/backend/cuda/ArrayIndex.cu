/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <ArrayIndex.hpp>
#include <err_cuda.hpp>

namespace cuda
{

template<typename in_t, typename idx_t>
Array<in_t>* arrayIndex(const Array<in_t> &input, const Array<idx_t> &indices, const unsigned dim)
{
    CUDA_NOT_SUPPORTED();
}

#define INSTANTIATE(T)  \
    template Array<T> * arrayIndex<T, float   >(const Array<T> &input, const Array<float   > &indices, const unsigned dim); \
    template Array<T> * arrayIndex<T, double  >(const Array<T> &input, const Array<double  > &indices, const unsigned dim); \
    template Array<T> * arrayIndex<T, int     >(const Array<T> &input, const Array<int     > &indices, const unsigned dim); \
    template Array<T> * arrayIndex<T, unsigned>(const Array<T> &input, const Array<unsigned> &indices, const unsigned dim); \
    template Array<T> * arrayIndex<T, uchar   >(const Array<T> &input, const Array<uchar   > &indices, const unsigned dim);

INSTANTIATE(float   );
INSTANTIATE(cfloat  );
INSTANTIATE(double  );
INSTANTIATE(cdouble );
INSTANTIATE(int     );
INSTANTIATE(unsigned);
INSTANTIATE(uchar   );
INSTANTIATE(char    );

}
