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
#include <reduce.hpp>
#include <ops.hpp>
#include <functional>
#include <complex>
#include <platform.hpp>
#include <queue.hpp>
#include <kernel/reduce.hpp>

using af::dim4;

template<>
struct Binary<cdouble, af_add_t>
{
    cdouble init()
    {
        return cdouble(0,0);
    }

    cdouble operator()(cdouble lhs, cdouble rhs)
    {
        return cdouble(real(lhs)+real(rhs), imag(lhs)+imag(rhs));
    }
};

namespace cpu
{

template<af_op_t op, typename Ti, typename To>
using reduce_dim_func = std::function<void(Param<To>, const dim_t,
                                           CParam<Ti>, const dim_t,
                                           const int, bool, double)>;

template<af_op_t op, typename Ti, typename To>
Array<To> reduce(const Array<Ti> &in, const int dim, bool change_nan, double nanval)
{
    dim4 odims = in.dims();
    odims[dim] = 1;
    in.eval();

    Array<To> out = createEmptyArray<To>(odims);
    static const reduce_dim_func<op, Ti, To>  reduce_funcs[4] = { kernel::reduce_dim<op, Ti, To, 1>()
                                                                , kernel::reduce_dim<op, Ti, To, 2>()
                                                                , kernel::reduce_dim<op, Ti, To, 3>()
                                                                , kernel::reduce_dim<op, Ti, To, 4>()};

    getQueue().enqueue(reduce_funcs[in.ndims() - 1], out, 0, in, 0, dim, change_nan, nanval);

    return out;
}

template<af_op_t op, typename Ti, typename To>
To reduce_all(const Array<Ti> &in, bool change_nan, double nanval)
{
    in.eval();
    getQueue().sync();

    Transform<Ti, To, op> transform;
    Binary<To, op> reduce;

    To out = reduce.init();

    // Decrement dimension of select dimension
    af::dim4 dims = in.dims();
    af::dim4 strides = in.strides();
    const Ti *inPtr = in.get();

    for(dim_t l = 0; l < dims[3]; l++) {
        dim_t off3 = l * strides[3];

        for(dim_t k = 0; k < dims[2]; k++) {
            dim_t off2 = k * strides[2];

            for(dim_t j = 0; j < dims[1]; j++) {
                dim_t off1 = j * strides[1];

                for(dim_t i = 0; i < dims[0]; i++) {
                    dim_t idx = i + off1 + off2 + off3;

                    To in_val = transform(inPtr[idx]);
                    if (change_nan) in_val = IS_NAN(in_val) ? nanval : in_val;
                    out = reduce(in_val, out);
                }
            }
        }
    }

    return out;
}

#define INSTANTIATE(ROp, Ti, To)                                        \
    template Array<To> reduce<ROp, Ti, To>(const Array<Ti> &in, const int dim, \
                                           bool change_nan, double nanval); \
    template To reduce_all<ROp, Ti, To>(const Array<Ti> &in,            \
                                        bool change_nan, double nanval);

//min
INSTANTIATE(af_min_t, float  , float  )
INSTANTIATE(af_min_t, double , double )
INSTANTIATE(af_min_t, cfloat , cfloat )
INSTANTIATE(af_min_t, cdouble, cdouble)
INSTANTIATE(af_min_t, int    , int    )
INSTANTIATE(af_min_t, uint   , uint   )
INSTANTIATE(af_min_t, intl   , intl   )
INSTANTIATE(af_min_t, uintl  , uintl  )
INSTANTIATE(af_min_t, char   , char   )
INSTANTIATE(af_min_t, uchar  , uchar  )
INSTANTIATE(af_min_t, short  , short  )
INSTANTIATE(af_min_t, ushort , ushort )

//max
INSTANTIATE(af_max_t, float  , float  )
INSTANTIATE(af_max_t, double , double )
INSTANTIATE(af_max_t, cfloat , cfloat )
INSTANTIATE(af_max_t, cdouble, cdouble)
INSTANTIATE(af_max_t, int    , int    )
INSTANTIATE(af_max_t, uint   , uint   )
INSTANTIATE(af_max_t, intl   , intl   )
INSTANTIATE(af_max_t, uintl  , uintl  )
INSTANTIATE(af_max_t, char   , char   )
INSTANTIATE(af_max_t, uchar  , uchar  )
INSTANTIATE(af_max_t, short  , short  )
INSTANTIATE(af_max_t, ushort , ushort )

//sum
INSTANTIATE(af_add_t, float  , float  )
INSTANTIATE(af_add_t, double , double )
INSTANTIATE(af_add_t, cfloat , cfloat )
INSTANTIATE(af_add_t, cdouble, cdouble)
INSTANTIATE(af_add_t, int    , int    )
INSTANTIATE(af_add_t, int    , float  )
INSTANTIATE(af_add_t, uint   , uint   )
INSTANTIATE(af_add_t, uint   , float  )
INSTANTIATE(af_add_t, intl   , intl   )
INSTANTIATE(af_add_t, intl   , double )
INSTANTIATE(af_add_t, uintl  , uintl  )
INSTANTIATE(af_add_t, uintl  , double )
INSTANTIATE(af_add_t, char   , int    )
INSTANTIATE(af_add_t, char   , float  )
INSTANTIATE(af_add_t, uchar  , uint   )
INSTANTIATE(af_add_t, uchar  , float  )
INSTANTIATE(af_add_t, short  , int    )
INSTANTIATE(af_add_t, short  , float  )
INSTANTIATE(af_add_t, ushort , uint   )
INSTANTIATE(af_add_t, ushort , float  )

//mul
INSTANTIATE(af_mul_t, float  , float  )
INSTANTIATE(af_mul_t, double , double )
INSTANTIATE(af_mul_t, cfloat , cfloat )
INSTANTIATE(af_mul_t, cdouble, cdouble)
INSTANTIATE(af_mul_t, int    , int    )
INSTANTIATE(af_mul_t, uint   , uint   )
INSTANTIATE(af_mul_t, intl   , intl   )
INSTANTIATE(af_mul_t, uintl  , uintl  )
INSTANTIATE(af_mul_t, char   , int    )
INSTANTIATE(af_mul_t, uchar  , uint   )
INSTANTIATE(af_mul_t, short  , int    )
INSTANTIATE(af_mul_t, ushort , uint   )

// count
INSTANTIATE(af_notzero_t, float  , uint)
INSTANTIATE(af_notzero_t, double , uint)
INSTANTIATE(af_notzero_t, cfloat , uint)
INSTANTIATE(af_notzero_t, cdouble, uint)
INSTANTIATE(af_notzero_t, int    , uint)
INSTANTIATE(af_notzero_t, uint   , uint)
INSTANTIATE(af_notzero_t, intl   , uint)
INSTANTIATE(af_notzero_t, uintl  , uint)
INSTANTIATE(af_notzero_t, char   , uint)
INSTANTIATE(af_notzero_t, uchar  , uint)
INSTANTIATE(af_notzero_t, short  , uint)
INSTANTIATE(af_notzero_t, ushort , uint)

//anytrue
INSTANTIATE(af_or_t, float  , char)
INSTANTIATE(af_or_t, double , char)
INSTANTIATE(af_or_t, cfloat , char)
INSTANTIATE(af_or_t, cdouble, char)
INSTANTIATE(af_or_t, int    , char)
INSTANTIATE(af_or_t, uint   , char)
INSTANTIATE(af_or_t, intl   , char)
INSTANTIATE(af_or_t, uintl  , char)
INSTANTIATE(af_or_t, char   , char)
INSTANTIATE(af_or_t, uchar  , char)
INSTANTIATE(af_or_t, short  , char)
INSTANTIATE(af_or_t, ushort , char)

//alltrue
INSTANTIATE(af_and_t, float  , char)
INSTANTIATE(af_and_t, double , char)
INSTANTIATE(af_and_t, cfloat , char)
INSTANTIATE(af_and_t, cdouble, char)
INSTANTIATE(af_and_t, int    , char)
INSTANTIATE(af_and_t, uint   , char)
INSTANTIATE(af_and_t, intl   , char)
INSTANTIATE(af_and_t, uintl  , char)
INSTANTIATE(af_and_t, char   , char)
INSTANTIATE(af_and_t, uchar  , char)
INSTANTIATE(af_and_t, short  , char)
INSTANTIATE(af_and_t, ushort , char)

}
