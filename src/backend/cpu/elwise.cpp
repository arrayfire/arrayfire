/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/array.h>
#include <af/arith.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <elwise.hpp>
#include <algorithm>

namespace cpu
{

    using std::swap;
    template<typename Tl, typename Tr, typename To, typename Op>
    Array<To>* binOp(const Array<Tl> &lhs, const Array<Tr> &rhs)
    {
        Array<To> *res= createEmptyArray<To>(lhs.dims());
        std::transform(lhs.get(), lhs.get() + lhs.elements(),
                       rhs.get(),
                       res->get(),
                       Op());

        return res;
    }

#define INSTANTIATE(L, R, O, OP)                                        \
    template Array<O>* binOp<L, R, O, OP>(const Array<L> &lhs, const Array<R> &rhs); \

    INSTANTIATE(float ,  float,  float, std::plus<float>);
    INSTANTIATE(float , double, double, std::plus<double>);
    INSTANTIATE(double,  float, double, std::plus<double>);
    INSTANTIATE(double, double, double, std::plus<double>);

} //namespace af
