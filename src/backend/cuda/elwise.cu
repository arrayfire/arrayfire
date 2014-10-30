/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/arith.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <elwise.hpp>
#include <kernel/elwise.hpp>


namespace cuda
{

    using std::swap;

    template<typename Tl, typename Tr, typename To, typename Op>
    Array<To>* binOp(const Array<Tl> &lhs, const Array<Tr> &rhs)
    {
        Array<To> *res= createValueArray<To>(lhs.dims(), 0);
        kernel::binaryOp(res->get(), lhs.get(), rhs.get(), lhs.elements());
        return res;
    }



#define INSTANTIATE(L, R, O, OP)                                        \
    template Array<O>* binOp<L, R, O, OP>(const Array<L> &lhs, const Array<R> &rhs); \

    INSTANTIATE(float ,  float,  float, std::plus<float>);
    INSTANTIATE(float , double, double, std::plus<double>);
    INSTANTIATE(double,  float, double, std::plus<double>);
    INSTANTIATE(double, double, double, std::plus<double>);

} //namespace cuda
