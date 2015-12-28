/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <gradient.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_cpu.hpp>
#include <debug_cpu.hpp>
#include <kernel/gradient.hpp>

namespace cpu
{

template<typename T>
void gradient(Array<T> &grad0, Array<T> &grad1, const Array<T> &in)
{
    grad0.eval();
    grad1.eval();
    in.eval();

    ENQUEUE(kernel::gradient<T>, grad0, grad1, in);
}

#define INSTANTIATE(T)                                                                  \
    template void gradient<T>(Array<T> &grad0, Array<T> &grad1, const Array<T> &in);    \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

}
