/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <qr.hpp>
#include <err_common.hpp>

#if defined(WITH_LINEAR_ALGEBRA)

namespace opencl
{

template<typename T>
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in)
{
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<T> qr_inplace(Array<T> &in)
{
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_QR(T)                                                                           \
    template Array<T> qr_inplace<T>(Array<T> &in);                                                \
    template void qr<T>(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in);

INSTANTIATE_QR(float)
INSTANTIATE_QR(cfloat)
INSTANTIATE_QR(double)
INSTANTIATE_QR(cdouble)

}

#else

namespace opencl
{

template<typename T>
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in)
{
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<T> qr_inplace(Array<T> &in)
{
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_QR(T)                                                                           \
    template Array<T> qr_inplace<T>(Array<T> &in);                                                \
    template void qr<T>(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in);

INSTANTIATE_QR(float)
INSTANTIATE_QR(cfloat)
INSTANTIATE_QR(double)
INSTANTIATE_QR(cdouble)

}

#endif
