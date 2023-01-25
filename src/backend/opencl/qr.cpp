/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <qr.hpp>

#include <err_opencl.hpp>

#if defined(WITH_LINEAR_ALGEBRA)

#include <blas.hpp>
#include <copy.hpp>
#include <cpu/cpu_qr.hpp>
#include <identity.hpp>
#include <kernel/triangle.hpp>
#include <magma/magma.h>
#include <magma/magma_data.h>
#include <magma/magma_helper.h>
#include <platform.hpp>

namespace arrayfire {
namespace opencl {

template<typename T>
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &orig) {
    if (OpenCLCPUOffload()) { return cpu::qr(q, r, t, orig); }

    const dim4 NullShape(0, 0, 0, 0);

    dim4 iDims = orig.dims();
    int M      = iDims[0];
    int N      = iDims[1];

    dim4 endPadding(M - iDims[0], max(M, N) - iDims[1], 0, 0);
    Array<T> in =
        (endPadding == NullShape
             ? copyArray(orig)
             : padArrayBorders(orig, NullShape, endPadding, AF_PAD_ZERO));
    in.resetDims(iDims);

    int MN = std::min(M, N);
    int NB = magma_get_geqrf_nb<T>(M);

    int NUM      = (2 * MN + ((N + 31) / 32) * 32) * NB;
    Array<T> tmp = createEmptyArray<T>(dim4(NUM));

    std::vector<T> h_tau(MN);

    int info           = 0;
    cl::Buffer *in_buf = in.get();
    cl::Buffer *dT     = tmp.get();

    magma_geqrf3_gpu<T>(M, N, (*in_buf)(), in.getOffset(), in.strides()[1],
                        &h_tau[0], (*dT)(), tmp.getOffset(), getQueue()(),
                        &info);

    r = createEmptyArray<T>(in.dims());
    kernel::triangle<T>(r, in, true, false);

    cl::Buffer *r_buf = r.get();
    magmablas_swapdblk<T>(MN - 1, NB, (*r_buf)(), r.getOffset(), r.strides()[1],
                          1, (*dT)(), tmp.getOffset() + MN * NB, NB, 0,
                          getQueue()());

    q = in;  // No need to copy
    q.resetDims(dim4(M, M));
    cl::Buffer *q_buf = q.get();

    magma_ungqr_gpu<T>(q.dims()[0], q.dims()[1], std::min(M, N), (*q_buf)(),
                       q.getOffset(), q.strides()[1], &h_tau[0], (*dT)(),
                       tmp.getOffset(), NB, getQueue()(), &info);

    t = createHostDataArray(dim4(MN), &h_tau[0]);
}

template<typename T>
Array<T> qr_inplace(Array<T> &in) {
    if (OpenCLCPUOffload()) { return cpu::qr_inplace(in); }

    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];
    int MN     = std::min(M, N);

    getQueue().finish();  // FIXME: Does this need to be here?
    cl::CommandQueue Queue2(getContext(), getDevice());
    cl_command_queue queues[] = {getQueue()(), Queue2()};

    std::vector<T> h_tau(MN);
    cl::Buffer *in_buf = in.get();

    int info = 0;
    magma_geqrf2_gpu<T>(M, N, (*in_buf)(), in.getOffset(), in.strides()[1],
                        &h_tau[0], queues, &info);

    Array<T> t = createHostDataArray(dim4(MN), &h_tau[0]);
    return t;
}

#define INSTANTIATE_QR(T)                                         \
    template Array<T> qr_inplace<T>(Array<T> & in);               \
    template void qr<T>(Array<T> & q, Array<T> & r, Array<T> & t, \
                        const Array<T> &in);

INSTANTIATE_QR(float)
INSTANTIATE_QR(cfloat)
INSTANTIATE_QR(double)
INSTANTIATE_QR(cdouble)

}  // namespace opencl
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace opencl {

template<typename T>
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<T> qr_inplace(Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_QR(T)                                         \
    template Array<T> qr_inplace<T>(Array<T> & in);               \
    template void qr<T>(Array<T> & q, Array<T> & r, Array<T> & t, \
                        const Array<T> &in);

INSTANTIATE_QR(float)
INSTANTIATE_QR(cfloat)
INSTANTIATE_QR(double)
INSTANTIATE_QR(cdouble)

}  // namespace opencl
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA
