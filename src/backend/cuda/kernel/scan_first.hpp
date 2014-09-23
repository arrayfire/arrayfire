#include <af/defines.h>
#include <ops.hpp>
#include <backend.hpp>
#include <Param.hpp>
#include <dispatch.hpp>
#include <math.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include "config.hpp"

namespace cuda
{
namespace kernel
{
    template<typename Ti, typename To, af_op_t op, bool isFinalPass, uint DIMX>
    __global__
    static void scan_first_kernel(Param<To> out,
                                  Param<To> tmp,
                                  CParam<Ti>  in,
                                  uint blocks_x,
                                  uint blocks_y,
                                  uint lim)
    {
        const uint tidx = threadIdx.x;
        const uint tidy = threadIdx.y;

        const uint zid = blockIdx.x / blocks_x;
        const uint wid = blockIdx.y / blocks_y;
        const uint blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const uint blockIdx_y = blockIdx.y - (blocks_y) * wid;
        const uint xid = blockIdx_x * blockDim.x * lim + tidx;
        const uint yid = blockIdx_y * blockDim.y + tidy;

        const Ti *iptr = in.ptr;
        To *optr = out.ptr;
        To *tptr = tmp.ptr;

        iptr += wid *  in.strides[3] + zid *  in.strides[2] + yid *  in.strides[1];
        optr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];
        tptr += wid * tmp.strides[3] + zid * tmp.strides[2] + yid * tmp.strides[1];

        bool cond_yzw = (yid < out.dims[1]) && (zid < out.dims[2]) && (wid < out.dims[3]);

        const uint DIMY = THREADS_PER_BLOCK / DIMX;
        const uint SHARED_MEM_SIZE = (DIMX + 1) * (2 * DIMY);

        __shared__ To s_val[SHARED_MEM_SIZE];
        __shared__ To s_tmp[DIMY];

        To *sptr = s_val + tidy * (DIMX + 1);
        const uint s_off = (DIMY * (DIMX + 1));
        int mul = 1;

        Transform<Ti, To, op> transform;
        Binary<To, op> binop;

        const To init = binop.init();
        uint id = xid;
        To val = init;

        const bool isLast = (tidx == (DIMX - 1));

        for (int k = 0; k < lim; k++) {

            if (isLast) s_tmp[tidy] = val;

            bool cond = (cond_yzw && (id < out.dims[0]));
            val = cond ? transform(iptr[id]) : init;
            sptr[tidx] = val;
            __syncthreads();

            for (int off = 1; off < DIMX; off *= 2) {
                if (tidx >= off) val = binop(val, sptr[tidx - off]);

                sptr += mul * s_off;
                sptr[tidx] = val;
                mul *= -1;

                __syncthreads();
            }

            val = binop(val, s_tmp[tidy]);
            if (cond) optr[id] = val;
            id += blockDim.x;
        }

        if (!isFinalPass && isLast) {
            tptr[blockIdx_x] = val;
        }
    }

    template<typename To, af_op_t op>
    __global__
    static void bcast_first_kernel(Param<To> out,
                                   CParam<To> tmp,
                                   uint blocks_x,
                                   uint blocks_y,
                                   uint lim)
    {
        const uint tidx = threadIdx.x;
        const uint tidy = threadIdx.y;

        const uint zid = blockIdx.x / blocks_x;
        const uint wid = blockIdx.y / blocks_y;
        const uint blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const uint blockIdx_y = blockIdx.y - (blocks_y) * wid;
        const uint xid = blockIdx_x * blockDim.x * lim + tidx;
        const uint yid = blockIdx_y * blockDim.y + tidy;

        To *optr = out.ptr;
        const To *tptr = tmp.ptr;

        optr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];
        tptr += wid * tmp.strides[3] + zid * tmp.strides[2] + yid * tmp.strides[1];

        bool cond = (yid < out.dims[1]) && (zid < out.dims[2]) && (wid < out.dims[3]);

        if (!cond) return;
        if (blockIdx_x == 0) return;

        Binary<To, op> binop;
        To accum = tptr[blockIdx_x - 1];

        for (int k = 0, id = xid;
             k < lim && id < out.dims[0];
             k++, id += blockDim.x) {

            optr[id] = binop(accum, optr[id]);
        }

    }

    template<typename Ti, typename To, af_op_t op, bool isFinalPass>
    void scan_first_launcher(Param<To> out,
                             Param<To> tmp,
                             CParam<Ti> in,
                             const uint blocks_x,
                             const uint blocks_y,
                             const uint threads_x)
    {

        dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
        dim3 blocks(blocks_x * out.dims[2],
                    blocks_y * out.dims[3]);

        uint lim = divup(out.dims[0], (threads_x * blocks_x));

        switch (threads_x) {
        case 32:
            (scan_first_kernel<Ti, To, op, isFinalPass,  32>)<<<blocks, threads>>>(
                out, tmp, in, blocks_x, blocks_y, lim); break;
        case 64:
            (scan_first_kernel<Ti, To, op, isFinalPass,  64>)<<<blocks, threads>>>(
                out, tmp, in, blocks_x, blocks_y, lim); break;
        case 128:
            (scan_first_kernel<Ti, To, op, isFinalPass,  128>)<<<blocks, threads>>>(
                out, tmp, in, blocks_x, blocks_y, lim); break;
        case 256:
            (scan_first_kernel<Ti, To, op, isFinalPass,  256>)<<<blocks, threads>>>(
                out, tmp, in, blocks_x, blocks_y, lim); break;
        }

        POST_LAUNCH_CHECK();
    }



    template<typename To, af_op_t op>
    void bcast_first_launcher(Param<To> out,
                              CParam<To> tmp,
                              const uint blocks_x,
                              const uint blocks_y,
                              const uint threads_x)
    {

        dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
        dim3 blocks(blocks_x * out.dims[2],
                    blocks_y * out.dims[3]);

        uint lim = divup(out.dims[0], (threads_x * blocks_x));

        (bcast_first_kernel<To, op>)<<<blocks, threads>>>(
            out, tmp, blocks_x, blocks_y, lim);

        POST_LAUNCH_CHECK();
    }

    template<typename Ti, typename To, af_op_t op>
    void scan_first(Param<To> out, CParam<Ti> in)
    {
        uint threads_x = nextpow2(std::max(32u, (uint)out.dims[0]));
        threads_x = std::min(threads_x, THREADS_PER_BLOCK);
        uint threads_y = THREADS_PER_BLOCK / threads_x;

        uint blocks_x = divup(out.dims[0], threads_x * REPEAT);
        uint blocks_y = divup(out.dims[1], threads_y);

        if (blocks_x == 1) {

            scan_first_launcher<Ti, To, op, true>(out, out, in,
                                                  blocks_x, blocks_y,
                                                  threads_x);

        } else {

            Param<To> tmp = out;

            tmp.dims[0] = blocks_x;
            tmp.strides[0] = 1;
            for (int k = 1; k < 4; k++) tmp.strides[k] = tmp.strides[k - 1] * tmp.dims[k - 1];

            dim_type tmp_elements = tmp.strides[3] * tmp.dims[3];
            CUDA_CHECK(cudaMalloc(&(tmp.ptr), tmp_elements * sizeof(To)));

            scan_first_launcher<Ti, To, op, false>(out, tmp, in,
                                                   blocks_x, blocks_y,
                                                   threads_x);

            //FIXME: Is there an alternative to the if condition ?
            if (op == af_notzero_t) {
                scan_first_launcher<To, To, af_add_t, true>(tmp, tmp, tmp,
                                                             1, blocks_y,
                                                             threads_x);
            } else {
                scan_first_launcher<To, To,       op, true>(tmp, tmp, tmp,
                                                            1, blocks_y,
                                                            threads_x);
            }

            bcast_first_launcher<To, op>(out, tmp, blocks_x, blocks_y, threads_x);

            CUDA_CHECK(cudaFree(tmp.ptr));
        }
    }

}
}
