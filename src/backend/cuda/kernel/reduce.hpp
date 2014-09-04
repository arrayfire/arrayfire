#include <af/defines.h>
#include <ops.hpp>
#include <backend.hpp>
#include "../helper.hpp"
#include "../Param.hpp"

namespace cuda
{
namespace kernel
{
    static const uint THREADS_PER_BLOCK = 256;
    static const uint THREADS_X = 32;
    static const uint THREADS_Y = THREADS_PER_BLOCK / THREADS_X;
    static const uint REPEAT    = 32;

    template<typename Ti, typename To, af_op_t op, uint dim, uint DIMY>
    __global__
    static void reduce_dim_kernel(Param<To> out,
                                  CParam <Ti> in,
                                  uint blocks_x, uint blocks_y, uint offset_dim)
    {
        const uint tidx = threadIdx.x;
        const uint tidy = threadIdx.y;
        const uint tid  = tidy * THREADS_X + tidx;

        const uint zid = blockIdx.x / blocks_x;
        const uint wid = blockIdx.y / blocks_y;
        const uint blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const uint blockIdx_y = blockIdx.y - (blocks_y) * wid;
        const uint xid = blockIdx_x * blockDim.x + tidx;
        const uint yid = blockIdx_y;

        uint ids[4] = {xid, yid, zid, wid};

        const Ti *iptr = in.ptr;
        To *optr = out.ptr;

        // There is only one element per block for out
        // There are blockDim.y elements per block for in
        // Hence increment ids[dim] just after offseting out and before offsetting in
        optr += ids[3] * out.strides[3] + ids[2] * out.strides[2] + ids[1] * out.strides[1] + ids[0];
        const uint id_dim_out = ids[dim];

        ids[dim] = ids[dim] * blockDim.y + tidy;
        iptr  += ids[3] * in.strides[3] + ids[2] * in.strides[2] + ids[1] * in.strides[1] + ids[0];
        const uint id_dim_in = ids[dim];

        const uint istride_dim = in.strides[dim];

        bool is_valid =
            (ids[0] < in.dims[0]) &&
            (ids[1] < in.dims[1]) &&
            (ids[2] < in.dims[2]) &&
            (ids[3] < in.dims[3]);

        transform_op<Ti, To, op> Transform;
        reduce_op<To, op> Reduce;

        __shared__ To s_val[THREADS_X * DIMY];

        To out_val = Reduce.init();
        for (int id = id_dim_in; is_valid && (id < in.dims[dim]); id += offset_dim * blockDim.y) {
            To in_val = Transform(*iptr);
            out_val = Reduce.calc(in_val, out_val);
            iptr = iptr + offset_dim * blockDim.y * istride_dim;
        }

        s_val[tid] = out_val;

        To *s_ptr = s_val + tid;
        __syncthreads();

        if (DIMY == 8) {
            if (tidy < 4) *s_ptr = Reduce.calc(*s_ptr, s_ptr[THREADS_X * 4]);
            __syncthreads();
        }

        if (DIMY >= 4) {
            if (tidy < 2) *s_ptr = Reduce.calc(*s_ptr, s_ptr[THREADS_X * 2]);
            __syncthreads();
        }

        if (DIMY >= 2) {
            if (tidy < 1) *s_ptr = Reduce.calc(*s_ptr, s_ptr[THREADS_X * 1]);
            __syncthreads();
        }

        if (tidy == 0 && is_valid &&
            (id_dim_out < out.dims[dim])) {
            *optr = *s_ptr;
        }

    }

    template<typename Ti, typename To, af_op_t op, int dim>
    void reduce_dim_launcher(Param<To> out, CParam<Ti> in,
                             const uint threads_y, const uint blocks_dim[4])
    {
        dim3 threads(THREADS_X, threads_y);

        dim3 blocks(blocks_dim[0] * blocks_dim[2],
                    blocks_dim[1] * blocks_dim[3]);

        switch (threads_y) {
        case 8:
            return (reduce_dim_kernel<Ti, To, op, dim, 8>)<<<blocks, threads>>>(
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim]);
        case 4:
            return (reduce_dim_kernel<Ti, To, op, dim, 4>)<<<blocks, threads>>>(
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim]);
        case 2:
            return (reduce_dim_kernel<Ti, To, op, dim, 2>)<<<blocks, threads>>>(
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim]);
        case 1:
            return (reduce_dim_kernel<Ti, To, op, dim, 1>)<<<blocks, threads>>>(
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim]);
        case 32:
            return (reduce_dim_kernel<Ti, To, op, dim,32>)<<<blocks, threads>>>(
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim]);
        case 16:
            return (reduce_dim_kernel<Ti, To, op, dim,16>)<<<blocks, threads>>>(
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim]);
        }
    }

    template<typename Ti, typename To, af_op_t op, int dim>
    void reduce_dim(Param<To> out,  CParam<Ti> in)
    {
        uint threads_y = std::min(THREADS_Y, nextpow2(in.dims[dim]));
        uint threads_x = THREADS_X;

        uint blocks_dim[] = {divup(in.dims[0], threads_x),
                             in.dims[1], in.dims[2], in.dims[3]};

        blocks_dim[dim] = divup(in.dims[dim], threads_y * REPEAT);

        Param<To> tmp = out;

        dim_type tmp_elements = 1;
        if (blocks_dim[dim] > 1) {
            tmp.dims[dim] = blocks_dim[dim];

            for (int k = 0; k < 4; k++) tmp_elements *= tmp.dims[k];

            // FIXME: Add checks, use memory manager
            cudaMalloc(&tmp.ptr, tmp_elements * sizeof(To));

            for (int k = dim + 1; k < 4; k++) tmp.strides[k] *= blocks_dim[dim];
        }

        reduce_dim_launcher<Ti, To, op, dim>(tmp, in, threads_y, blocks_dim);

        if (blocks_dim[dim] > 1) {
            blocks_dim[dim] = 1;

            if (op == af_notzero_t) {
                reduce_dim_launcher<To, To, af_add_t, dim>(out, tmp, threads_y, blocks_dim);
            } else {
                reduce_dim_launcher<To, To,       op, dim>(out, tmp, threads_y, blocks_dim);
            }

            // FIXME: Add checks, use memory manager
            cudaFree(tmp.ptr);
        }

    }

    template<typename Ti, typename To, af_op_t op, uint DIMX>
    __global__
    static void reduce_first_kernel(Param<To> out,
                                    CParam<Ti>  in,
                                    uint blocks_x, uint blocks_y)
    {
        const uint tidx = threadIdx.x;
        const uint tidy = threadIdx.y;
        const uint tid  = tidy * blockDim.x + tidx;

        const uint zid = blockIdx.x / blocks_x;
        const uint wid = blockIdx.y / blocks_y;
        const uint blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const uint blockIdx_y = blockIdx.y - (blocks_y) * wid;
        const uint xid = blockIdx_x * blockDim.x + tidx;
        const uint yid = blockIdx_y * blockDim.y + tidy;

        const Ti *iptr = in.ptr;
        To *optr = out.ptr;

        iptr += wid *  in.strides[3] + zid *  in.strides[2] + yid *  in.strides[1];
        optr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];

        if (yid >= in.dims[1] ||
            zid >= in.dims[2] ||
            wid >= in.dims[3]) return;

        transform_op<Ti, To, op> Transform;
        reduce_op<To, op> Reduce;

        __shared__ To s_val[THREADS_PER_BLOCK];

        To out_val = Reduce.init();
        for (int id = xid; id < in.dims[0]; id += blockDim.x * blocks_x) {
            To in_val = Transform(iptr[id]);
            out_val = Reduce.calc(in_val, out_val);
        }

        s_val[tid] = out_val;
        __syncthreads();
        To *s_ptr = s_val + tidy * DIMX;

        if (DIMX == 256) {
            if (tidx < 128) s_ptr[tidx] = Reduce.calc(s_ptr[tidx], s_ptr[tidx + 128]);
            __syncthreads();
        }

        if (DIMX >= 128) {
            if (tidx <  64) s_ptr[tidx] = Reduce.calc(s_ptr[tidx], s_ptr[tidx +  64]);
            __syncthreads();
        }

        if (DIMX >=  64) {
            if (tidx <  32) s_ptr[tidx] = Reduce.calc(s_ptr[tidx], s_ptr[tidx +  32]);
            __syncthreads();
        }

        if (tidx < 16) s_ptr[tidx] = Reduce.calc(s_ptr[tidx], s_ptr[tidx + 16]);
        if (tidx <  8) s_ptr[tidx] = Reduce.calc(s_ptr[tidx], s_ptr[tidx +  8]);
        if (tidx <  4) s_ptr[tidx] = Reduce.calc(s_ptr[tidx], s_ptr[tidx +  4]);
        if (tidx <  2) s_ptr[tidx] = Reduce.calc(s_ptr[tidx], s_ptr[tidx +  2]);
        if (tidx <  1) s_ptr[tidx] = Reduce.calc(s_ptr[tidx], s_ptr[tidx +  1]);

        if (tidx == 0) {
            optr[blockIdx_x] = s_ptr[0];
        }
    }

    template<typename Ti, typename To, af_op_t op>
    void reduce_first_launcher(Param<To> out, CParam<Ti> in,
                               const uint blocks_x, const uint blocks_y, const uint threads_x)
    {

        dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
        dim3 blocks(blocks_x * in.dims[2],
                    blocks_y * in.dims[3]);

        switch (threads_x) {
        case 32:
            return (reduce_first_kernel<Ti, To, op,  32>)<<<blocks, threads>>>(
                out, in, blocks_x, blocks_y);
        case 64:
            return (reduce_first_kernel<Ti, To, op,  64>)<<<blocks, threads>>>(
                out, in, blocks_x, blocks_y);
        case 128:
            return (reduce_first_kernel<Ti, To, op,  128>)<<<blocks, threads>>>(
                out, in, blocks_x, blocks_y);
        case 256:
            return (reduce_first_kernel<Ti, To, op,  256>)<<<blocks, threads>>>(
                out, in, blocks_x, blocks_y);
        case 512:
            return (reduce_first_kernel<Ti, To, op,  512>)<<<blocks, threads>>>(
                out, in, blocks_x, blocks_y);
        case 1024:
            return (reduce_first_kernel<Ti, To, op,  1024>)<<<blocks, threads>>>(
                out, in, blocks_x, blocks_y);
        }
    }

    template<typename Ti, typename To, af_op_t op>
    void reduce_first(Param<To> out, CParam<Ti> in)
    {
        uint threads_x = nextpow2(std::max(32u, (uint)in.dims[0]));
        threads_x = std::min(threads_x, THREADS_PER_BLOCK);
        uint threads_y = THREADS_PER_BLOCK / threads_x;

        uint blocks_x = divup(in.dims[0], threads_x * REPEAT);
        uint blocks_y = divup(in.dims[1], threads_y);

        Param<To> tmp = out;

        if (blocks_x > 1) {
            // FIXME: Add checks, Use memory manager
            cudaMalloc(
                &(tmp.ptr),
                blocks_x * in.dims[1] * in.dims[2] * in.dims[3] * sizeof(To)
                );

            tmp.dims[0] = blocks_x;
            for (int k = 1; k < 4; k++) tmp.strides[k] *= blocks_x;
        }

        reduce_first_launcher<Ti, To, op>(tmp, in, blocks_x, blocks_y, threads_x);

        if (blocks_x > 1) {

            //FIXME: Is there an alternative to the if condition ?
            if (op == af_notzero_t) {
                reduce_first_launcher<To, To, af_add_t>(out, tmp, 1, blocks_y, threads_x);
            } else {
                reduce_first_launcher<To, To,       op>(out, tmp, 1, blocks_y, threads_x);
            }

            // FIXME: Add checks, memory manager
            cudaFree(tmp.ptr);
        }
    }

    template<typename Ti, typename To, af_op_t op>
    void reduce(Param<To> out, CParam<Ti> in, dim_type dim)
    {
        switch (dim) {
        case 0: return reduce_first<Ti, To, op   >(out, in);
        case 1: return reduce_dim  <Ti, To, op, 1>(out, in);
        case 2: return reduce_dim  <Ti, To, op, 2>(out, in);
        case 3: return reduce_dim  <Ti, To, op, 3>(out, in);
        }
    }
}
}
