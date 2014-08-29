#include <af/defines.h>
#include <ops.hpp>
#include <backend.hpp>
#include "../helper.hpp"

namespace cuda
{
namespace kernel
{
    typedef struct
    {
        dim_type dim[4];
    } dims_t;

    static const uint THREADS_PER_BLOCK = 256;
    static const uint THREADS_X = 32;
    static const uint THREADS_Y = THREADS_PER_BLOCK / THREADS_X;
    static const uint REPEAT    = 32;

    template<typename Ti, typename To, af_op_t op, uint dim, uint DIMY>
    __global__
    static void reduce_dim_kernel(To *out, const dims_t ostrides, const dims_t odims,
                                  const Ti *in, const dims_t istrides, const dims_t idims,
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

        // There is only one element per block for out
        // There are blockDim.y elements per block for in
        // Hence increment ids[dim] just after offseting out and before offsetting in
        out += ids[3] * ostrides.dim[3] + ids[2] * ostrides.dim[2] + ids[1] * ostrides.dim[1] + ids[0];
        const uint id_dim_out = ids[dim];

        ids[dim] = ids[dim] * blockDim.y + tidy;
        in  += ids[3] * istrides.dim[3] + ids[2] * istrides.dim[2] + ids[1] * istrides.dim[1] + ids[0];
        const uint id_dim_in = ids[dim];

        const uint istride_dim = istrides.dim[dim];

        bool is_valid =
            (ids[0] < idims.dim[0]) &&
            (ids[1] < idims.dim[1]) &&
            (ids[2] < idims.dim[2]) &&
            (ids[3] < idims.dim[3]);

        transform_op<Ti, To, op> Transform;
        reduce_op<To, op> Reduce;

        __shared__ To s_val[THREADS_X * DIMY];

        To out_val = Reduce.init();
        for (int id = id_dim_in; is_valid && (id < idims.dim[dim]); id += offset_dim * blockDim.y) {
            To in_val = Transform(*in);
            out_val = Reduce.calc(in_val, out_val);
            in = in + offset_dim * blockDim.y * istride_dim;
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
            (id_dim_out < odims.dim[dim])) {
            *out = *s_ptr;
        }

    }

    template<typename Ti, typename To, af_op_t op, int dim>
    void reduce_dim_launcher(To *out, const dims_t ostrides, const dims_t odims,
                             const Ti *in, const dims_t istrides, const dims_t idims,
                             const uint threads_y, const uint blocks_dim[4])
    {
        dim3 threads(THREADS_X, threads_y);

        dim3 blocks(blocks_dim[0] * blocks_dim[2],
                    blocks_dim[1] * blocks_dim[3]);

        switch (threads_y) {
        case 8:
            return (reduce_dim_kernel<Ti, To, op, dim, 8>)<<<blocks, threads>>>(
                out, ostrides, odims, in, istrides, idims,
                blocks_dim[0], blocks_dim[1], blocks_dim[dim]);
        case 4:
            return (reduce_dim_kernel<Ti, To, op, dim, 4>)<<<blocks, threads>>>(
                out, ostrides, odims, in, istrides, idims,
                blocks_dim[0], blocks_dim[1], blocks_dim[dim]);
        case 2:
            return (reduce_dim_kernel<Ti, To, op, dim, 2>)<<<blocks, threads>>>(
                out, ostrides, odims, in, istrides, idims,
                blocks_dim[0], blocks_dim[1], blocks_dim[dim]);
        case 1:
            return (reduce_dim_kernel<Ti, To, op, dim, 1>)<<<blocks, threads>>>(
                out, ostrides, odims, in, istrides, idims,
                blocks_dim[0], blocks_dim[1], blocks_dim[dim]);
        case 32:
            return (reduce_dim_kernel<Ti, To, op, dim,32>)<<<blocks, threads>>>(
                out, ostrides, odims, in, istrides, idims,
                blocks_dim[0], blocks_dim[1], blocks_dim[dim]);
        case 16:
            return (reduce_dim_kernel<Ti, To, op, dim,16>)<<<blocks, threads>>>(
                out, ostrides, odims, in, istrides, idims,
                blocks_dim[0], blocks_dim[1], blocks_dim[dim]);
        }
    }

    template<typename Ti, typename To, af_op_t op, int dim>
    void reduce_dim(To *out, const dim_type *_ostrides, const dim_type *_odims,
                    const Ti *in, const dim_type *_istrides, const dim_type *_idims)
    {
        dims_t ostrides = {{_ostrides[0], _ostrides[1], _ostrides[2], _ostrides[3]}};
        dims_t istrides = {{_istrides[0], _istrides[1], _istrides[2], _istrides[3]}};
        dims_t odims = {{_odims[0], _odims[1], _odims[2], _odims[3]}};
        dims_t idims = {{_idims[0], _idims[1], _idims[2], _idims[3]}};
        uint threads_y = std::min(THREADS_Y, nextpow2(idims.dim[dim]));
        uint threads_x = THREADS_X;

        uint blocks_dim[] = {divup(idims.dim[0], threads_x),
                             idims.dim[1], idims.dim[2], idims.dim[3]};

        blocks_dim[dim] = divup(idims.dim[dim], threads_y * REPEAT);

        To *tmp = out;
        dims_t tstrides = {{_ostrides[0], _ostrides[1], _ostrides[2], _ostrides[3]}};
        dims_t tdims    = {{_odims[0], _odims[1], _odims[2], _odims[3]}};

        dim_type tmp_elements = 1;
        if (blocks_dim[dim] > 1) {

            tdims.dim[dim] = blocks_dim[dim];

            for (int k = 0; k < 4; k++) tmp_elements *= tdims.dim[k];

            // FIXME: Add checks, use memory manager
            cudaMalloc(&tmp, tmp_elements * sizeof(To));

            for (int k = dim + 1; k < 4; k++) tstrides.dim[k] *= blocks_dim[dim];
        }

        reduce_dim_launcher<Ti, To, op, dim>(tmp, tstrides, tdims,
                                             in, istrides, idims,
                                             threads_y, blocks_dim);

        if (blocks_dim[dim] > 1) {
            blocks_dim[dim] = 1;

            if (op == af_notzero_t) {
                reduce_dim_launcher<To, To, af_add_t, dim>(out, ostrides, odims,
                                                           tmp, tstrides, tdims,
                                                           threads_y, blocks_dim);
            } else {
                reduce_dim_launcher<To, To,           op, dim>(out, ostrides, odims,
                                                               tmp, tstrides, tdims,
                                                               threads_y, blocks_dim);;
            }

            // FIXME: Add checks, use memory manager
            cudaFree(tmp);
        }

    }

    template<typename Ti, typename To, af_op_t op, uint DIMX>
    __global__
    static void reduce_first_kernel(To *out, const dims_t ostrides,
                                    const Ti *in, const dims_t istrides, const dims_t idims,
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

        in  += wid * istrides.dim[3] + zid * istrides.dim[2] + yid * istrides.dim[1];
        out += wid * ostrides.dim[3] + zid * ostrides.dim[2] + yid * ostrides.dim[1];

        if (yid >= idims.dim[1] ||
            zid >= idims.dim[2] ||
            wid >= idims.dim[3]) return;

        transform_op<Ti, To, op> Transform;
        reduce_op<To, op> Reduce;

        __shared__ To s_val[THREADS_PER_BLOCK];

        To out_val = Reduce.init();
        for (int id = xid; id < idims.dim[0]; id += blockDim.x * blocks_x) {
            To in_val = Transform(in[id]);
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
            out[blockIdx_x] = s_ptr[0];
        }
    }

    template<typename Ti, typename To, af_op_t op>
    void reduce_first_launcher(To *out, const dims_t ostrides,
                               const Ti *in, const dims_t istrides, const dims_t idims,
                               const uint blocks_x, const uint blocks_y, const uint threads_x)
    {

        dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
        dim3 blocks(blocks_x * idims.dim[2],
                    blocks_y * idims.dim[3]);

        switch (threads_x) {
        case 32:
            return (reduce_first_kernel<Ti, To, op,  32>)<<<blocks, threads>>>(
                out, ostrides, in, istrides, idims, blocks_x, blocks_y);
        case 64:
            return (reduce_first_kernel<Ti, To, op,  64>)<<<blocks, threads>>>(
                out, ostrides, in, istrides, idims, blocks_x, blocks_y);
        case 128:
            return (reduce_first_kernel<Ti, To, op,  128>)<<<blocks, threads>>>(
                out, ostrides, in, istrides, idims, blocks_x, blocks_y);
        case 256:
            return (reduce_first_kernel<Ti, To, op,  256>)<<<blocks, threads>>>(
                out, ostrides, in, istrides, idims, blocks_x, blocks_y);
        case 512:
            return (reduce_first_kernel<Ti, To, op,  512>)<<<blocks, threads>>>(
                out, ostrides, in, istrides, idims, blocks_x, blocks_y);
        case 1024:
            return (reduce_first_kernel<Ti, To, op,  1024>)<<<blocks, threads>>>(
                out, ostrides, in, istrides, idims, blocks_x, blocks_y);
        }
    }

    template<typename Ti, typename To, af_op_t op>
    void reduce_first(To *out, const dim_type *_ostrides,
                      const Ti *in, const dim_type *_istrides, const dim_type *_idims)
    {
        dims_t ostrides = {{_ostrides[0], _ostrides[1], _ostrides[2], _ostrides[3]}};
        dims_t istrides = {{_istrides[0], _istrides[1], _istrides[2], _istrides[3]}};
        dims_t idims    = {{   _idims[0],    _idims[1],    _idims[2],    _idims[3]}};

        uint threads_x = nextpow2(std::max(32u, (uint)_idims[0]));
        threads_x = std::min(threads_x, THREADS_PER_BLOCK);
        uint threads_y = THREADS_PER_BLOCK / threads_x;

        uint blocks_x = divup(_idims[0], threads_x * REPEAT);
        uint blocks_y = divup(_idims[1], threads_y);

        To *tmp = out;
        dims_t tstrides = {{_ostrides[0], _ostrides[1], _ostrides[2], _ostrides[3]}};

        if (blocks_x > 1) {
            // FIXME: Add checks, Use memory manager
            cudaMalloc(&tmp, blocks_x * _idims[1] * _idims[2] * _idims[3] * sizeof(To));
            for (int k = 1; k < 4; k++) tstrides.dim[k] *= blocks_x;
        }

        reduce_first_launcher<Ti, To, op>(tmp, tstrides, in, istrides, idims,
                                          blocks_x, blocks_y, threads_x);

        if (blocks_x > 1) {
            dims_t tdims = {{blocks_x, _idims[1], _idims[2], _idims[3]}};

            //FIXME: Is there an alternative to the if condition ?
            if (op == af_notzero_t) {
                reduce_first_launcher<To, To, af_add_t>(out, ostrides, tmp, tstrides, tdims,
                                                        1, blocks_y, threads_x);
            } else {
                reduce_first_launcher<To, To,       op>(out, ostrides, tmp, tstrides, tdims,
                                                        1, blocks_y, threads_x);
            }

            // FIXME: Add checks, memory manager
            cudaFree(tmp);
        }
    }

    template<typename Ti, typename To, af_op_t op>
    void reduce(To *out, const dim_type *ostrides, const dim_type *odims,
                const Ti *in, const dim_type *istrides, const dim_type *idims, dim_type dim)
    {
        switch (dim) {
        case 0: return reduce_first<Ti, To, op> (out, ostrides,        in, istrides, idims);
        case 1: return reduce_dim<Ti, To, op, 1>(out, ostrides, odims, in, istrides, idims);
        case 2: return reduce_dim<Ti, To, op, 2>(out, ostrides, odims, in, istrides, idims);
        case 3: return reduce_dim<Ti, To, op, 3>(out, ostrides, odims, in, istrides, idims);
        }
    }
}
}
