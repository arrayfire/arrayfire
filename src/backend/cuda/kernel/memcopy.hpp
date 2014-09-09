#include <af/defines.h>
#include <backend.hpp>
#include <dispatch.hpp>
#include <backend.hpp>

namespace cuda
{
namespace kernel
{

    typedef struct
    {
        dim_type dim[4];
    } dims_t;

    static const uint DIMX = 32;
    static const uint DIMY =  8;

    template<typename T>
    __global__ static void
    memcopy_kernel(T *out, const dims_t ostrides,
                   const T *in, const dims_t idims,
                   const dims_t istrides, uint blocks_x, uint blocks_y)
    {
        const uint tidx = threadIdx.x;
        const uint tidy = threadIdx.y;

        const uint zid = blockIdx.x / blocks_x;
        const uint wid = blockIdx.y / blocks_y;
        const uint blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const uint blockIdx_y = blockIdx.y - (blocks_y) * wid;
        const uint xid = blockIdx_x * blockDim.x + tidx;
        const uint yid = blockIdx_y * blockDim.y + tidy;

        // FIXME: Do more work per block
        out += wid * ostrides.dim[3] + zid * ostrides.dim[2] + yid * ostrides.dim[1];
        in  += wid * istrides.dim[3] + zid * istrides.dim[2] + yid * istrides.dim[1];

        dim_type istride0 = istrides.dim[0];
        if (xid < idims.dim[0] &&
            yid < idims.dim[1] &&
            zid < idims.dim[2] &&
            wid < idims.dim[3]) {
            out[xid] = in[xid * istride0];
        }

    }

    template<typename T>
    void memcopy(T *out, const dim_type *ostrides,
                 const T *in, const dim_type *idims,
                 const dim_type *istrides, uint ndims)
    {
        dim3 threads(DIMX, DIMY);

        if (ndims == 1) {
            threads.x *= threads.y;
            threads.y  = 1;
       }

        // FIXME: DO more work per block
        uint blocks_x = divup(idims[0], threads.x);
        uint blocks_y = divup(idims[1], threads.y);

        dim3 blocks(blocks_x * idims[2],
                    blocks_y * idims[3]);

        dims_t _ostrides = {{ostrides[0], ostrides[1], ostrides[2], ostrides[3]}};
        dims_t _istrides = {{istrides[0], istrides[1], istrides[2], istrides[3]}};
        dims_t _idims = {{idims[0], idims[1], idims[2], idims[3]}};

        (memcopy_kernel<T>)<<<blocks, threads>>>(out, _ostrides,
                                                 in, _idims, _istrides,
                                                 blocks_x, blocks_y);
    }

}
}
