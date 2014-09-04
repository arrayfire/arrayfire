#include <stdio.h>
#include <complex.hpp>

namespace cuda
{
    namespace kernel
    {
        typedef struct
        {
            dim_type dim[4];
        } dims_t;

        // Kernel Launch Config Values
        static const dim_type TX = 16;
        static const dim_type TY = 16;
        static const dim_type THREADS = 256;

        /* divide and round up */
        static inline unsigned divup(unsigned n, unsigned threads)
        {
            return (n % threads) ? (n / threads + 1) : (n / threads);
        }

        ///////////////////////////////////////////////////////////////////////////
        // nearest-neighbor resampling
        ///////////////////////////////////////////////////////////////////////////
        template<typename Ty, typename Tp>
        __device__ inline static
        void core_nearest1(const dim_type idx, const dim_type idy, const dim_type idz, const dim_type idw,
                                 Ty *d_out, const dims_t odims, const dim_type oElems,
                           const Ty *d_in,  const dims_t idims, const dim_type iElems,
                           const Tp *d_pos, const dims_t pdims,
                           const dims_t ostrides, const dims_t istrides,
                           const dims_t pstrides, const float offGrid)
        {
            const dim_type omId = idw * ostrides.dim[3] + idz * ostrides.dim[2]
                                + idy * ostrides.dim[1] + idx;
            const dim_type pmId = idx;

            const Tp x = d_pos[pmId];
            if (x < 0 || idims.dim[0] < x+1) {
                d_out[omId] = constant<Ty>(offGrid);
                return;
            }

            dim_type ioff = idw * istrides.dim[3] + idz * istrides.dim[2] + idy * istrides.dim[1];
            const dim_type iMem = round(x) + ioff;

            Ty yt = d_in[iMem];
            d_out[omId] = yt;
        }

        template<typename Ty, typename Tp>
        __device__ inline static
        void core_nearest2(const dim_type idx, const dim_type idy, const dim_type idz, const dim_type idw,
                                 Ty *d_out, const dims_t odims, const dim_type oElems,
                           const Ty *d_in,  const dims_t idims, const dim_type iElems,
                           const Tp *d_pos, const dims_t pdims, const Tp *d_qos, const dims_t qdims,
                           const dims_t ostrides, const dims_t istrides,
                           const dims_t pstrides, const dims_t qstrides, const float offGrid)
        {
            const dim_type omId = idw * ostrides.dim[3] + idz * ostrides.dim[2]
                                + idy * ostrides.dim[1] + idx;
            const dim_type pmId = idy * pstrides.dim[1] + idx;
            const dim_type qmId = idy * qstrides.dim[1] + idx;

            const Tp x = d_pos[pmId], y = d_qos[qmId];
            if (x < 0 || y < 0 || idims.dim[0] < x+1 || idims.dim[1] < y+1) {
                d_out[omId] = constant<Ty>(offGrid);
                return;
            }

            const dim_type grid_x = round(x), grid_y = round(y); // nearest grid
            const dim_type imId = idw * istrides.dim[3] + idz * istrides.dim[2]
                                + grid_y * istrides.dim[1] + grid_x;

            Ty val = d_in[imId];

            d_out[omId] = val;
        }

        ///////////////////////////////////////////////////////////////////////////
        // linear resampling
        ///////////////////////////////////////////////////////////////////////////
        template<typename Ty, typename Tp>
        __device__ inline static
        void core_linear1(const dim_type idx, const dim_type idy, const dim_type idz, const dim_type idw,
                                Ty *d_out, const dims_t odims, const dim_type oElems,
                          const Ty *d_in,  const dims_t idims, const dim_type iElems,
                          const Tp *d_pos, const dims_t pdims,
                          const dims_t ostrides, const dims_t istrides,
                          const dims_t pstrides, const float offGrid)
        {
            const dim_type omId = idw * ostrides.dim[3] + idz * ostrides.dim[2]
                                + idy * ostrides.dim[1] + idx;
            const dim_type pmId = idx;

            const Tp pVal = d_pos[pmId];
            if (pVal < 0 || idims.dim[0] < pVal+1) {
                d_out[omId] = constant<Ty>(offGrid);
                return;
            }

            const Tp grid_x = floor(pVal);  // nearest grid
            const Tp off_x = pVal - grid_x; // fractional offset

            Tp w = 0;
            Ty y = constant<Ty>(0);
            dim_type ioff = idw * istrides.dim[3] + idz * istrides.dim[2] + idy * istrides.dim[1];
            for(dim_type xx = 0; xx <= (pVal < idims.dim[0] - 1); ++xx) {
                Tp fxx = (Tp)(xx);
                Tp wx = 1 - fabs(off_x - fxx);
                dim_type imId = (dim_type)(fxx + grid_x) + ioff;
                Ty yt = d_in[imId];
                y = y + (yt * wx);
                w = w + wx;
            }
            d_out[omId] = (y / w);
        }

        template<typename Ty, typename Tp>
        __device__ inline static
        void core_linear2(const dim_type idx, const dim_type idy, const dim_type idz, const dim_type idw,
                                Ty *d_out, const dims_t odims, const dim_type oElems,
                          const Ty *d_in,  const dims_t idims, const dim_type iElems,
                          const Tp *d_pos, const dims_t pdims, const Tp *d_qos, const dims_t qdims,
                          const dims_t ostrides, const dims_t istrides,
                          const dims_t pstrides, const dims_t qstrides, const float offGrid)
        {
            const dim_type omId = idw * ostrides.dim[3] + idz * ostrides.dim[2]
                                + idy * ostrides.dim[1] + idx;
            const dim_type pmId = idy * pstrides.dim[1] + idx;
            const dim_type qmId = idy * qstrides.dim[1] + idx;

            const Tp x = d_pos[pmId], y = d_qos[qmId];
            if (x < 0 || y < 0 || idims.dim[0] < x+1 || idims.dim[1] < y+1) {
                d_out[omId] = constant<Ty>(offGrid);
                return;
            }

            const Tp grid_x = floor(x),   grid_y = floor(y);   // nearest grid
            const Tp off_x  = x - grid_x, off_y  = y - grid_y; // fractional offset

            Tp w = 0;
            Ty z = constant<Ty>(0);
            dim_type ioff = idw * istrides.dim[3] + idz * istrides.dim[2];
            for(dim_type yy = 0; yy <= (y < idims.dim[1] - 1); ++yy) {
                Tp fyy = (Tp)(yy);
                Tp wy = 1 - fabs(off_y - fyy);
                dim_type idyy = (dim_type)(fyy + grid_y);
                for(dim_type xx = 0; xx <= (x < idims.dim[0] - 1); ++xx) {
                    Tp fxx = (Tp)(xx);
                    Tp wxy = (1 - fabs(off_x - fxx)) * wy;
                    dim_type imId = idyy * istrides.dim[1] + (dim_type)(fxx + grid_x) + ioff;
                    Ty zt = d_in[imId];
                    z = z + (zt * wxy);
                    w = w + wxy;
                }
            }
            d_out[omId] = z / w;
        }

        ///////////////////////////////////////////////////////////////////////////
        // Approx Kernel
        ///////////////////////////////////////////////////////////////////////////
        template<typename Ty, typename Tp, af_interp_type method>
        __global__
        void approx1_kernel(      Ty* d_out, const dims_t odims, const dim_type oElems,
                            const Ty* d_in,  const dims_t idims, const dim_type iElems,
                            const Tp* d_pos, const dims_t pdims,
                            const dims_t ostrides, const dims_t istrides,
                            const dims_t pstrides, const float offGrid, const dim_type blocksMatX)
        {
            const dim_type idw = blockIdx.y / odims.dim[2];
            const dim_type idz = blockIdx.y - idw * odims.dim[2];

            const dim_type idy = blockIdx.x / blocksMatX;
            const dim_type blockIdx_x = blockIdx.x - idy * blocksMatX;
            const dim_type idx = blockIdx_x * blockDim.x + threadIdx.x;

            if (idx >= odims.dim[0] || idy >= odims.dim[1] ||
                idz >= odims.dim[2] || idw >= odims.dim[3])
                return;

            switch(method) {
                case AF_INTERP_NEAREST:
                    core_nearest1(idx, idy, idz, idw, d_out, odims, oElems, d_in, idims, iElems,
                            d_pos, pdims, ostrides, istrides, pstrides, offGrid);
                    break;
                case AF_INTERP_LINEAR:
                    core_linear1(idx, idy, idz, idw, d_out, odims, oElems, d_in, idims, iElems,
                            d_pos, pdims, ostrides, istrides, pstrides, offGrid);
                    break;
                default:
                    break;
            }
        }

        template<typename Ty, typename Tp, af_interp_type method>
        __global__
        void approx2_kernel(      Ty *d_out, const dims_t odims, const dim_type oElems,
                            const Ty *d_in,  const dims_t idims, const dim_type iElems,
                            const Tp *d_pos, const dims_t pdims, const Tp *d_qos, const dims_t qdims,
                            const dims_t ostrides, const dims_t istrides,
                            const dims_t pstrides, const dims_t qstrides,
                            const float offGrid, const dim_type blocksMatX, const dim_type blocksMatY)
        {
            const dim_type idz = blockIdx.x / blocksMatX;
            const dim_type idw = blockIdx.y / blocksMatY;

            dim_type blockIdx_x = blockIdx.x - idz * blocksMatX;
            dim_type blockIdx_y = blockIdx.y - idw * blocksMatY;

            dim_type idx = threadIdx.x + blockIdx_x * blockDim.x;
            dim_type idy = threadIdx.y + blockIdx_y * blockDim.y;

            if (idx >= odims.dim[0] || idy >= odims.dim[1] ||
                idz >= odims.dim[2] || idw >= odims.dim[3])
                return;

            switch(method) {
                case AF_INTERP_NEAREST:
                    core_nearest2(idx, idy, idz, idw, d_out, odims, oElems, d_in, idims, iElems,
                            d_pos, pdims, d_qos, qdims, ostrides, istrides, pstrides, qstrides, offGrid);
                    break;
                case AF_INTERP_LINEAR:
                    core_linear2(idx, idy, idz, idw, d_out, odims, oElems, d_in, idims, iElems,
                            d_pos, pdims, d_qos, qdims, ostrides, istrides, pstrides, qstrides, offGrid);
                    break;
                default:
                    break;
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <typename Ty, typename Tp, af_interp_type method>
        void approx1(      Ty *out, const dim_type *odims, const dim_type oElems,
                     const Ty *in,  const dim_type *idims, const dim_type iElems,
                     const Tp *pos, const dim_type *pdims, const dim_type *ostrides,
                     const dim_type *istrides, const dim_type *pstrides, const float offGrid)
        {
            dim3 threads(THREADS, 1, 1);
            dim_type blocksPerMat = divup(odims[0], threads.x);
            dim3 blocks(blocksPerMat * odims[1], odims[2] * odims[3]);

            dims_t _odims = {{odims[0], odims[1], odims[2], odims[3]}};
            dims_t _idims = {{idims[0], idims[1], idims[2], idims[3]}};
            dims_t _pdims = {{pdims[0], pdims[1], pdims[2], pdims[3]}};
            dims_t _ostrides = {{ostrides[0], ostrides[1], ostrides[2], ostrides[3]}};
            dims_t _istrides = {{istrides[0], istrides[1], istrides[2], istrides[3]}};
            dims_t _pstrides = {{pstrides[0], pstrides[1], pstrides[2], pstrides[3]}};

            approx1_kernel<Ty, Tp, method><<<blocks, threads>>>
                          (out, _odims, oElems, in, _idims, iElems, pos, _pdims,
                           _ostrides, _istrides, _pstrides, offGrid, blocksPerMat);
        }

        template <typename Ty, typename Tp, af_interp_type method>
        void approx2(      Ty *out, const dim_type *odims, const dim_type oElems,
                     const Ty *in,  const dim_type *idims, const dim_type iElems,
                     const Tp *pos, const dim_type *pdims, const Tp *qos, const dim_type *qdims,
                     const dim_type *ostrides, const dim_type *istrides,
                     const dim_type *pstrides, const dim_type *qstrides,
                     const float offGrid)
        {
            dim3 threads(TX, TY, 1);
            dim_type blocksPerMatX = divup(odims[0], threads.x);
            dim_type blocksPerMatY = divup(odims[1], threads.y);
            dim3 blocks(blocksPerMatX * odims[2], blocksPerMatY * odims[3]);

            dims_t _odims = {{odims[0], odims[1], odims[2], odims[3]}};
            dims_t _idims = {{idims[0], idims[1], idims[2], idims[3]}};
            dims_t _pdims = {{pdims[0], pdims[1], pdims[2], pdims[3]}};
            dims_t _qdims = {{qdims[0], qdims[1], qdims[2], qdims[3]}};
            dims_t _ostrides = {{ostrides[0], ostrides[1], ostrides[2], ostrides[3]}};
            dims_t _istrides = {{istrides[0], istrides[1], istrides[2], istrides[3]}};
            dims_t _pstrides = {{pstrides[0], pstrides[1], pstrides[2], pstrides[3]}};
            dims_t _qstrides = {{qstrides[0], qstrides[1], qstrides[2], qstrides[3]}};

            approx2_kernel<Ty, Tp, method><<<blocks, threads>>>
                          (out, _odims, oElems, in, _idims, iElems, pos, _pdims, qos, _qdims,
                           _ostrides, _istrides, _pstrides, _qstrides, offGrid,
                           blocksPerMatX, blocksPerMatY);
        }
    }
}
