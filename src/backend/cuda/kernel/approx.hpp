#include <math.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>

namespace cuda
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const dim_type TX = 16;
        static const dim_type TY = 16;
        static const dim_type THREADS = 256;

        ///////////////////////////////////////////////////////////////////////////
        // nearest-neighbor resampling
        ///////////////////////////////////////////////////////////////////////////
        template<typename Ty, typename Tp>
        __device__ inline static
        void core_nearest1(const dim_type idx, const dim_type idy, const dim_type idz, const dim_type idw,
                           Param<Ty> out, CParam<Ty> in, CParam<Tp> pos,
                           const float offGrid)
        {
            const dim_type omId = idw * out.strides[3] + idz * out.strides[2]
                                + idy * out.strides[1] + idx;
            const dim_type pmId = idx;

            const Tp x = pos.ptr[pmId];
            if (x < 0 || in.dims[0] < x+1) {
                out.ptr[omId] = scalar<Ty>(offGrid);
                return;
            }

            dim_type ioff = idw * in.strides[3] + idz * in.strides[2] + idy * in.strides[1];
            const dim_type iMem = round(x) + ioff;

            Ty yt = in.ptr[iMem];
            out.ptr[omId] = yt;
        }

        template<typename Ty, typename Tp>
        __device__ inline static
        void core_nearest2(const dim_type idx, const dim_type idy, const dim_type idz, const dim_type idw,
                           Param<Ty> out, CParam<Ty> in,
                           CParam<Tp> pos, CParam<Tp> qos, const float offGrid)
        {
            const dim_type omId = idw * out.strides[3] + idz * out.strides[2]
                                + idy * out.strides[1] + idx;
            const dim_type pmId = idy * pos.strides[1] + idx;
            const dim_type qmId = idy * qos.strides[1] + idx;

            const Tp x = pos.ptr[pmId], y = qos.ptr[qmId];
            if (x < 0 || y < 0 || in.dims[0] < x+1 || in.dims[1] < y+1) {
                out.ptr[omId] = scalar<Ty>(offGrid);
                return;
            }

            const dim_type grid_x = round(x), grid_y = round(y); // nearest grid
            const dim_type imId = idw * in.strides[3] + idz * in.strides[2]
                             + grid_y * in.strides[1] + grid_x;

            Ty val = in.ptr[imId];
            out.ptr[omId] = val;
        }

        ///////////////////////////////////////////////////////////////////////////
        // linear resampling
        ///////////////////////////////////////////////////////////////////////////
        template<typename Ty, typename Tp>
        __device__ inline static
        void core_linear1(const dim_type idx, const dim_type idy, const dim_type idz, const dim_type idw,
                          Param<Ty> out, CParam<Ty> in, CParam<Tp> pos,
                          const float offGrid)
        {
            const dim_type omId = idw * out.strides[3] + idz * out.strides[2]
                                + idy * out.strides[1] + idx;
            const dim_type pmId = idx;

            const Tp pVal = pos.ptr[pmId];
            if (pVal < 0 || in.dims[0] < pVal+1) {
                out.ptr[omId] = scalar<Ty>(offGrid);
                return;
            }

            const Tp grid_x = floor(pVal);  // nearest grid
            const Tp off_x = pVal - grid_x; // fractional offset

            dim_type ioff = idw * in.strides[3] + idz * in.strides[2] + idy * in.strides[1] + grid_x;

            // Check if pVal and pVal + 1 are both valid indices
            bool cond = (pVal < in.dims[0] - 1);
            // Compute Left and Right Weighted Values
            Ty yl = ((Tp)1.0 - off_x) * in.ptr[ioff];
            Ty yr = cond ? (off_x) * in.ptr[ioff + 1] : scalar<Ty>(0);
            Ty yo = yl + yr;
            // Compute Weight used
            Tp wt = cond ? (Tp)1.0 : (Tp)(1.0 - off_x);
            // Write final value
            out.ptr[omId] = (yo / wt);
        }

        template<typename Ty, typename Tp>
        __device__ inline static
        void core_linear2(const dim_type idx, const dim_type idy, const dim_type idz, const dim_type idw,
                           Param<Ty> out, CParam<Ty> in,
                           CParam<Tp> pos, CParam<Tp> qos, const float offGrid)
        {
            const dim_type omId = idw * out.strides[3] + idz * out.strides[2]
                                + idy * out.strides[1] + idx;
            const dim_type pmId = idy * pos.strides[1] + idx;
            const dim_type qmId = idy * qos.strides[1] + idx;

            const Tp x = pos.ptr[pmId], y = qos.ptr[qmId];
            if (x < 0 || y < 0 || in.dims[0] < x+1 || in.dims[1] < y+1) {
                out.ptr[omId] = scalar<Ty>(offGrid);
                return;
            }

            const Tp grid_x = floor(x),   grid_y = floor(y);   // nearest grid
            const Tp off_x  = x - grid_x, off_y  = y - grid_y; // fractional offset

            dim_type ioff = idw * in.strides[3] + idz * in.strides[2] + grid_y * in.strides[1] + grid_x;

            // Check if pVal and pVal + 1 are both valid indices
            bool condY = (y < in.dims[1] - 1);
            bool condX = (x < in.dims[0] - 1);

            // Compute wieghts used
            Tp wt00 = ((Tp)1.0 - off_x) * ((Tp)1.0 - off_y);
            Tp wt10 = (condY) ? ((Tp)1.0 - off_x) * (off_y) : 0;
            Tp wt01 = (condX) ? (off_x) * ((Tp)1.0 - off_y) : 0;
            Tp wt11 = (condX && condY) ? (off_x) * (off_y)  : 0;

            Tp wt = wt00 + wt10 + wt01 + wt11;

            // Compute Weighted Values
            Ty zero = scalar<Ty>(0);
            Ty y00 =                    wt00 * in.ptr[ioff];
            Ty y10 = (condY) ?          wt10 * in.ptr[ioff + in.strides[1]]     : zero;
            Ty y01 = (condX) ?          wt01 * in.ptr[ioff + 1]                 : zero;
            Ty y11 = (condX && condY) ? wt11 * in.ptr[ioff + in.strides[1] + 1] : zero;
            Ty yo = y00 + y10 + y01 + y11;

            // Write Final Value
            out.ptr[omId] = (yo / wt);
        }

        ///////////////////////////////////////////////////////////////////////////
        // Approx Kernel
        ///////////////////////////////////////////////////////////////////////////
        template<typename Ty, typename Tp, af_interp_type method>
        __global__
        void approx1_kernel(Param<Ty> out, CParam<Ty> in, CParam<Tp> pos,
                            const float offGrid, const dim_type blocksMatX)
        {
            const dim_type idw = blockIdx.y / out.dims[2];
            const dim_type idz = blockIdx.y - idw * out.dims[2];

            const dim_type idy = blockIdx.x / blocksMatX;
            const dim_type blockIdx_x = blockIdx.x - idy * blocksMatX;
            const dim_type idx = blockIdx_x * blockDim.x + threadIdx.x;

            if (idx >= out.dims[0] || idy >= out.dims[1] ||
                idz >= out.dims[2] || idw >= out.dims[3])
                return;

            switch(method) {
                case AF_INTERP_NEAREST:
                    core_nearest1(idx, idy, idz, idw, out, in, pos, offGrid);
                    break;
                case AF_INTERP_LINEAR:
                    core_linear1(idx, idy, idz, idw, out, in, pos, offGrid);
                    break;
                default:
                    break;
            }
        }

        template<typename Ty, typename Tp, af_interp_type method>
        __global__
        void approx2_kernel(Param<Ty> out, CParam<Ty> in,
                      CParam<Tp> pos, CParam<Tp> qos, const float offGrid,
                      const dim_type blocksMatX, const dim_type blocksMatY)
        {
            const dim_type idz = blockIdx.x / blocksMatX;
            const dim_type idw = blockIdx.y / blocksMatY;

            dim_type blockIdx_x = blockIdx.x - idz * blocksMatX;
            dim_type blockIdx_y = blockIdx.y - idw * blocksMatY;

            dim_type idx = threadIdx.x + blockIdx_x * blockDim.x;
            dim_type idy = threadIdx.y + blockIdx_y * blockDim.y;

            if (idx >= out.dims[0] || idy >= out.dims[1] ||
                idz >= out.dims[2] || idw >= out.dims[3])
                return;

            switch(method) {
                case AF_INTERP_NEAREST:
                    core_nearest2(idx, idy, idz, idw, out, in, pos, qos, offGrid);
                    break;
                case AF_INTERP_LINEAR:
                    core_linear2(idx, idy, idz, idw, out, in, pos, qos, offGrid);
                    break;
                default:
                    break;
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <typename Ty, typename Tp, af_interp_type method>
        void approx1(Param<Ty> out, CParam<Ty> in,
               CParam<Tp> pos, const float offGrid)
        {
            dim3 threads(THREADS, 1, 1);
            dim_type blocksPerMat = divup(out.dims[0], threads.x);
            dim3 blocks(blocksPerMat * out.dims[1], out.dims[2] * out.dims[3]);

            approx1_kernel<Ty, Tp, method><<<blocks, threads>>>
                          (out, in, pos, offGrid, blocksPerMat);
            POST_LAUNCH_CHECK();
        }

        template <typename Ty, typename Tp, af_interp_type method>
        void approx2(Param<Ty> out, CParam<Ty> in,
                    CParam<Tp> pos, CParam<Tp> qos, const float offGrid)
        {
            dim3 threads(TX, TY, 1);
            dim_type blocksPerMatX = divup(out.dims[0], threads.x);
            dim_type blocksPerMatY = divup(out.dims[1], threads.y);
            dim3 blocks(blocksPerMatX * out.dims[2], blocksPerMatY * out.dims[3]);

            approx2_kernel<Ty, Tp, method><<<blocks, threads>>>
                          (out, in, pos, qos, offGrid, blocksPerMatX, blocksPerMatY);
            POST_LAUNCH_CHECK();
        }
    }
}
