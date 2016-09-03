/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
namespace cuda
{
    namespace kernel
    {

        template<typename Ty, typename Tp>
        __device__
        Ty linearInterpFunc(Ty val[2], Tp frac)
        {
            return (1 - frac) * val[0] + frac * val[1];
        }

        template<typename Ty, typename Tp>
        __device__
        Ty bilinearInterpFunc(Ty val[2][2], Tp xfrac, Tp yfrac)
        {
            Ty res[2];
            res[0] = linearInterpFunc(val[0], xfrac);
            res[1] = linearInterpFunc(val[1], xfrac);
            return linearInterpFunc(res, yfrac);
        }

        template<typename Ty, typename Tp>
        __device__ inline static
        Ty cubicInterpFunc(Ty val[4], Tp xfrac)
        {
            Ty a0 =
                scalar<Ty>(-0.5) * val[0] + scalar<Ty>( 1.5) * val[1] +
                scalar<Ty>(-1.5) * val[2] + scalar<Ty>( 0.5) * val[3];

            Ty a1 =
                scalar<Ty>( 1.0) * val[0] + scalar<Ty>(-2.5) * val[1] +
                scalar<Ty>( 2.0) * val[2] + scalar<Ty>(-0.5) * val[3];

            Ty a2 = scalar<Ty>(-0.5) * val[0] + scalar<Ty>(0.5) * val[2];

            Ty a3 = val[1];

            Tp xfrac2 = xfrac * xfrac;
            Tp xfrac3 = xfrac2 * xfrac;

            return a0 * xfrac3 + a1 * xfrac2 + a2 * xfrac + a3;
        }

        template<typename Ty, typename Tp>
        __device__ inline static
        Ty bicubicInterpFunc(Ty val[4][4], Tp xfrac, Tp yfrac) {
            Ty res[4];
            res[0] = cubicInterpFunc(val[0], xfrac);
            res[1] = cubicInterpFunc(val[1], xfrac);
            res[2] = cubicInterpFunc(val[2], xfrac);
            res[3] = cubicInterpFunc(val[3], xfrac);
            return cubicInterpFunc(res, yfrac);
        }

        template<typename Ty, typename Tp, af_interp_type>
        struct Interp1
        {
            __device__ Ty operator()(CParam<Ty> in, int ioff, Tp x)
            {
                const int idx = round(x) + ioff;
                return in.ptr[idx];
            }
        };

        template<typename Ty, typename Tp>
        struct Interp1<Ty, Tp, AF_INTERP_LINEAR>
        {
            __device__ Ty operator()(CParam<Ty> in, int ioff, Tp x)
            {
                const int grid_x = floor(x);    // nearest grid
                const Tp off_x = x - grid_x;    // fractional offset
                const int idx = ioff + grid_x;
                Ty val[2] = {in.ptr[idx], x + 1 < in.dims[0] ? in.ptr[idx + 1] : scalar<Ty>(0)};
                return linearInterpFunc(val, off_x);
            }
        };

        template<typename Ty, typename Tp>
        struct Interp1<Ty, Tp, AF_INTERP_CUBIC>
        {
            __device__ Ty operator()(CParam<Ty> in, int ioff, Tp x)
            {
                const int grid_x = floor(x);    // nearest grid
                const Tp off_x = x - grid_x;    // fractional offset
                const int idx = ioff + grid_x;

                bool cond[4] = {grid_x - 1 >= 0, true, grid_x + 1 < in.dims[0], grid_x + 2 < in.dims[0]};
                int  off[4]  = {cond[0] ? -1 : 0, 0, cond[2] ? 1 : 0, cond[3] ? 2 : (cond[2] ? 1 : 0)};

                Ty val[4];
                for (int i = 0; i < 4; i++) {
                    val[i] = in.ptr[idx + off[i]];
                }
                return cubicInterpFunc(val, off_x);
            }
        };

        template<typename Ty, typename Tp, af_interp_type>
        struct Interp2
        {
            __device__ Ty operator()(CParam<Ty> in, int ioff, Tp x, Tp y)
            {
                const int idx = ioff + round(y) * in.strides[1] + round(x);
                return in.ptr[idx];
            }
        };

        template<typename Ty, typename Tp>
        struct Interp2<Ty, Tp, AF_INTERP_BILINEAR>
        {
            __device__ Ty operator()(CParam<Ty> in, int ioff, Tp x, Tp y)
            {
                const int grid_x = floor(x);
                const Tp off_x = x - grid_x;

                const int grid_y = floor(y);
                const Tp off_y = y - grid_y;

                const int idx = ioff + grid_y * in.strides[1] + grid_x;

                bool condX[2] = {true, x + 1 < in.dims[0]};
                bool condY[2] = {true, y + 1 < in.dims[1]};

                Ty val[2][2];
                for (int j = 0; j < 2; j++) {
                    int off_y = idx + j * in.strides[1];
                    for (int i = 0; i < 2; i++) {
                        val[j][i] = condX[i] && condY[j] ? in.ptr[off_y + i] : scalar<Ty>(0);
                    }
                }

                return bilinearInterpFunc(val, off_x, off_y);
            }
        };

        template<typename Ty, typename Tp>
        struct Interp2<Ty, Tp, AF_INTERP_LINEAR>
        {
            __device__ Ty operator()(CParam<Ty> in, int ioff, Tp x, Tp y)
            {
                return Interp2<Ty, Tp, AF_INTERP_BILINEAR>()(in, ioff, x, y);
            }
        };

        template<typename Ty, typename Tp>
        struct Interp2<Ty, Tp, AF_INTERP_BICUBIC>
        {
            __device__ Ty operator()(CParam<Ty> in, int ioff, Tp x, Tp y)
            {
                const int grid_x = floor(x);
                const Tp off_x = x - grid_x;

                const int grid_y = floor(y);
                const Tp off_y = y - grid_y;

                const int idx = ioff + grid_y * in.strides[1] + grid_x;

                //for bicubic interpolation, work with 4x4 val at a time
                Ty val[4][4];

                // used for setting values at boundaries
                bool condX[4] = {grid_x - 1 >= 0, true, grid_x + 1 < in.dims[0], grid_x + 2 < in.dims[0]};
                bool condY[4] = {grid_y - 1 >= 0, true, grid_y + 1 < in.dims[1], grid_y + 2 < in.dims[1]};
                int  offX[4]  = {condX[0] ? -1 : 0, 0, condX[2] ? 1 : 0 , condX[3] ? 2 : (condX[2] ? 1 : 0)};
                int  offY[4]  = {condY[0] ? -1 : 0, 0, condY[2] ? 1 : 0 , condY[3] ? 2 : (condY[2] ? 1 : 0)};

#pragma unroll
                for (int j = 0; j < 4; j++) {
                    int ioff_j = idx + offY[j] * in.strides[1];
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        val[j][i] = in.ptr[ioff_j + offX[i]];
                    }
                }

                return bicubicInterpFunc(val, off_x, off_y);
            }
        };

        template<typename Ty, typename Tp>
        struct Interp2<Ty, Tp, AF_INTERP_CUBIC>
        {
            __device__ Ty operator()(CParam<Ty> in, int ioff, Tp x, Tp y)
            {
                return Interp2<Ty, Tp, AF_INTERP_BICUBIC>()(in, ioff, x, y);
            }
        };
    }
}
