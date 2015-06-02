/*******************************************************
 * Copyright (c) 2014, ArrayFire
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
        template<typename T>
        struct itype_t
        {
            typedef float wtype;
            typedef float vtype;
        };

        template<>
        struct itype_t<double>
        {
            typedef double wtype;
            typedef double vtype;
        };

        template<>
        struct itype_t<cfloat>
        {
            typedef float  wtype;
            typedef cfloat vtype;
        };

        template<>
        struct itype_t<cdouble>
        {
            typedef double  wtype;
            typedef cdouble vtype;
        };

        template<typename T>
        __device__
        void transform_n(T *optr, Param<T> out, const T *iptr, CParam<T> in, const float *tmat,
                         const int xido, const int yido, const int nimages)
        {
            // Compute input index
            int xidi = round(xido * tmat[0]
                             + yido * tmat[1]
                             + tmat[2]);
            int yidi = round(xido * tmat[3]
                             + yido * tmat[4]
                             + tmat[5]);

            // Makes scale give same output as resize
            // But fails rotate tests
            //if (xidi >= in.dims[0]) { xidi = in.dims[0] - 1; }
            //if (yidi >= in.dims[1]) { yidi = in.dims[1] - 1; }

            const int loci = yidi * in.strides[1]  + xidi;
            const int loco = yido * out.strides[1] + xido;

            for(int i = 0; i < nimages; i++) {
                // Compute memory location of indices
                int ioff = loci + i * in.strides[2];
                int ooff = loco + i * out.strides[2];

                // Copy to output
                T val = scalar<T>(0);
                if (xidi < in.dims[0] && yidi < in.dims[1] && xidi >= 0 && yidi >= 0) val = iptr[ioff];

                optr[ooff] = val;
            }
        }

        template<typename T>
        __device__
        void transform_b(T *optr, Param<T> out, const T *iptr, CParam<T> in, const float *tmat,
                         const int xido, const int yido, const int nimages)
        {
            const int loco = (yido * out.strides[1] + xido);

            // Compute input index
            const float xidi = xido * tmat[0]
                             + yido * tmat[1]
                                    + tmat[2];
            const float yidi = xido * tmat[3]
                             + yido * tmat[4]
                                    + tmat[5];

            if (xidi < -0.0001 || yidi < -0.0001 || in.dims[0] < xidi || in.dims[1] < yidi) {
                for(int i = 0; i < nimages; i++) {
                    optr[loco + i * out.strides[2]] = scalar<T>(0.0f);
                }
                return;
            }

            typedef typename itype_t<T>::wtype WT;
            typedef typename itype_t<T>::vtype VT;

            const WT grd_x = floor(xidi),  grd_y = floor(yidi);
            const WT off_x = xidi - grd_x, off_y = yidi - grd_y;

            // Check if pVal and pVal + 1 are both valid indices
            const bool condY = (yidi < in.dims[1] - 1);
            const bool condX = (xidi < in.dims[0] - 1);

            // Compute weights used
            const WT wt00 = (1.0 - off_x) * (1.0 - off_y);
            const WT wt10 = (condY) ? (1.0 - off_x) * (off_y)     : 0;
            const WT wt01 = (condX) ? (off_x) * (1.0 - off_y)     : 0;
            const WT wt11 = (condX && condY) ? (off_x) * (off_y)  : 0;

            const WT wt = wt00 + wt10 + wt01 + wt11;

            const int loci = grd_y * in.strides[1] + grd_x;
            T zero = scalar<T>(0.0f);
            for(int i = 0; i < nimages; i++) {
                const int ioff = loci + (i * in.strides[2]);
                const int ooff = loco + (i * out.strides[2]);

                // Compute Weighted Values
                VT v00 =                    wt00 * iptr[ioff];
                VT v10 = (condY) ?          wt10 * iptr[ioff + in.strides[1]]     : zero;
                VT v01 = (condX) ?          wt01 * iptr[ioff + 1]                 : zero;
                VT v11 = (condX && condY) ? wt11 * iptr[ioff + in.strides[1] + 1] : zero;
                VT vo  = v00 + v10 + v01 + v11;

                optr[ooff] = (vo / wt);
            }
        }
    }
}
