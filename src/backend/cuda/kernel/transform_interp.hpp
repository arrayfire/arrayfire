namespace cuda
{
    namespace kernel
    {
        template<typename T>
        __device__
        void transform_n(T *optr, Param<T> out, const T *iptr, CParam<T> in, const float *tmat,
                         const dim_type xido, const dim_type yido, const dim_type nimages)
        {
            // Compute input index
            dim_type xidi = round(xido * tmat[0]
                                + yido * tmat[1]
                                       + tmat[2]);
            dim_type yidi = round(xido * tmat[3]
                                + yido * tmat[4]
                                       + tmat[5]);

            // Makes scale give same output as resize
            // But fails rotate tests
            //if (xidi >= in.dims[0]) { xidi = in.dims[0] - 1; }
            //if (yidi >= in.dims[1]) { yidi = in.dims[1] - 1; }

            const dim_type loci = yidi * in.strides[1]  + xidi;
            const dim_type loco = yido * out.strides[1] + xido;

            for(int i = 0; i < nimages; i++) {
                // Compute memory location of indices
                dim_type ioff = loci + i * in.strides[2];
                dim_type ooff = loco + i * out.strides[2];

                // Copy to output
                T val = scalar<T>(0);
                if (xidi < in.dims[0] && yidi < in.dims[1] && xidi >= 0 && yidi >= 0) val = iptr[ioff];

                optr[ooff] = val;
            }
        }

        template<typename T>
        __device__
        void transform_b(T *optr, Param<T> out, const T *iptr, CParam<T> in, const float *tmat,
                         const dim_type xido, const dim_type yido, const dim_type nimages)
        {
            const dim_type loco = (yido * out.strides[1] + xido);

            // Compute input index
            const float xidi = xido * tmat[0]
                             + yido * tmat[1]
                                    + tmat[2];
            const float yidi = xido * tmat[3]
                             + yido * tmat[4]
                                    + tmat[5];

            if (xidi < 0 || yidi < 0 || in.dims[0] < xidi || in.dims[1] < yidi) {
                for(int i = 0; i < nimages; i++) {
                    optr[loco + i * out.strides[2]] = scalar<T>(0.0f);
                }
                return;
            }

            const float grd_x = floor(xidi),  grd_y = floor(yidi);
            const float off_x = xidi - grd_x, off_y = yidi - grd_y;

            // Check if pVal and pVal + 1 are both valid indices
            const bool condY = (yidi < in.dims[1] - 1);
            const bool condX = (xidi < in.dims[0] - 1);

            // Compute weights used
            const float wt00 = (1.0 - off_x) * (1.0 - off_y);
            const float wt10 = (condY) ? (1.0 - off_x) * (off_y)     : 0;
            const float wt01 = (condX) ? (off_x) * (1.0 - off_y)     : 0;
            const float wt11 = (condX && condY) ? (off_x) * (off_y)  : 0;

            const float wt = wt00 + wt10 + wt01 + wt11;

            const dim_type loci = grd_y * in.strides[1] + grd_x;
            T zero = scalar<T>(0.0f);
            for(int i = 0; i < nimages; i++) {
                const dim_type ioff = loci + (i * in.strides[2]);
                const dim_type ooff = loco + (i * out.strides[2]);

                // Compute Weighted Values
                T v00 =                    wt00 * iptr[ioff];
                T v10 = (condY) ?          wt10 * iptr[ioff + in.strides[1]]     : zero;
                T v01 = (condX) ?          wt01 * iptr[ioff + 1]                 : zero;
                T v11 = (condX && condY) ? wt11 * iptr[ioff + in.strides[1] + 1] : zero;
                T vo = v00 + v10 + v01 + v11;

                optr[ooff] = (vo / wt);
            }
        }
    }
}
