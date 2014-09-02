#include <af/array.h>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <transform.hpp>
#include <stdexcept>

namespace cpu
{
    template <typename T>
    void calc_affine_inverse(T *txo, const T *txi)
    {
        T det = txi[0]*txi[4] - txi[1]*txi[3];

        txo[0] = txi[4] / det;
        txo[1] = txi[3] / det;
        txo[3] = txi[1] / det;
        txo[4] = txi[0] / det;

        txo[2] = txi[2] * -txo[0] + txi[5] * -txo[1];
        txo[5] = txi[2] * -txo[3] + txi[5] * -txo[4];
    }

    template <typename T>
    void calc_affine_inverse(T *tmat, const T *tmat_ptr, const bool inverse)
    {
        // The way kernel is structured, it expects an inverse
        // transform matrix by default.
        // If it is an forward transform, then we need its inverse
        if(inverse) {
            for(int i = 0; i < 6; i++)
                tmat[i] = tmat_ptr[i];
        } else {
            calc_affine_inverse(tmat, tmat_ptr);
        }
    }

    template<typename T>
    void transform_op(T *out, const T *in, const float *tmat, const af::dim4 &idims,
                      const af::dim4 &ostrides, const af::dim4 &istrides,
                      const dim_type nimages, const dim_type o_offset,
                      const dim_type xx, const dim_type yy)
    {
        // Compute output index
        const dim_type xi = round(xx * tmat[0]
                                + yy * tmat[1]
                                     + tmat[2]);
        const dim_type yi = round(xx * tmat[3]
                                + yy * tmat[4]
                                     + tmat[5]);

        // Compute memory location of indices
        dim_type loci = (yi * istrides[1] + xi);
        dim_type loco = (yy * ostrides[1] + xx);

        // Copy to output
        for(int i_idx = 0; i_idx < nimages; i_idx++) {
            T val = 0;
            dim_type i_off = i_idx * istrides[2];
            dim_type o_off = o_offset + i_idx * ostrides[2];

            if (xi < idims[0] && yi < idims[1] && xi >= 0 && yi >= 0)
                val = in[i_off + loci];

            out[o_off + loco] = val;
        }
    }

    template<typename T>
    void transform_(T *out, const T *in, const float *tf,
                    const af::dim4 &odims, const af::dim4 &idims,
                    const af::dim4 &ostrides, const af::dim4 &istrides,
                    const af::dim4 &tstrides, const bool inverse)
    {
        dim_type nimages     = idims[2];
        // Multiplied in src/backend/transform.cpp
        dim_type ntransforms = odims[2] / idims[2];

        // For each transform channel
        for(int t_idx = 0; t_idx < ntransforms; t_idx++) {
            // Compute inverse if required
            const float *tmat_ptr = tf + t_idx * 6;
            float tmat[6];
            calc_affine_inverse(tmat, tmat_ptr, inverse);

            // Offset for output pointer
            dim_type o_offset = t_idx * nimages * ostrides[2];

            // Do transform for image
            for(int yy = 0; yy < odims[1]; yy++) {
                for(int xx = 0; xx < odims[0]; xx++) {
                    transform_op(out, in, tmat, idims, ostrides, istrides,
                                 nimages, o_offset, xx, yy);
                }
            }
        }
    }

    template<typename T>
    Array<T>* transform(const Array<T> &in, const Array<float> &transform, const af::dim4 &odims,
                        const bool inverse)
    {
        const af::dim4 idims = in.dims();

        Array<T> *out = createEmptyArray<T>(odims);

        transform_<T>(out->get(), in.get(), transform.get(), odims, idims,
                      out->strides(), in.strides(), transform.strides(), inverse);

        return out;
    }


#define INSTANTIATE(T)                                                                         \
    template Array<T>* transform(const Array<T> &in, const Array<float> &transform,            \
                                 const af::dim4 &odims, const bool inverse);                   \


    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
}

