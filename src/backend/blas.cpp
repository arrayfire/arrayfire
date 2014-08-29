#include <af/blas.h>
#include <blas.hpp>
#include <handle.hpp>
#include <Array.hpp>
#include <af/array.h>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <helper.hpp>
#include <backend.hpp>

template<typename T>
static inline af_array matmul(const af_array lhs, const af_array rhs,
                    af_blas_transpose optLhs, af_blas_transpose optRhs)
{
    return getHandle(*detail::matmul<T>(getArray<T>(lhs), getArray<T>(rhs), optLhs, optRhs));
}

template<typename T>
static inline af_array dot(const af_array lhs, const af_array rhs,
                    af_blas_transpose optLhs, af_blas_transpose optRhs)
{
    return getHandle(*detail::dot<T>(getArray<T>(lhs), getArray<T>(rhs), optLhs, optRhs));
}

af_err af_matmul(   af_array *out,
                    const af_array lhs, const af_array rhs,
                    af_blas_transpose optLhs, af_blas_transpose optRhs)
{
    using namespace detail;
    af_err ret = AF_ERR_RUNTIME;

    try {
        ArrayInfo lhsInfo = getInfo(lhs);
        ArrayInfo rhsInfo = getInfo(rhs);

        if(lhsInfo.getType() == rhsInfo.getType()) {
            af_array output = 0;

            switch(lhsInfo.getType()) {
                case f32: output = matmul<float  >(lhs, rhs, optLhs, optRhs);   break;
                case c32: output = matmul<cfloat >(lhs, rhs, optLhs, optRhs);   break;
                case f64: output = matmul<double >(lhs, rhs, optLhs, optRhs);   break;
                case c64: output = matmul<cdouble>(lhs, rhs, optLhs, optRhs);   break;
                default:  ret = AF_ERR_NOT_SUPPORTED;           break;
            }
            if (ret!=AF_ERR_NOT_SUPPORTED) {
                std::swap(*out, output);
                ret = AF_SUCCESS;
            }
        }
        else {
            ret = AF_ERR_DIFF_TYPE;
        }
    }
    CATCHALL
    return ret;
}

af_err af_dot(      af_array *out,
                    const af_array lhs, const af_array rhs,
                    af_blas_transpose optLhs, af_blas_transpose optRhs)
{
    using namespace detail;
    af_err ret = AF_ERR_RUNTIME;

    try {
        ArrayInfo lhsInfo = getInfo(lhs);
        ArrayInfo rhsInfo = getInfo(rhs);

        if(lhsInfo.getType() == rhsInfo.getType()) {
            af_array output = 0;

            switch(lhsInfo.getType()) {
                //case f32: output = dot<float  >(lhs, rhs, optLhs, optRhs);   break;
                //case c32: output = dot<cfloat >(lhs, rhs, optLhs, optRhs);   break;
                //case f64: output = dot<double >(lhs, rhs, optLhs, optRhs);   break;
                //case c64: output = dot<cdouble>(lhs, rhs, optLhs, optRhs);   break;
                default:  ret = AF_ERR_NOT_SUPPORTED;           break;
            }
            if (ret!=AF_ERR_NOT_SUPPORTED) {
                std::swap(*out, output);
                ret = AF_SUCCESS;
            }
        }
        else {
            ret = AF_ERR_DIFF_TYPE;
        }
    }
    CATCHALL
    return ret;
}
