/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <lu.hpp>
#include <err_common.hpp>

#if defined(WITH_LINEAR_ALGEBRA)

//#include <af/dim4.hpp>
//#include <handle.hpp>
//#include <iostream>
//#include <cassert>
//#include <err_cpu.hpp>
//#include <err_common.hpp>
//
//#if defined(LAPACK_All)
//    #include <atlas/clapack.h>
//    #define LAPACK_NAME(fn) clapack_##fn
//    #define ORDER CblasColMajor
//    #define ORDER_TYPE CBLAS_ORDER
//#elif defined(LAPACK_Intel)
//    #include <lapack.h>
//    #define LAPACK_PREFIX LAPACKE
//    #define LAPACK_SUFFIX
//    #define ORDER LAPACK_COL_MAJOR
//    #define ORDER_TYPE int
//#else
//    #warning "HERE NOT IN ALL"
//#endif
//
//namespace cpu
//{
//using std::is_floating_point;
//using std::conditional;
//
//template<typename T, typename BT>
//using ptr_type = typename conditional<is_complex<T>::value,
//                                      BT *,
//                                      T*>::type;
//template<typename T, typename BT>
//using getrf_func_def = int (*)(ORDER_TYPE, int, int,
//                               ptr_type<T, BT>, const int,
//                               ptr_type<int, int>);
//
//#define LU_FUNC_DEF( FUNC )                                                      \
//template<typename T, typename BT> FUNC##_func_def<T, BT> FUNC##_func();
//
//
//#define LU_FUNC( FUNC, TYPE, BASE_TYPE, PREFIX )                                 \
//template<> FUNC##_func_def<TYPE, BASE_TYPE>     FUNC##_func<TYPE, BASE_TYPE>()     \
//{ return & LAPACK_NAME(PREFIX##FUNC); }
//
//LU_FUNC_DEF( getrf )
//LU_FUNC(getrf , float   , float  , s)
//LU_FUNC(getrf , double  , double , d)
//LU_FUNC(getrf , cfloat  , void   , c)
//LU_FUNC(getrf , cdouble , void   , z)
//
//#ifdef OS_WIN
//#define BT af::dtype_traits<T>::base_type
//#else
//template<typename T> struct lapack_types;
//
//template<>
//struct lapack_types<float> {
//    typedef float base_type;
//};
//
//template<>
//struct lapack_types<cfloat> {
//    typedef void base_type;
//};
//
//template<>
//struct lapack_types<double> {
//    typedef double base_type;
//};
//
//template<>
//struct lapack_types<cdouble> {
//    typedef void base_type;
//};
//#define BT typename lapack_types<T>::base_type
//#endif
//
//template<typename T>
//Array<int> lu_inplace(Array<T> &in)
//{
//    dim4 iDims = in.dims();
//    int M = iDims[0];
//    int N = iDims[1];
//
//    //FIXME: Leaks on errors.
//    Array<int> pivot = createEmptyArray<int>(af::dim4(min(M, N), 1, 1, 1));
//
//    getrf_func<T, BT>()(ORDER, M, N,
//                        in.get(), M, pivot.get());
//
//    return pivot;
//}
//
//#define INSTANTIATE_LU(TYPE)                                                          \
//    template Array<int> lu_inplace<TYPE>(Array<TYPE> &in);
//
//INSTANTIATE_LU(float)
//INSTANTIATE_LU(cfloat)
//INSTANTIATE_LU(double)
//INSTANTIATE_LU(cdouble)
//}

namespace cpu
{

template<typename T>
void lu(Array<T> &lower, Array<T> &upper, Array<int> &pivot, const Array<T> &in)
{
    AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<int> lu_inplace(Array<T> &in);
{
    AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_LU(T)                                               \
    template Array<int> lu_inplace<T>(Array<T> &in);                 \
    template void lu<T>(Array<T> &lower, Array<T> &upper, Array<int> &pivot, const Array<T> &in);

INSTANTIATE_LU(float)
INSTANTIATE_LU(cfloat)
INSTANTIATE_LU(double)
INSTANTIATE_LU(cdouble)

}

#else

namespace cpu
{

template<typename T>
void lu(Array<T> &lower, Array<T> &upper, Array<int> &pivot, const Array<T> &in)
{
    AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<int> lu_inplace(Array<T> &in)
{
    AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_LU(T)                                               \
    template Array<int> lu_inplace<T>(Array<T> &in);                 \
    template void lu<T>(Array<T> &lower, Array<T> &upper, Array<int> &pivot, const Array<T> &in);

INSTANTIATE_LU(float)
INSTANTIATE_LU(cfloat)
INSTANTIATE_LU(double)
INSTANTIATE_LU(cdouble)

}

#endif
