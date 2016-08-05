/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/exception.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <arrayfire.h>
#include <af/traits.hpp>

#ifdef __cplusplus
extern "C" {
#endif

#if AF_API_VERSION >= 31
/**
   Get the stream for the CUDA device with \p id in ArrayFire context

   \param[out] stream CUDA Stream of device with \p id in ArrayFire context
   \param[in] id ArrayFire device id
   \returns \ref af_err error code

   \ingroup cuda_mat
 */
AFAPI af_err afcu_get_stream(cudaStream_t* stream, int id);
#endif

#if AF_API_VERSION >= 31
/**
   Get the native device id of the CUDA device with \p id in ArrayFire context

   \param[out] nativeid native device id of the CUDA device with \p id in ArrayFire context
   \param[in] id ArrayFire device id
   \returns \ref af_err error code

   \ingroup cuda_mat
 */
AFAPI af_err afcu_get_native_id(int* nativeid, int id);
#endif

#if AF_API_VERSION >= 32
/**
   Set the CUDA device with given native id as the active device for ArrayFire

   \param[in] nativeid native device id of the CUDA device
   \returns \ref af_err error code

   \ingroup cuda_mat
 */
AFAPI af_err afcu_set_native_id(int nativeid);
#endif

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

namespace afcu
{

#if AF_API_VERSION >= 31
/**
   Get the stream for the CUDA device with \p id in ArrayFire context

   \param[in] id ArrayFire device id
   \returns cuda stream used by CUDA device

   \ingroup cuda_mat
 */
static inline cudaStream_t getStream(int id)
{
    cudaStream_t retVal;
    af_err err = afcu_get_stream(&retVal, id);
    if (err!=AF_SUCCESS)
        throw af::exception("Failed to get CUDA stream from ArrayFire");
    return retVal;
}
#endif

#if AF_API_VERSION >= 31
/**
   Get the native device id of the CUDA device with \p id in ArrayFire context

   \param[in] id ArrayFire device id
   \returns cuda native id of device

   \ingroup cuda_mat
 */
static inline int getNativeId(int id)
{
    int retVal;
    af_err err = afcu_get_native_id(&retVal, id);
    if (err!=AF_SUCCESS)
        throw af::exception("Failed to get CUDA device native id from ArrayFire");
    return retVal;
}
#endif

#if AF_API_VERSION >= 32
/**
   Set the CUDA device with given native id as the active device for ArrayFire

   \param[in] nativeId native device id of the CUDA device

   \ingroup cuda_mat
 */
static inline void setNativeId(int nativeId)
{
    af_err err = afcu_set_native_id(nativeId);
    if (err!=AF_SUCCESS)
        throw af::exception("Failed to change active CUDA device to the device with given native id");
}
#endif

#if AF_API_VERSION >= 34
/**
   Interop function that converts a thrust::device_vector to an af::array

   \param[in] tvec thrust::device_vector to be converted
   \return af::array with converted contents of the thrust vector

   \ingroup cuda_mat
 */
template<typename T>
static af::array array(thrust::device_vector<T> &tvec)
{
    af::array output;
    if(tvec.empty()) return output;
    T* ptr = thrust::raw_pointer_cast(tvec.data());
    af::array tmp = af::array(tvec.size(), ptr, afDevice);
    output = tmp.copy();
    tmp.lock();
    return output;
}
#endif

#if AF_API_VERSION >= 34
/**
   Interop function that converts a thrust::host_vector to an af::array

   \param[in] tvec thrust::host_vector to  be converted
   \return af::array with converted contents of the thrust vector

   \ingroup cuda_mat
 */
template<typename T>
static af::array array(thrust::host_vector<T> &tvec)
{
    af::array output;
    if(tvec.empty()) return output;
    //af_dtype type = (af_dtype)dtype_traits<T>::af_type;
    T* ptr = thrust::raw_pointer_cast(tvec.data());
    output = af::array(tvec.size(), ptr);
    return output;
}
#endif

#if AF_API_VERSION >= 34
/**
   Interop function that converts an af::array to a thrust::host_vector

   \param[in] arr af::array to be converted to thrust::host_vector
   \return thrust::host_vector with converted contents of af::array

   \ingroup cuda_mat
 */
template<typename T>
static thrust::host_vector<T> toHostVector(af::array &arr)
{
    af_dtype vec_type = (af_dtype)af::dtype_traits<T>::af_type;
    if(arr.type() != vec_type) { throw af::exception("Thrust vector data type and array data type are mismatching"); }
    //todo check continuity
    thrust::host_vector<T> hvec(arr.elements());
    T* h_ptr = thrust::raw_pointer_cast(hvec.data());
    arr.host(h_ptr);
    return hvec;
}
#endif

#if AF_API_VERSION >= 34
/**
   Interop function that converts an af::array to a thrust::device_vector

   \param[in] arr af::array to be converted to thrust::device_vector
   \return thrust::device_vector with converted contents of af::array

   \ingroup cuda_mat
 */
template<typename T>
static thrust::device_vector<T> toDeviceVector(af::array &arr)
{
    af_dtype vec_type = (af_dtype)af::dtype_traits<T>::af_type;
    if(arr.type() != vec_type) { throw af::exception("Thrust vector data type and array data type are mismatching"); }
    arr.lock();
    //todo check continuity
    thrust::device_ptr<T> af_ptr = thrust::device_pointer_cast(arr.device<T>());
    thrust::device_vector<T> dvec(af_ptr, af_ptr + arr.elements());
    arr.unlock();
    return dvec;
}
#endif

}
#endif
