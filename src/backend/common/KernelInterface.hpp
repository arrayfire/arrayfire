/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cstddef>
#include <string>

namespace arrayfire {
namespace common {

/// Kernel Interface that should be implemented by each backend
template<typename ModuleType, typename KernelType, typename EnqueuerType,
         typename DevPtrType>
class KernelInterface {
   private:
    ModuleType mModuleHandle;
    KernelType mKernelHandle;
    std::string mName;

   public:
    KernelInterface(std::string name, ModuleType mod, KernelType ker)
        : mModuleHandle(mod), mKernelHandle(ker), mName(name) {}

    /// \brief Set kernel
    ///
    /// \param[in] ker is backend specific kernel handle
    inline void set(KernelType ker) { mKernelHandle = ker; }

    /// \brief Get kernel
    ///
    /// \returns handle to backend specific kernel
    inline KernelType get() const { return mKernelHandle; }

    /// \brief Get module
    ///
    /// \returns handle to backend specific module
    inline ModuleType getModuleHandle() { return mModuleHandle; }

    /// \brief Get device pointer associated with name(label)
    ///
    /// This function is only useful with CUDA NVRTC based compilation
    /// at the moment, calling this function for OpenCL backend build
    /// will return a null pointer.
    virtual DevPtrType getDevPtr(const char* name) = 0;

    /// \brief Copy data from device memory to read-only memory
    ///
    /// This function copies data of `bytes` size from the device pointer to a
    /// read-only memory.
    ///
    /// \param[in] dst is the device pointer to which data will be copied
    /// \param[in] src is the device pointer from which data will be copied
    /// \param[in] bytes are the number of bytes of data to be copied
    virtual void copyToReadOnly(DevPtrType dst, DevPtrType src,
                                size_t bytes) = 0;

    /// \brief Copy a single scalar to device memory
    ///
    /// This function copies a single value of type T from host variable
    /// to the device memory pointed by `dst`
    ///
    /// \param[in] dst is the device pointer to which data will be copied
    /// \param[in] value is a poiner to the scalar value that is set at device
    ///            pointer
    /// \param[in] syncCopy will indicate if the backend call to upload the
    ///            scalar value to GPU memory has to wait for copy to finish
    ///            or proceed ahead without wait
    virtual void setFlag(DevPtrType dst, int* scalarValPtr,
                         const bool syncCopy = false) = 0;

    /// \brief Fetch a scalar from device memory
    ///
    /// This function copies a single value of type T from device memory
    ///
    /// \param[in] src is the device pointer from which data will be copied
    ///
    /// \returns the integer scalar
    virtual int getFlag(DevPtrType src) = 0;

    /// \brief Enqueue Kernel per queueing criteria forwarding other parameters
    ///
    /// This operator overload enables Kernel object to work as functor that
    /// internally executes the kernel stored in the Kernel object.
    /// All parameters that are passed in after the EnqueueArgs object are
    /// essentially forwarded to kenel launch API
    ///
    /// \param[in] qArgs is an object of type EnqueueArgsType like
    //             cl::EnqueueArgs in OpenCL backend
    /// \param[in] args is the placeholder for variadic arguments
    template<typename EnqueueArgsType, typename... Args>
    void operator()(const EnqueueArgsType& qArgs, Args... args) {
        EnqueuerType launch;
        launch(mName, mKernelHandle, qArgs, std::forward<Args>(args)...);
    }
};

}  // namespace common
}  // namespace arrayfire
