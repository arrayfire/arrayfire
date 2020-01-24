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
#include <utility>

namespace common {

/// Kernel Interface that should be implemented by each backend
template<typename ModuleType, typename KernelType, typename EnqueuerType,
         typename DevPtrType>
class KernelInterface {
   private:
    ModuleType mProgram;
    KernelType mKernel;

   public:
    KernelInterface(ModuleType mod, KernelType ker)
        : mProgram(mod), mKernel(ker) {}

    /// \brief Set module and kernel
    ///
    /// \param[in] mod is backend specific module handle
    /// \param[in] ker is backend specific kernel handle
    void set(ModuleType mod, KernelType ker) {
        mProgram = mod;
        mKernel  = ker;
    }

    /// \brief Get module
    ///
    /// \returns handle to backend specific module
    inline ModuleType getModule() { return mProgram; }

    /// \brief Get kernel
    ///
    /// \returns handle to backend specific kernel
    inline KernelType getKernel() { return mKernel; }

    /// \brief Get device pointer associated with name(label)
    ///
    /// This function is only useful with CUDA NVRTC based compilation
    /// at the moment, calling this function for OpenCL backend build
    /// will return a null pointer.
    virtual DevPtrType get(const char* name) = 0;

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
    /// \param[in] value is the integer scalar to set at device pointer
    virtual void setScalar(DevPtrType dst, int value) = 0;

    /// \brief Fetch a scalar from device memory
    ///
    /// This function copies a single value of type T from device memory
    ///
    /// \param[in] src is the device pointer from which data will be copied
    ///
    /// \returns the integer scalar
    virtual int getScalar(DevPtrType src) = 0;

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
        launch(mKernel, qArgs, std::forward<Args>(args)...);
    }
};

}  // namespace common
