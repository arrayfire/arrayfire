/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

namespace arrayfire {
namespace common {

/// Instances of this object are stored in jit kernel cache
template<typename ModuleType>
class ModuleInterface {
   private:
    ModuleType mModuleHandle;

   public:
    /// \brief Creates an uninitialized Module
    ModuleInterface() = default;

    /// \brief Creates a module given a backend specific ModuleType
    ///
    /// \param[in] mod The backend specific module
    ModuleInterface(ModuleType mod) : mModuleHandle(mod) {}

    /// \brief Set module
    ///
    /// \param[in] mod is backend specific module handle
    inline void set(ModuleType mod) { mModuleHandle = mod; }

    /// \brief Get module
    ///
    /// \returns handle to backend specific module
    inline const ModuleType& get() const { return mModuleHandle; }

    /// \brief Unload module
    virtual void unload() = 0;

    /// \brief Returns true if the module mModuleHandle is initialized
    virtual operator bool() const = 0;
};

}  // namespace common
}  // namespace arrayfire
