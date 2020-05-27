/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

namespace common {

/// Instances of this object are stored in jit kernel cache
template<typename ModuleType>
class ModuleInterface {
   private:
    ModuleType mModuleHandle;

   public:
    ModuleInterface(ModuleType mod) : mModuleHandle(mod) {}

    /// \brief Set module
    ///
    /// \param[in] mod is backend specific module handle
    inline void set(ModuleType mod) { mModuleHandle = mod; }

    /// \brief Get module
    ///
    /// \returns handle to backend specific module
    inline ModuleType get() const { return mModuleHandle; }

    /// \brief Unload module
    virtual void unload() = 0;
};

}  // namespace common
