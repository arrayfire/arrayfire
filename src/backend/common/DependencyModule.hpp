/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/Logger.hpp>
#include <common/defines.hpp>
#include <common/module_loading.hpp>

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace spdlog {
class logger;
}
namespace arrayfire {
namespace common {

using Version = std::tuple<int, int, int>;  // major, minor, patch

/// Allows you to create classes which dynamically load dependencies at runtime
///
/// Creates a dependency module which will dynamically load a library
/// at runtime instead of at link time. This class will be a component of a
/// module class which will have member functions for each of the functions
/// we use in ArrayFire
class DependencyModule {
    LibHandle handle;
    std::shared_ptr<spdlog::logger> logger;
    std::vector<void*> functions;

   public:
    /// Loads the library \p plugin_file_name from the \p paths locations
    /// \param plugin_file_name  The name of the library without any prefix or
    ///                          extensions
    /// \param paths             The locations to search for the libraries if
    ///                          not found in standard locations
    DependencyModule(const char* plugin_file_name,
                     const char** paths = nullptr);

    DependencyModule(const std::vector<std::string>& plugin_base_file_name,
                     const std::vector<std::string>& suffixes,
                     const std::vector<std::string>& paths,
                     const size_t verListSize = 0,
                     const Version* versions  = nullptr);

    ~DependencyModule() noexcept;

    /// Returns a function pointer to the function with the name symbol_name
    template<typename T>
    T getSymbol(const char* symbol_name) {
        functions.push_back(getFunctionPointer(handle, symbol_name));
        return (T)functions.back();
    }

    /// Returns true if the module was successfully loaded
    bool isLoaded() const noexcept;

    /// Returns true if all of the symbols for the module were loaded
    bool symbolsLoaded() const noexcept;

    /// Returns the last error message that occurred because of loading the
    /// library
    static std::string getErrorMessage() noexcept;

    spdlog::logger* getLogger() const noexcept;
};

}  // namespace common
}  // namespace arrayfire

/// Creates a function pointer
#define MODULE_MEMBER(NAME) decltype(&::NAME) NAME

/// Dynamically loads the function pointer at runtime
#define MODULE_FUNCTION_INIT(NAME) \
    NAME = module.getSymbol<decltype(&::NAME)>(#NAME);
