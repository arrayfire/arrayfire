/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <common/defines.hpp>

#include <utility>
#include <type_traits>
#include <vector>
#include <string>

namespace common {

/// Allows you to create classes which dynamically load dependencies at runtime
///
/// Creates a dependency module which will dynamically load a library
/// at runtime instead of at link time. This class will be a component of a
/// module class which will have member functions for each of the functions
/// we use in ArrayFire
class DependencyModule {
  LibHandle handle;
  std::vector<void*> functions;
  void* getFunctionPointer(LibHandle handle, const char* name);

public:
  DependencyModule(const char* plugin_file_name, const char** paths = nullptr);

  ~DependencyModule();

  /// Returns a function pointer to the function with the name symbol_name
  template<typename T>
  T getSymbol(const char* symbol_name) {
      functions.push_back(getFunctionPointer(handle, symbol_name));
      return (T)functions.back();
  }

  /// Returns true if the module was successfully loaded
  bool isLoaded();

  /// Returns true if the module was successfully loaded
  bool symbolsLoaded();

  /// Returns the last error message that occurred because of loading the
  /// library
  std::string getErrorMessage();
};

}

/// Creates a function pointer
#define MODULE_MEMBER(NAME)                     \
  decltype(&::NAME) NAME

/// Dynamically loads the function pointer at runtime
#define MODULE_FUNCTION_INIT(NAME)                \
  NAME = module.getSymbol<decltype(&::NAME)>(#NAME)
