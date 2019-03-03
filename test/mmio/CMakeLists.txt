# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)

project(MatrixMarketIO LANGUAGES C)

add_library(mmio STATIC mmio.c)

target_include_directories(mmio
    PUBLIC
      $<BUILD_INTERFACE:${MatrixMarketIO_SOURCE_DIR}>
    )

target_compile_definitions(mmio PUBLIC USE_MTX)

if(WIN32)
  target_compile_definitions(mmio PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
