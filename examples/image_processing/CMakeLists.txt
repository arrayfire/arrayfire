# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(ArrayFire-Example-Image-Processing
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)

add_definitions("-DASSETS_DIR=\"${ASSETS_DIR}\"")

if(ArrayFire_CPU_FOUND)
  # Adaptive Thresholding example
  add_executable(adaptive_thresholding_cpu adaptive_thresholding.cpp)
  target_link_libraries(adaptive_thresholding_cpu ArrayFire::afcpu)

  # Binary Thresholding example
  add_executable(binary_thresholding_cpu binary_thresholding.cpp)
  target_link_libraries(binary_thresholding_cpu ArrayFire::afcpu)

  # Brain Segmentation example
  add_executable(brain_segmentation_cpu brain_segmentation.cpp)
  target_link_libraries(brain_segmentation_cpu ArrayFire::afcpu)

  # Confidence Connected Components example
  add_executable(confidence_connected_components_cpu
      confidence_connected_components.cpp)
  target_link_libraries(confidence_connected_components_cpu ArrayFire::afcpu)

  # Edge detection example
  add_executable(edge_cpu edge.cpp)
  target_link_libraries(edge_cpu ArrayFire::afcpu)

  # Filters example
  add_executable(filters_cpu filters.cpp)
  target_link_libraries(filters_cpu ArrayFire::afcpu)

  # Image example
  add_executable(image_demo_cpu image_demo.cpp)
  target_link_libraries(image_demo_cpu ArrayFire::afcpu)

  # Image Editing example
  add_executable(image_editing_cpu image_editing.cpp)
  target_link_libraries(image_editing_cpu ArrayFire::afcpu)

  # Morph example
  add_executable(morphing_cpu morphing.cpp)
  target_link_libraries(morphing_cpu ArrayFire::afcpu)

  # Optical Flow example
  add_executable(optical_flow_cpu optical_flow.cpp)
  target_link_libraries(optical_flow_cpu ArrayFire::afcpu)

  # Pyramids example
  add_executable(pyramids_cpu pyramids.cpp)
  target_link_libraries(pyramids_cpu ArrayFire::afcpu)

  # Gradient anisotropic diffusion example
  add_executable(gradient_diffusion_cpu gradient_diffusion.cpp)
  target_link_libraries(gradient_diffusion_cpu ArrayFire::afcpu)

  #Image Deconvolution Example
  add_executable(deconvolution_cpu deconvolution.cpp)
  target_link_libraries(deconvolution_cpu ArrayFire::afcpu)
endif()

if(ArrayFire_CUDA_FOUND)
  add_executable(adaptive_thresholding_cuda adaptive_thresholding.cpp)
  target_link_libraries(adaptive_thresholding_cuda ArrayFire::afcuda)

  add_executable(binary_thresholding_cuda binary_thresholding.cpp)
  target_link_libraries(binary_thresholding_cuda ArrayFire::afcuda)

  add_executable(brain_segmentation_cuda brain_segmentation.cpp)
  target_link_libraries(brain_segmentation_cuda ArrayFire::afcuda)

  add_executable(confidence_connected_components_cuda
      confidence_connected_components.cpp)
  target_link_libraries(confidence_connected_components_cuda ArrayFire::afcuda)

  add_executable(edge_cuda edge.cpp)
  target_link_libraries(edge_cuda ArrayFire::afcuda)

  add_executable(filters_cuda filters.cpp)
  target_link_libraries(filters_cuda ArrayFire::afcuda)

  add_executable(image_demo_cuda image_demo.cpp)
  target_link_libraries(image_demo_cuda ArrayFire::afcuda)

  add_executable(image_editing_cuda image_editing.cpp)
  target_link_libraries(image_editing_cuda ArrayFire::afcuda)

  add_executable(morphing_cuda morphing.cpp)
  target_link_libraries(morphing_cuda ArrayFire::afcuda)

  add_executable(optical_flow_cuda optical_flow.cpp)
  target_link_libraries(optical_flow_cuda ArrayFire::afcuda)

  add_executable(pyramids_cuda pyramids.cpp)
  target_link_libraries(pyramids_cuda ArrayFire::afcuda)

  # Gradient anisotropic diffusion example
  add_executable(gradient_diffusion_cuda gradient_diffusion.cpp)
  target_link_libraries(gradient_diffusion_cuda ArrayFire::afcuda)

  #Image Deconvolution Example
  add_executable(deconvolution_cuda deconvolution.cpp)
  target_link_libraries(deconvolution_cuda ArrayFire::afcuda)
endif()

if(ArrayFire_OpenCL_FOUND)
  add_executable(adaptive_thresholding_opencl adaptive_thresholding.cpp)
  target_link_libraries(adaptive_thresholding_opencl ArrayFire::afopencl)

  add_executable(binary_thresholding_opencl binary_thresholding.cpp)
  target_link_libraries(binary_thresholding_opencl ArrayFire::afopencl)

  add_executable(brain_segmentation_opencl brain_segmentation.cpp)
  target_link_libraries(brain_segmentation_opencl ArrayFire::afopencl)

  add_executable(confidence_connected_components_opencl
      confidence_connected_components.cpp)
  target_link_libraries(confidence_connected_components_opencl ArrayFire::afopencl)

  add_executable(edge_opencl edge.cpp)
  target_link_libraries(edge_opencl ArrayFire::afopencl)

  add_executable(filters_opencl filters.cpp)
  target_link_libraries(filters_opencl ArrayFire::afopencl)

  add_executable(image_demo_opencl image_demo.cpp)
  target_link_libraries(image_demo_opencl ArrayFire::afopencl)

  add_executable(image_editing_opencl image_editing.cpp)
  target_link_libraries(image_editing_opencl ArrayFire::afopencl)

  add_executable(morphing_opencl morphing.cpp)
  target_link_libraries(morphing_opencl ArrayFire::afopencl)

  add_executable(optical_flow_opencl optical_flow.cpp)
  target_link_libraries(optical_flow_opencl ArrayFire::afopencl)

  add_executable(pyramids_opencl pyramids.cpp)
  target_link_libraries(pyramids_opencl ArrayFire::afopencl)

  # Gradient anisotropic diffusion example
  add_executable(gradient_diffusion_opencl gradient_diffusion.cpp)
  target_link_libraries(gradient_diffusion_opencl ArrayFire::afopencl)

  #Image Deconvolution Example
  add_executable(deconvolution_opencl deconvolution.cpp)
  target_link_libraries(deconvolution_opencl ArrayFire::afopencl)
endif()

if(ArrayFire_oneAPI_FOUND)
  add_executable(adaptive_thresholding_oneapi adaptive_thresholding.cpp)
  target_link_libraries(adaptive_thresholding_oneapi ArrayFire::afoneapi)

  add_executable(binary_thresholding_oneapi binary_thresholding.cpp)
  target_link_libraries(binary_thresholding_oneapi ArrayFire::afoneapi)

  add_executable(brain_segmentation_oneapi brain_segmentation.cpp)
  target_link_libraries(brain_segmentation_oneapi ArrayFire::afoneapi)

  add_executable(confidence_connected_components_oneapi
      confidence_connected_components.cpp)
  target_link_libraries(confidence_connected_components_oneapi ArrayFire::afoneapi)

  add_executable(edge_oneapi edge.cpp)
  target_link_libraries(edge_oneapi ArrayFire::afoneapi)

  add_executable(filters_oneapi filters.cpp)
  target_link_libraries(filters_oneapi ArrayFire::afoneapi)

  add_executable(image_demo_oneapi image_demo.cpp)
  target_link_libraries(image_demo_oneapi ArrayFire::afoneapi)

  add_executable(image_editing_oneapi image_editing.cpp)
  target_link_libraries(image_editing_oneapi ArrayFire::afoneapi)

  add_executable(morphing_oneapi morphing.cpp)
  target_link_libraries(morphing_oneapi ArrayFire::afoneapi)

  add_executable(optical_flow_oneapi optical_flow.cpp)
  target_link_libraries(optical_flow_oneapi ArrayFire::afoneapi)

  add_executable(pyramids_oneapi pyramids.cpp)
  target_link_libraries(pyramids_oneapi ArrayFire::afoneapi)

  # Gradient anisotropic diffusion example
  add_executable(gradient_diffusion_oneapi gradient_diffusion.cpp)
  target_link_libraries(gradient_diffusion_oneapi ArrayFire::afoneapi)

  #Image Deconvolution Example
  add_executable(deconvolution_oneapi deconvolution.cpp)
  target_link_libraries(deconvolution_oneapi ArrayFire::afoneapi)
endif()
