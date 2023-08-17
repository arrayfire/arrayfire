# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(ArrayFire-Example-Linear-Algebra
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)

add_definitions("-DASSETS_DIR=\"${ASSETS_DIR}\"")

if(ArrayFire_CPU_FOUND)
  # Bagging example
  add_executable(bagging_cpu bagging.cpp)
  target_link_libraries(bagging_cpu ArrayFire::afcpu)

  # Deep Belief Network example
  add_executable(deep_belief_net_cpu deep_belief_net.cpp)
  target_link_libraries(deep_belief_net_cpu ArrayFire::afcpu)

  # Genetic Algorithm example
  add_executable(geneticalgorithm_cpu geneticalgorithm.cpp)
  target_link_libraries(geneticalgorithm_cpu ArrayFire::afcpu)

  # k Means example
  add_executable(kmeans_cpu kmeans.cpp)
  target_link_libraries(kmeans_cpu ArrayFire::afcpu)

  # Logistic Regression example
  add_executable(logistic_regression_cpu logistic_regression.cpp)
  target_link_libraries(logistic_regression_cpu ArrayFire::afcpu)

  # Naive Bayes example
  add_executable(naive_bayes_cpu naive_bayes.cpp)
  target_link_libraries(naive_bayes_cpu ArrayFire::afcpu)

  # Neural Network example
  add_executable(neural_network_cpu neural_network.cpp)
  target_link_libraries(neural_network_cpu ArrayFire::afcpu)

  # Preceptron example
  add_executable(perceptron_cpu perceptron.cpp)
  target_link_libraries(perceptron_cpu ArrayFire::afcpu)

  # Restricted Boltsmann Machine example
  add_executable(rbm_cpu rbm.cpp)
  target_link_libraries(rbm_cpu ArrayFire::afcpu)

  # Softmax Regression example
  add_executable(softmax_regression_cpu softmax_regression.cpp)
  target_link_libraries(softmax_regression_cpu ArrayFire::afcpu)
endif()

if(ArrayFire_CUDA_FOUND)
  add_executable(bagging_cuda bagging.cpp)
  target_link_libraries(bagging_cuda ArrayFire::afcuda)

  add_executable(deep_belief_net_cuda deep_belief_net.cpp)
  target_link_libraries(deep_belief_net_cuda ArrayFire::afcuda)

  add_executable(geneticalgorithm_cuda geneticalgorithm.cpp)
  target_link_libraries(geneticalgorithm_cuda ArrayFire::afcuda)

  add_executable(kmeans_cuda kmeans.cpp)
  target_link_libraries(kmeans_cuda ArrayFire::afcuda)

  add_executable(logistic_regression_cuda logistic_regression.cpp)
  target_link_libraries(logistic_regression_cuda ArrayFire::afcuda)

  add_executable(naive_bayes_cuda naive_bayes.cpp)
  target_link_libraries(naive_bayes_cuda ArrayFire::afcuda)

  add_executable(neural_network_cuda neural_network.cpp)
  target_link_libraries(neural_network_cuda ArrayFire::afcuda)

  add_executable(perceptron_cuda perceptron.cpp)
  target_link_libraries(perceptron_cuda ArrayFire::afcuda)

  add_executable(rbm_cuda rbm.cpp)
  target_link_libraries(rbm_cuda ArrayFire::afcuda)

  add_executable(softmax_regression_cuda softmax_regression.cpp)
  target_link_libraries(softmax_regression_cuda ArrayFire::afcuda)
endif()

if(ArrayFire_OpenCL_FOUND)
  add_executable(bagging_opencl bagging.cpp)
  target_link_libraries(bagging_opencl ArrayFire::afopencl)

  add_executable(deep_belief_net_opencl deep_belief_net.cpp)
  target_link_libraries(deep_belief_net_opencl ArrayFire::afopencl)

  add_executable(geneticalgorithm_opencl geneticalgorithm.cpp)
  target_link_libraries(geneticalgorithm_opencl ArrayFire::afopencl)

  add_executable(kmeans_opencl kmeans.cpp)
  target_link_libraries(kmeans_opencl ArrayFire::afopencl)

  add_executable(logistic_regression_opencl logistic_regression.cpp)
  target_link_libraries(logistic_regression_opencl ArrayFire::afopencl)

  add_executable(naive_bayes_opencl naive_bayes.cpp)
  target_link_libraries(naive_bayes_opencl ArrayFire::afopencl)

  add_executable(neural_network_opencl neural_network.cpp)
  target_link_libraries(neural_network_opencl ArrayFire::afopencl)

  add_executable(perceptron_opencl perceptron.cpp)
  target_link_libraries(perceptron_opencl ArrayFire::afopencl)

  add_executable(rbm_opencl rbm.cpp)
  target_link_libraries(rbm_opencl ArrayFire::afopencl)

  add_executable(softmax_regression_opencl softmax_regression.cpp)
  target_link_libraries(softmax_regression_opencl ArrayFire::afopencl)
endif()

if(ArrayFire_oneAPI_FOUND)
  add_executable(bagging_oneapi bagging.cpp)
  target_link_libraries(bagging_oneapi ArrayFire::afoneapi)

  add_executable(deep_belief_net_oneapi deep_belief_net.cpp)
  target_link_libraries(deep_belief_net_oneapi ArrayFire::afoneapi)

  add_executable(geneticalgorithm_oneapi geneticalgorithm.cpp)
  target_link_libraries(geneticalgorithm_oneapi ArrayFire::afoneapi)

  add_executable(kmeans_oneapi kmeans.cpp)
  target_link_libraries(kmeans_oneapi ArrayFire::afoneapi)

  add_executable(logistic_regression_oneapi logistic_regression.cpp)
  target_link_libraries(logistic_regression_oneapi ArrayFire::afoneapi)

  add_executable(naive_bayes_oneapi naive_bayes.cpp)
  target_link_libraries(naive_bayes_oneapi ArrayFire::afoneapi)

  add_executable(neural_network_oneapi neural_network.cpp)
  target_link_libraries(neural_network_oneapi ArrayFire::afoneapi)

  add_executable(perceptron_oneapi perceptron.cpp)
  target_link_libraries(perceptron_oneapi ArrayFire::afoneapi)

  add_executable(rbm_oneapi rbm.cpp)
  target_link_libraries(rbm_oneapi ArrayFire::afoneapi)

  add_executable(softmax_regression_oneapi softmax_regression.cpp)
  target_link_libraries(softmax_regression_oneapi ArrayFire::afoneapi)
endif()
