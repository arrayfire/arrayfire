Using ArrayFire with Microsoft Windows and Visual Studio {#using_on_windows}
=====

## Pre-requisites

If you have not already done so, please make sure you have installed,
configured, and tested ArrayFire following the
[installation instructions](\ref installing).

## Testing the installation

### Step 1: Running pre-built executables

The ArrayFire installer ships with a few pre-built executables with the examples.
These should run out of the box.

Note: For the CUDA executables, you will need to copy CUDA_PATH\nvvm\bin\nvvm64_30_0.dll
to the location of the executables.

### Step 2: Verify the path addition functions correctly

1. Open Visual Studio 2013. Open the HelloWorld solution which is located at
   `AF_PATH/examples/helloworld/helloworld.sln`.
2. Build and run the `helloworld` example. Be sure to, select the
   platform/configuration of your choice using the platform drop-down
   (the options are CPU, CUDA, OpenCL, and Unified) and Solution Configuration
   drop down (options of Release and Debug) menus.
3. Run the `helloworld` example

## Creating your own Visual Studio Project

### A new project from scratch

If you are creating a new project which is intended to be platform-independent,
the best option is to simply copy the existing `helloworld` solution files
and modify them to suit your needs. This will retain all the platform based
settings that have been configured in the examples. You can find the example
in the `AF_PATH/examples/helloworld/helloworld.sln` directory.

### Adding ArrayFire CPU/OpenCL to a new/existing project

If you are adding ArrayFire to a new or existing project that will contain
custom CPU or OpenCL kernels, you only need to make a few modifications to
your project soultion:

1. Open an existing project or create a new "Empty C/C++ project in Visual Studio"
2. Add `$(AF_PATH)/include;` to
   _Project Properties -> C/C++ -> General -> Additional Include Directories_
3. Add `$(AF_PATH)/lib;` to
  _Project Properties -> Linker -> General -> Additional Library Directories_
4. Add `afcpu.lib` or `afcuda.lib` or `afopencl.lib` to
  _Project Properties -> Linker -> Input -> Additional Dependencies_
  based on your preferred backend.
5. (Optional) You make choose to define `NOMINMAX`, `AF_<CPU/CUDA/OPENCL>`
  and/or `AF_<DEBUG/RELEASE>` in your projects. This can be added to
  _Project Properties -> C/C++ -> General -> Preprocessor-> Preprocessory definitions_.

### Adding ArrayFire CUDA to a new/existing project

Lastly, if your project contains custom CUDA code, the instructions are slightly
different:

1. Create a custom "CUDA NVCC project" in Visual Studio
2. Follow steps 2-5 from the _Adding ArrayFire CPU/OpenCL to a new/existing project_
   instructions above
3. Add the following lines to the
   _Project Properties -> Build Events -> Post Build Events_
   dialog:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.c}
echo copy "$(CUDA_PATH)\nvvm\bin\nvvm64*.dll" "$(OutDir)"
copy "$(CUDA_PATH)\nvvm\bin\nvvm64*.dll" "$(OutDir)"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4. Ensure that you use x64 based configurations.

Please note that this method will not work with the ArrayFire examples as
our implementations are built with the Visual Studio CL compiler rather than
NVCC to ensure they are supported across various platforms.
