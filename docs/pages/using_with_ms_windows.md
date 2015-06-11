Using ArrayFire with Microsoft Windows and Visual Studio {#using_on_windows}
=====

## Pre-requisites

Before you get started, make sure you have the necessary pre-requisites.

- If you are using CUDA, please make sure you have [CUDA 7](https://developer.nvidia.com/cuda-downloads) installed on your system.
     - [Contact us](support@arrayfire.com) for custom builds (eg. different toolkits)

- If you are using OpenCL, please make sure you have one of the following SDKs.
     - [AMD OpenCL SDK](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/)
     - [Intel OpenCL SDK](https://software.intel.com/en-us/articles/download-the-latest-intel-amt-software-development-kit-sdk)
     - [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)

## Step 0: Running pre-built executables

The ArrayFire installer ships with a few pre-built executables with the examples.
These should run out of the box.

Note: For the CUDA executables, you will need to copy CUDA_PATH\nvvm\bin\nvvm64_30_0.dll
to the location of the executables.

## Step 1: Adding ArrayFire to PATH for all users

The ArrayFire installer for Windows creates a user `PATH` variable containing
`%AF_PATH%/lib`. This is required so that Windows knows where to find the
ArrayFire DLLs. This variable fixes the DLL finding only for the user that
installs ArrayFire.

To allow DLL detection for all users, it needs to be added to the system
`PATH` variable. For this, follow the steps:

1. Open Advanced System Settings:
  * Windows 8: Move the Mouse pointer to the bottom right corner of the screen,
    Right click, choose System. Then click "Advanced System Settings"
  * Windows 7: Open the Start Menu and Right Click on "Computer". Then choose
    Properties and click "Advanced System Settings"

2. In _Advanced System Settings_ window, click on _Advanced_ tab

3. Click on _Environment Variables_, then under **System Variables**, find
   `PATH`, and click on it.

4. In edit mode, append `%AF_PATH%/lib`. NOTE: Ensure that there is a semi-colon
   separating `%AF_PATH%/lib` from any existing content (e.g.
   `EXISTING_PATHS;%AF_PATH%/lib;`) otherwise other software may not function
   correctly.

## Step 2: Verify the path addition functions correctly

1. Open Visual Studio 2013. Open the HelloWorld solution which is located at
   `AF_PATH/examples/helloworld/helloworld.sln`.
2. Build and run the `helloworld` example. Be sure to, select the
   platform/configuration of your choice using the platform drop-down
   (the options are CPU, CUDA, and OpenCL) and Solution Configuration drop down
   (options of Release and Debug) menus.
3. Run the `helloworld` example

## Step 3: Creating your own Visual Studio Project

### A new project from scratch

If you are creating a new project which is intended to be platform-independent,
the best option is to simply copy the existing `helloworld` solution files
and modify them to suit your needs. This will retain all the platform based
settings that have been configured in the examples.

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

     ```
     echo copy "$(CUDA_PATH)\nvvm\bin\nvvm64*.dll" "$(OutDir)"
     copy "$(CUDA_PATH)\nvvm\bin\nvvm64*.dll" "$(OutDir)"
     ```

4. Ensure that you use x64 based configurations.

Please note that this method will not work with the ArrayFire examples as
our implementations are built with the Visual Studio CL compiler rather than
NVCC to ensure they are supported across various platforms.
