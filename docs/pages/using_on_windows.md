Using ArrayFire with Microsoft Windows and Visual Studio {#using_on_windows}
============================================================================

If you have not already done so, please make sure you have installed,
configured, and tested ArrayFire following the [installation instructions](#installing).

# The big picture {#big-picture-windows}

The ArrayFire Windows installer creates the following:
1. **AF_PATH** environment variable to point to the installation location. The
   default install location is `C:\Program Files\ArrayFire\v3`
2. **AF_PATH/include** : Header files for ArrayFire (include directory)
3. **AF_PATH/lib** : All ArrayFire backends libraries, dlls and dependency dlls
   (library directory)
4. **AF_PATH/examples** : Examples to get started.
5. **AF_PATH/cmake** : CMake config files
6. **AF_PATH/uninstall.exe** : Uninstaller

The installer will prompt the user for following three options.
* Do not add **%%AF_PATH%/lib** to PATH
* Add **%%AF_PATH%/lib** to PATH environment variable of current user
* Add **%%AF_PATH%/lib** to PATH environment variable for all users

If you chose not to modify PATH during installation please make sure to do so
manually so that all applications using ArrayFire libraries will be able to find
the required DLLs.

# Build and Run Helloworld {#section1}

This can be done in two ways either by using CMake build tool or using Visual
Studio directly.

##  Using CMake {#section1part1}
1. Download and install [CMake](https://cmake.org/download/), preferrably the
   latest version.
2. Open CMake-GUI and set the field __Where is the source code__ to the root
   directory of examples.
3. Set the field __Where to build the binaries__ to
   **path_to_examples_root_dir/build** and click the `Configure` button towards
   the lower left bottom.
4. CMake will prompt you asking if it has to create the `build` directory if
   it's not already present. Click yes to create the build directory.
5. Before the configuration begins, CMake will show you a list(drop-down menu)
   of available Visual Studio versions on your system to chose from. Select one
   and check the radio button that says **Use default native compilers** and
   click finish button in the bottom right corner.
6. CMake will show you errors in red text if any once configuration is finished.
   Ideally, you wouldn't need to do anything and CMake should be able to find
   ArrayFire automatically. Please let us know if it didn't on your machine.
7. Click **Generate** button to generate the Visual Studio solution files for
   the examples.
8. Click **Open Project** button that is right next to **Generate** button to
   open the solution file.
9. You will see a bunch of examples segregated into three sets named after the
   compute backends of ArrayFire: cpu, cuda & opencl if you have installed all
   backends. Select the helloworld project from any of the installed backends
   and mark it as startup project and hit `F5`.
10. Once the helloworld example builds, you will see a console window with the
    output from helloworld program.

## Using Visual Studio {#section1part2}

1. Open Visual Studio of your choice and create an empty C++ project.
2. Right click the project and add an existing source file
   `examples/helloworld/helloworld.cpp` to this project.
3. Add `"$(AF_PATH)/include;"` to _Project Properties -> C/C++ -> General ->
   Additional Include Directories_.
4. Add `"$(AF_PATH)/lib;"` to _Project Properties -> Linker -> General ->
   Additional Library Directories_.
5. Add `afcpu.lib` or `afcuda.lib` or `afopencl.lib` to _Project Properties ->
   Linker -> Input -> Additional Dependencies_. based on your preferred backend.
6. (Optional) You may choose to define `NOMINMAX`, `AF_<CPU/CUDA/OPENCL>` and/or
   `AF_<DEBUG/RELEASE>` in your projects. This can be added to _Project
   Properties -> C/C++ -> General -> Preprocessor-> Preprocessory definitions_.
7. Build and run the project. You will see a console window with the output from
   helloworld program.

# Using ArrayFire within Existing Visual Studio Projects {#section2}
This is divided into three parts:
* [Part A: Adding ArrayFire to an existing solution (Single Backend)](#section2partA)
* [Part B: Adding ArrayFire CUDA to a new/existing CUDA project](#section2partB)
* [Part C: Project with all ArrayFire backends](#section2partC)

## Part A: Adding ArrayFire to an existing solution (Single Backend) {#section2partA}

Note: If you plan on using Native CUDA code in the project, use the steps under
[Part B](#section2partB).

Adding a single backend to an existing project is quite simple.

1. Add `"$(AF_PATH)/include;"` to _Project Properties -> C/C++ -> General ->
   Additional Include Directories_.
2. Add `"$(AF_PATH)/lib;"` to _Project Properties -> Linker -> General ->
   Additional Library Directories_.
3. Add `afcpu.lib`, `afcuda.lib`, `afopencl.lib`, or `af.lib` to _Project
   Properties -> Linker -> Input -> Additional Dependencies_. based on your
   preferred backend.

## Part B: Adding ArrayFire CUDA to a new/existing CUDA project {#section2partB}
Lastly, if your project contains custom CUDA code, the instructions are slightly
different as it requires using a CUDA NVCC Project:

1. Create a custom "CUDA NVCC project" in Visual Studio
2. Add `"$(AF_PATH)/include;"` to _Project Properties -> CUDA C/C++ -> General
   -> Additional Include Directories_.
3. Add `"$(AF_PATH)/lib;"` to _Project Properties -> Linker -> General ->
   Additional Library Directories_.
4. Add `afcpu.lib`, `afcuda.lib`, `afopencl.lib`, or `af.lib` to _Project Properties ->
   Linker -> Input -> Additional Dependencies_. based on your preferred backend.

### Part C: Project with all ArrayFire backends {#section2partC}
If you wish to create a project that allows you to use all the ArrayFire
backends with ease, you should use `af.lib` in step 3 from [Part
A](#section2partA).

You can alternately download the template project from [ArrayFire Template
Projects](https://github.com/arrayfire/arrayfire-project-templates)

# <a name="section3" />Using ArrayFire with CMake
ArrayFire ships with a series of CMake scripts to make finding and using our
library easy.

First create a file called `CMakeLists.txt` in your project directory:

    cd your-project-directory
    touch CMakeLists.txt

and populate it with the following code:

    find_package(ArrayFire)
    add_executable(<my_executable> [list your source files here])

    # To use Unified backend, do the following.
    # Unified backend lets you choose the backend at runtime
    target_link_libraries(<my_executable> ArrayFire::af)

where `<my_executable>` is the name of the executable you wish to create. See the
[CMake documentation](https://cmake.org/documentation/) for more information on
how to use CMake. To link with a specific backend directly, replace the
`ArrayFire::af` with the following for their respective backends.

* `ArrayFire::afcpu` for CPU backend.
* `ArrayFire::afcuda` for CUDA backend.
* `ArrayFire::afopencl` for OpenCL backend.

Next we need to instruct CMake to create build instructions and then compile. We
suggest using CMake's out-of-source build functionality to keep your build and
source files cleanly separated. To do this open the CMake GUI.

* Under source directory, add the path to your project
* Under build directory, add the path to your project and append /build
* Click configure and choose a 64 bit Visual Studio generator.
* If configuration was successful, click generate. This will create a
  my-project.sln file under build. Click `Open Project` in CMake-GUI to open the
  solution and compile the ALL_BUILD project.
