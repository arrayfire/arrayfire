Using ArrayFire with Microsoft Windows and Visual Studio {#using_on_windows}
=====

If you have not already done so, please make sure you have installed,
configured, and tested ArrayFire following the
[installation instructions](\ref installing).

## The big picture
The ArrayFire Windows installer creates the following:
1. `AF_PATH` environment variable to point to the installation location. The
   default install location is `C:\Program Files\ArrayFire\v3`
2. `AF_PATH/include`         : Header files for ArrayFire (include directory)
3. `AF_PATH/lib`             : All ArrayFire backends libraries, dlls and dependency dlls (library directory)
4. `AF_PATH/examples`        : Examples to get started. Some examples also have pre-built exectuables
5. `AF_PATH/cmake`           : CMake config files for automatic configuration by external projects
6. `AF_PATH/uninstall.exe`   : Uninstaller
7. `AF_PATH/*`               : Other miscellenous files including licenses, logos, copyrights

The installer also appends `%%AF_PATH%/lib` to the User PATH variable.

To add `%%AF_PATH%/lib` to PATH for all users see the windows section in
[installation instructions](\ref installing).

### <a name="nvvm_dlls" />Dealing with CUDA NMMV DLLs
When using CUDA with ArrayFire you may encounter a linker error indicating the
NVVM DLLs are missing. This is because the NVVM DLLs are not part of the
standard `CUDA_PATH\bin` installation directory that is added to your `PATH`
when the CUDA installer runs. Thus, NVVM will not be found during runtime. There
are a few ways to deal with this issue:

1. Copy the DLLs to the exectuable location. This is, by far, the cleanest
   solution and we recommend doing this with ArrayFire projects. To do so,
   create a post-build event to copy the NVVM DLL as discusses below in
   [Step 3 - Part A](#s3partA).
2. Copy `CUDA_PATH\nvvm\bin\nvvm64_30_0.dll` to `CUDA_PATH\bin`. This is a one time
   copy such that the NVVM DLL is now with all the other CUDA dlls and in a
   directory that is a part of PATH and hence the DLL can be detected automatically.
3. Add `%%CUDA_PATH%\nvvm\bin` to the system PATH environment variable.
   This will allow automatic detection by the system and No further copying will
   be required. ArrayFire does not add this to PATH since the CUDA installer
   doesn't add it to PATH.

## <a name="step1" />Step 1: Running pre-built executables

The ArrayFire installer ships with a few pre-built executables with the examples.
These should run out of the box when double clicked.

Some prebuilt examples are:
* Helloworld (examples/helloworld)
* BLAS (examples/benchmarks)
* FFT (examples/benchmarks)
* Pi Estimation (examples/benchmarks)
* Conway (Graphics) (examples/graphics)

Note: For the CUDA executables, you will need to copy `CUDA_PATH\nvvm\bin\nvvm64_30_0.dll`
to the location of the executables.

## <a name="step2" />Step 2: Build and Run a Project

1. Open Visual Studio 2013. Load the HelloWorld solution which is located at
   `AF_PATH/examples/helloworld/helloworld.sln`.
2. Build the `helloworld` example. Be sure to, select the platform/configuration
   of your choice using the platform drop-down (the options are CPU, CUDA,
   OpenCL, and Unified) and Solution Configuration drop down (options of Release
   and Debug) menus.
3. Run the `helloworld` example.

## <a name="step3" />Step 3: Using ArrayFire within Visual Studio
This is divided into 4 parts:
* [Part A: Adding ArrayFire to an existing solution (Single Backend)](#s3partA)
* [Part B: Adding ArrayFire CUDA to a new/existing CUDA project](#s3partB)
* [Part C: Project with all ArrayFire backends](#s3partC)
* [Part D: ArrayFire with CMake](#s3partD)

### <a name="s3partA" />Part A: Adding ArrayFire to an existing solution (Single Backend)
Note: If you plan on using Native CUDA code in the project, use the steps
under [Part B](#s3partB).

Adding a single backend to an existing project is quite simple.

1. Add `"$(AF_PATH)/include;"` to
   _Project Properties -> C/C++ -> General -> Additional Include Directories_.
2. Add `"$(AF_PATH)/lib;"` to
   _Project Properties -> Linker -> General -> Additional Library Directories_.
3. Add `afcpu.lib` or `afcuda.lib` or `afopencl.lib` to
   _Project Properties -> Linker -> Input -> Additional Dependencies_.
   based on your preferred backend.
4. (Optional) You may choose to define `NOMINMAX`, `AF_<CPU/CUDA/OPENCL>`
   and/or `AF_<DEBUG/RELEASE>` in your projects. This can be added to
   _Project Properties -> C/C++ -> General -> Preprocessor-> Preprocessory definitions_.

If you are using the CUDA backend, it is important to ensure that the CUDA NVVM
DLLs are copied to the exectuable directory. This can be done by adding a post
build event.

Open the _Project Properties -> Build Events -> Post Build Events_ dialog and
add the following lines to it.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.c}
echo copy "$(CUDA_PATH)\nvvm\bin\nvvm64*.dll" "$(OutDir)"
copy "$(CUDA_PATH)\nvvm\bin\nvvm64*.dll" "$(OutDir)"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### <a name="s3partB" />Part B: Adding ArrayFire CUDA to a new/existing CUDA project
Lastly, if your project contains custom CUDA code, the instructions are slightly
different as it requires using a CUDA NVCC Project:

1. Create a custom "CUDA NVCC project" in Visual Studio
2. Add `"$(AF_PATH)/include;"` to
   _Project Properties -> CUDA C/C++ -> General -> Additional Include Directories_.
3. Add `"$(AF_PATH)/lib;"` to
   _Project Properties -> Linker -> General -> Additional Library Directories_.
4. Add `afcpu.lib` or `afcuda.lib` or `afopencl.lib` to
   _Project Properties -> Linker -> Input -> Additional Dependencies_.
   based on your preferred backend.
5. (Optional) You may choose to define `NOMINMAX`, `AF_CUDA`
   and/or `AF_<DEBUG/RELEASE>` in your projects. This can be added to
   _Project Properties -> C/C++ -> General -> Preprocessor-> Preprocessory definitions_.
6. Pick a solution to handle the NVVM DLLs. We recommend the post build event
   method used in [Part A](#s3partA).

### <a name="s3partC" />Part C: Project with all ArrayFire backends
If you wish to create a project that allows you to use all the ArrayFire
backends with ease, the best way to go is to copy the *HelloWorld sln/vcxproj/cpp*
file trio and rename them to suit your project.

All the ArrayFire examples are pre-configured for all ArrayFire backends as well
as the Unified API. These can be chosen from the Solution/Platform configuration
drop down boxes.

You can alternately download the template project from
[ArrayFire Template Projects](https://github.com/arrayfire/arrayfire-project-templates)

### <a name="s3partD" />Part D: ArrayFire with CMake
*NOTE:* The ArrayFire installer sets up CMake file and registry so that it can be found
by CMake by simply using the `Find_PACKAGE(ArrayFire)` command.

If you are writing a new ArrayFire project in C/C++ from scratch, we suggest
you grab a copy of our
[CMake Project Example](https://github.com/arrayfire/arrayfire-project-templates);
however, it is useful to read the documentation below in case you need to add
ArrayFire to an existing project.

As [discussed above](#big-picture), ArrayFire ships with a series of CMake
scripts to make finding and using our library easy.
The scripts will automatically find all versions of the ArrayFire library
and pick the most powerful of the installed backends (typically CUDA).

First create a file called `CMakeLists.txt` in your project directory:

    cd your-project-directory
    touch CMakeLists.txt

and populate it with the following code:

    FIND_PACKAGE(ArrayFire)
    INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})

    ... [gather source files, etc.]

    # If you intend to use OpenCL, you need to find it
    FIND_PACKAGE(OpenCL)
    SET(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${OpenCL_LIBRARIES})

    # Or if you intend to use CUDA, you need it as well as NVVM:
    FIND_PACKAGE(CUDA)
    FIND_PACKAGE(NVVM) # this FIND script can be found in the ArrayFire CMake example repository
    SET(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${CUDA_LIBRARIES} ${NVVM_LIB})

    ADD_EXECUTABLE(my_executable [list your source files here])
    TARGET_LINK_LIBRARIES(my_executable ${ArrayFire_LIBRARIES} ${EXTRA_LIBS})

where `my_executable` is the name of the executable you wish to create.
See the [CMake documentation](https://cmake.org/documentation/) for more
information on how to use CMake.
Clearly the above code snippet precludes the use of both CUDA and OpenCL, see
the
[ArrayFire CMake Example](https://github.com/arrayfire/arrayfire-project-templates/tree/master/CMake)
for an example of how to build executables for both backends from the same
CMake script.

In the above code listing, the `FIND_PACKAGE` will find the ArrayFire include
files, libraries, and define several variables including:

    ArrayFire_INCLUDE_DIRS    - Location of ArrayFire's include directory.
    ArrayFire_LIBRARIES       - Location of ArrayFire's libraries.
                                This will default to a GPU backend if one
                                is found
    ArrayFire_FOUND           - True if ArrayFire has been located

If you wish to use a specific backend, the find script also defines these variables:

    ArrayFire_CPU_FOUND         - True of the ArrayFire CPU library has been found.
    ArrayFire_CPU_LIBRARIES     - Location of ArrayFire's CPU library, if found
    ArrayFire_CUDA_FOUND        - True of the ArrayFire CUDA library has been found.
    ArrayFire_CUDA_LIBRARIES    - Location of ArrayFire's CUDA library, if found
    ArrayFire_OpenCL_FOUND      - True of the ArrayFire OpenCL library has been found.
    ArrayFire_OpenCL_LIBRARIES  - Location of ArrayFire's OpenCL library, if found
    ArrayFire_Unified_FOUND     - True of the ArrayFire Unified library has been found.
    ArrayFire_Unified_LIBRARIES - Location of ArrayFire's Unified library, if found

Therefore, if you wish to target a specific specific backend, simply replace
`${ArrayFire_LIBRARIES}` with `${ArrayFire_CPU}`, `${ArrayFire_OPENCL}`,
`${ArrayFire_CUDA}`, or `${ArrayFire_Unified}` in the `TARGET_LINK_LIBRARIES`
command above.

Next we need to instruct CMake to create build instructions and then compile.
We suggest using CMake's out-of-source build functionality to keep your build
and source files cleanly separated. To do this open the CMake GUI.

* Under source directory, add the path to your project
* Under build directory, add the path to your project and append /build
* Click configure and choose Visual Studio 2013 Win 64 as the generator.
* If configuration was successful, click generate. This will create a
  my-project.sln file under build. You can open this in Visual Studio and
  compile the ALL_BUILD project.


The [ArrayFire CMake Example](https://github.com/arrayfire/arrayfire-project-templates/tree/master/CMake)
is a CMake project used to demo how ArrayFire can be using with a CMake project.

Note: The CMake project does not add the post build event to copy the NVVM DLLs
in case of CUDA backend. You will need to either copy it manually to the exectuable
directory, or pick another solution for it.

