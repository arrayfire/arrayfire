# Function to turn an OpenCL source file into a C string within a source file.
# xxd uses its input's filename to name the string and its length, so we
# need to move them to a name that depends only on the path output, not its
# input.  Otherwise, builds in different relative locations would put the
# source into different variable names, and everything would fall over.
# The actual name will be filename (.s replaced with underscores), and length
# name_len.
#
# Usage example:
#
# set(KERNELS a.cl b/c.cl)
# resource_to_cxx_source(
#   SOURCES ${KERNELS}
#   VARNAME OUTPUTS
# )
# add_executable(foo ${OUTPUTS})
#
# The namespace they are placed in is taken from filename.namespace.
#
# For example, if the input file is kernel.cl, the two variables will be
#  unsigned char ns::kernel_cl[];
#  unsigned int ns::kernel_cl_len;
#
# where ns is the contents of kernel.cl.namespace.

set(BIN2CPP_PROGRAM "bin2cpp")

function(FILE_TO_STRING)
    cmake_parse_arguments(RTCS "WITH_EXTENSION;NULLTERM" "VARNAME;EXTENSION;OUTPUT_DIR;TARGETS;NAMESPACE;BINARY" "SOURCES" ${ARGN})

    set(_output_files "")
    foreach(_input_file ${RTCS_SOURCES})
        get_filename_component(_path "${_input_file}" PATH)
        get_filename_component(_name "${_input_file}" NAME)
        get_filename_component(var_name "${_input_file}" NAME)
        get_filename_component(_name_we "${_input_file}" NAME_WE)

        set(_namespace "${RTCS_NAMESPACE}")
        set(_binary "")
        if(${RTCS_BINARY})
            set(_binary "--binary")
        endif(${RTCS_BINARY})
        if(RTCS_NULLTERM)
            set(_nullterm "--nullterm")
        endif(RTCS_NULLTERM)

        string(REPLACE "." "_" var_name ${var_name})
        string(REPLACE "\ " "_" namespace_name ${RTCS_NAMESPACE})

        set(_output_path "${CMAKE_CURRENT_BINARY_DIR}/${RTCS_OUTPUT_DIR}")
        if(RTCS_WITH_EXTENSION)
          set(_output_file "${_output_path}/${var_name}.${RTCS_EXTENSION}")
        else()
          set(_output_file "${_output_path}/${_name_we}.${RTCS_EXTENSION}")
        endif()

        add_custom_command(
            OUTPUT ${_output_file}
            DEPENDS ${_input_file} ${BIN2CPP_PROGRAM}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_output_path}"
            COMMAND ${CMAKE_COMMAND} -E echo "\\#include \\<${_path}/${_name_we}.hpp\\>"  >>"${_output_file}"
            COMMAND ${BIN2CPP_PROGRAM} --file ${_name} --namespace ${_namespace} --output ${_output_file} --name ${var_name} ${_binary} ${_nullterm}
            WORKING_DIRECTORY "${_path}"
            COMMENT "Compiling ${_input_file} to C++ source"
        )


        list(APPEND _output_files ${_output_file})
    endforeach()
    add_custom_target(${namespace_name}_${RTCS_OUTPUT_DIR}_bin_target DEPENDS ${_output_files})
    set_target_properties(${namespace_name}_${RTCS_OUTPUT_DIR}_bin_target PROPERTIES FOLDER "Generated Targets")

    set("${RTCS_VARNAME}" ${_output_files} PARENT_SCOPE)
    set("${RTCS_TARGETS}" ${namespace_name}_${RTCS_OUTPUT_DIR}_bin_target PARENT_SCOPE)
endfunction(FILE_TO_STRING)
