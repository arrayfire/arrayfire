
IF(${WITH_COVERAGE})
    Find_Program(COVERALLS_EXECUTABLE coveralls)

    #FIXME: Remove after eddyxu/cpp-coveralls #75 accepted
    SET(COVERALL_ARGS "--include src --include include --exclude src/backend/opencl/cl.hpp --exclude test --gcov-options '\\-lp'")

    ADD_CUSTOM_TARGET(coveralls
        COMMAND ${COVERALLS_EXECUTABLE}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Creating coverage"
        )
ENDIF(${WITH_COVERAGE})
