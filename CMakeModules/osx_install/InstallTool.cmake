
EXECUTE_PROCESS( COMMAND otool -L ${CMAKE_CURRENT_BINARY_DIR}/package/lib/libforge.dylib
                 COMMAND grep glfw
                 COMMAND cut -d\  -f1
                 COMMAND xargs -Jglfwlib install_name_tool -change glfwlib /usr/local/lib/libglfw3.dylib ${CMAKE_CURRENT_BINARY_DIR}/package/lib/libforge.dylib
                 OUTPUT_FILE /tmp/af.out
                 ERROR_FILE /tmp/af.err
)

