/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Parts of this code sourced from SnopyDogy
// https://gist.github.com/SnopyDogy/a9a22497a893ec86aa3e


#if defined(WITH_GRAPHICS)

#include <Array.hpp>
#include <graphics.hpp>
#include <stdexcept>
#include <err_cuda.hpp>

#include <iostream>
#include <cstring>
#include <cstdio>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using af::dim4;

namespace cuda
{
    struct Window
    {
        GLFWwindow*     pWindow;
        GLEWContext*    pGLEWContext;
        int             uiWidth;
        int             uiHeight;
        int             uiID;

        cudaGraphicsResource *cudaPBOResource; // handles OpenGL-CUDA exchange

        //OpenGL PBO and texture "names"
        GLuint gl_PBO;
        GLuint gl_Tex;
        GLuint gl_Shader;
    };

    typedef Window* WindowHandle;
    static unsigned int g_uiWindowCounter = 0;

    static std::vector<WindowHandle> windows;

    static WindowHandle current = NULL;

    // Print for OpenGL errors
    // Returns 1 if an OpenGL error occurred, 0 otherwise.
    #define CheckGL() printOglError(__FILE__, __LINE__)
    int printOglError(char *file, int line)
    {
        GLenum glErr;
        int retCode = 0;
        glErr = glGetError();
        if (glErr != GL_NO_ERROR)
        {
            printf("glError in file %s @ line %d: %s\n",
                    file, line, gluErrorString(glErr));
            retCode = 1;
        }
        return retCode;
    }

    // Required to be defined for GLEW MX to work, along with the GLEW_MX define in the perprocessor!
    static GLEWContext* glewGetContext()
    {
        return current->pGLEWContext;
    }

    // gl_Shader for displaying floating-point texture
    static const char *shader_code =
        "!!ARBfp1.0\n"
        "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
        "END";

    GLuint compileASMShader(GLenum program_type, const char *code)
    {
        GLuint program_id;
        glGenProgramsARB(1, &program_id);
        glBindProgramARB(program_type, program_id);
        glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

        GLint error_pos;
        glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

        if (error_pos != -1)
        {
            const GLubyte *error_string;
            error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
            fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
            return 0;
        }

        return program_id;
    }

    static void error_callback(int error, const char* description)
    {
        fputs(description, stderr);
    }

    static void key_callback(GLFWwindow* wind, int key, int scancode, int action, int mods)
    {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(wind, GL_TRUE);
        }
    }

    void MakeContextCurrent(WindowHandle wh)
    {
        if (wh != NULL)
        {
            glfwMakeContextCurrent(wh->pWindow);
            current = wh;
        }
    }

    void Draw()
    {
        // Safety check
        //MakeContextCurrent(current);

        // load texture from PBO
        glBindTexture(GL_TEXTURE_2D, current->gl_Tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, current->uiWidth, current->uiHeight, GL_RGB, GL_FLOAT, 0);

        // fragment program is required to display floating point texture
        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, current->gl_Shader);
        glEnable(GL_FRAGMENT_PROGRAM_ARB);
        glDisable(GL_DEPTH_TEST);

        // Draw to screen
        // GLFW uses -1 to 1 normalized coordinates
        // Textures go from 0 to 1 normalized coordinates
        glBegin(GL_QUADS);
        glTexCoord2f ( 0.0f,  1.0f);
        glVertex2f   (-1.0f, -1.0f);
        glTexCoord2f ( 1.0f,  1.0f);
        glVertex2f   ( 1.0f, -1.0f);
        glTexCoord2f ( 1.0f,  0.0f);
        glVertex2f   ( 1.0f,  1.0f);
        glTexCoord2f ( 0.0f,  0.0f);
        glVertex2f   (-1.0f,  1.0f);
        glEnd();

        // Unbind textures
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_FRAGMENT_PROGRAM_ARB);

        // Complete render
        glfwSwapBuffers(current->pWindow);
        glfwPollEvents();
    }

    void CopyArrayToPBO(const Array<float> &X)
    {
        const float *d_X = X.get();

        // Map resource. Copy data to PBO. Unmap resource.
        size_t num_bytes;
        float* d_pbo = NULL;
        cudaGraphicsMapResources(1, &current->cudaPBOResource, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&d_pbo, &num_bytes, current->cudaPBOResource);
        cudaMemcpy(d_pbo, d_X, num_bytes, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &current->cudaPBOResource, 0);

        // Unlock array
        // Not implemented yet
        // X.unlock();
    }

    WindowHandle CreateWindow(const int width, const int height, const char *title)
    {
        // save current active context info so we can restore it later!
        //WindowHandle previous = current;

        // create new window data:
        WindowHandle newWindow = new Window();
        if (newWindow == NULL)
            printf("Error\n");
            //Error out

        newWindow->pGLEWContext = NULL;
        newWindow->pWindow      = NULL;
        newWindow->uiID         = g_uiWindowCounter++;        //set ID and Increment Counter!
        newWindow->uiWidth      = width;
        newWindow->uiHeight     = height;

        // Initalize GLFW
        glfwSetErrorCallback(error_callback);
        if (!glfwInit()) {
            std::cerr << "ERROR: GLFW wasn't able to initalize" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Add Hints
        glfwWindowHint(GLFW_DEPTH_BITS, 3 * 8); //RGB * 8
        glfwWindowHint(GLFW_RESIZABLE, false);

        // Create the window itself
        newWindow->pWindow = glfwCreateWindow(width, height, title, NULL, NULL);

        // Confirm window was created successfully:
        if (newWindow->pWindow == NULL)
        {
            printf("Error: Could not Create GLFW Window!\n");
            delete newWindow;
            return NULL;
        }

        // Create GLEW Context
        newWindow->pGLEWContext = new GLEWContext();
        if (newWindow->pGLEWContext == NULL)
        {
            printf("Error: Could not create GLEW Context!\n");
            delete newWindow;
            return NULL;
        }

        // Set context (before glewInit())
        MakeContextCurrent(newWindow);

        //GLEW Initialization - Must be done
        GLenum err = glewInit();
        if (err != GLEW_OK) {
            printf("GLEW Error occured, Description: %s\n", glewGetErrorString(err));
            glfwDestroyWindow(newWindow->pWindow);
            delete newWindow;
            return NULL;
        }

        int b_width = width;
        int b_height = height;
        glfwGetFramebufferSize(newWindow->pWindow, &b_width, &b_height);

        glViewport(0, 0, b_width, b_height);

        glfwSetKeyCallback(newWindow->pWindow, key_callback);

        // Initialize OpenGL Items
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &(newWindow->gl_Tex));
        glBindTexture(GL_TEXTURE_2D, newWindow->gl_Tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, newWindow->uiWidth, newWindow->uiHeight, 0, GL_RGB, GL_FLOAT, NULL);

        glGenBuffers(1, &(newWindow->gl_PBO));
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, newWindow->gl_PBO);
        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, newWindow->uiWidth * newWindow->uiHeight * 3 * sizeof(float), NULL, GL_STREAM_COPY);

        // Register PBO with CUDA
        cudaGraphicsGLRegisterBuffer(&newWindow->cudaPBOResource, newWindow->gl_PBO,
                                     cudaGraphicsMapFlagsWriteDiscard);
        // load shader program
        newWindow->gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);

        //glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        //glBindTexture(GL_TEXTURE_2D, 0);

        windows.push_back(newWindow);
        MakeContextCurrent(newWindow);
        return newWindow;
    }

    void DeleteWindow(WindowHandle window)
    {
        // Cleanup
        MakeContextCurrent(window);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glDeleteBuffers(1, &window->gl_PBO);
        glDeleteTextures(1, &window->gl_Tex);
        glDeleteProgramsARB(1, &window->gl_Shader);

        // Delete GLEW context and GLFW window
        delete window->pGLEWContext;
        glfwDestroyWindow(window->pWindow);

        // Delete CUDA Resource
        cudaGraphicsUnregisterResource(window->cudaPBOResource);

        // Delete memory
        delete window;
        windows.erase(std::find(windows.begin(), windows.end(), window));
    }

    void ShutDown()
    {
        for(int i = 0; i < windows.size(); i++) {
            DeleteWindow(windows[i]);
        }
        glfwTerminate();
    }

    void CopyAndDraw(const Array<float> &in, WindowHandle window)
    {
        if(!glfwWindowShouldClose(window->pWindow)) {
            MakeContextCurrent(window);
            CopyArrayToPBO(in);
            Draw();
        } else {
            DeleteWindow(window);
        }
    }

    int image(const Array<float> &in, const int wId, const char* title)
    {
        WindowHandle window = NULL;
        int ret = -1;
        if(wId == -1) {
            window = CreateWindow(in.dims()[1], in.dims()[2], title);
            CopyAndDraw(in, window);
            ret = window->uiID;
        } else {
            for(int i = 0; i <= wId; i++) {
                if(windows[i]->uiID == wId) {
                    window = windows[i];
                    CopyAndDraw(in, window);
                    ret = window->uiID;
                    break;
                }
            }
            if(ret == -1)
                AF_ERROR("Invalide Window ID", AF_ERR_INVALID_ARG);
        }

        return ret;
    }
}

#else   // WITH_GRAPHICS
// No Graphics
#include <af/image.h>
#include <stdio.h>
int image(const Array<float> &in, const int wId, const char *title)
{
    printf("Error: Graphics requirements not available. See https://github.com/arrayfire/arrayfire\n");
    AF_ERROR("Graphics not Available", AF_ERR_NOT_CONFIGURED);
}
#endif  // WITH_GRAPHICS
