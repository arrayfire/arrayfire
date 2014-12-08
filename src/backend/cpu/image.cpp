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
#include <err_cpu.hpp>

#include <iostream>
#include <cstring>
#include <cstdio>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

using af::dim4;

namespace cpu
{
    struct Window
    {
        GLFWwindow*     pWindow;
        GLEWContext*    pGLEWContext;
        int             arrWidth;
        int             arrHeight;
        int             uiWidth;
        int             uiHeight;
        int             uiID;

        //OpenGL PBO and texture "names"
        GLuint gl_PBO;
        GLuint gl_Tex;
        GLuint gl_Shader;
    };

    typedef Window* WindowHandle;
    static unsigned int g_uiWindowCounter = 0;

    static std::vector<WindowHandle> windows;
    static std::vector<int> closedWindows;

    static WindowHandle current = NULL;

    // Print for OpenGL errors
    // Returns 1 if an OpenGL error occurred, 0 otherwise.

    #define CheckGL(msg)      glErrorCheck     (msg, __FILE__, __LINE__)
    #define ForceCheckGL(msg) glForceErrorCheck(msg, __FILE__, __LINE__)
    #define CheckGLSkip(msg)  glErrorSkip      (msg, __FILE__, __LINE__)

    inline GLenum glErrorSkip(const char *msg, const char* file, int line)
    {
    #ifndef NDEBUG
        GLenum x = glGetError();
        if (x != GL_NO_ERROR) {
            printf("GL Error Skipped at: %s:%d Message: %s Error Code: %d \"%s\"\n", file, line, msg, x, gluErrorString(x));
        }
        return x;
    #else
        return 0;
    #endif
    }

    inline GLenum glErrorCheck(const char *msg, const char* file, int line)
    {
    // Skipped in release mode
    #ifndef NDEBUG
        GLenum x = glGetError();

        if (x != GL_NO_ERROR) {
            printf("GL Error at: %s:%d Message: %s Error Code: %d \"%s\"\n", file, line, msg, x, gluErrorString(x));
            AF_ERROR("Error in Graphics", AF_ERR_GL_ERROR);
        }
        return x;
    #else
        return 0;
    #endif
    }

    inline GLenum glForceErrorCheck(const char *msg, const char* file, int line)
    {
        GLenum x = glGetError();

        if (x != GL_NO_ERROR) {
            printf("GL Error at: %s:%d Message: %s Error Code: %d \"%s\"\n", file, line, msg, x, gluErrorString(x));
            AF_ERROR("Error in Graphics", AF_ERR_GL_ERROR);
        }
        return x;
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
        AF_ERROR("Error in GLFW", AF_ERR_GL_ERROR);
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
        CheckGL("Before MakeContextCurrent");
        if (wh != NULL)
        {
            glfwMakeContextCurrent(wh->pWindow);
            current = wh;
        }
        CheckGL("In MakeContextCurrent");
    }

    void Draw()
    {
        CheckGL("Before Draw");

        // load texture from PBO
        glBindTexture(GL_TEXTURE_2D, current->gl_Tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, current->arrWidth, current->arrHeight, GL_RGB, GL_FLOAT, 0);

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

        ForceCheckGL("In Draw");
    }

    void CopyArrayToPBO(const Array<float> &X)
    {
        CheckGL("Before CopyArrayToPBO");
        const float *d_X = X.get();

        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, current->arrWidth * current->arrHeight * 3 * sizeof(float),
                     d_X, GL_STREAM_COPY);

        // Unlock array
        // Not implemented yet
        // X.unlock();
        CheckGL("In CopyArrayToPBO");
    }

    WindowHandle CreateWindow(const int width, const int height, const char *title,
                              const dim_type disp_w, const dim_type disp_h)
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
        newWindow->arrWidth     = width;
        newWindow->arrHeight    = height;
        newWindow->uiWidth      = disp_w;
        newWindow->uiHeight     = disp_h;

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
        newWindow->pWindow = glfwCreateWindow(newWindow->uiWidth, newWindow->uiHeight, title, NULL, NULL);

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

        int b_width  = newWindow->uiWidth;
        int b_height = newWindow->uiHeight;
        glfwGetFramebufferSize(newWindow->pWindow, &b_width, &b_height);

        glViewport(0, 0, b_width, b_height);

        glfwSetKeyCallback(newWindow->pWindow, key_callback);

        CheckGL("Before Texture Initialization");
        // Initialize OpenGL Items
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &(newWindow->gl_Tex));
        glBindTexture(GL_TEXTURE_2D, newWindow->gl_Tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, newWindow->arrWidth, newWindow->arrHeight, 0, GL_RGB, GL_FLOAT, NULL);

        CheckGL("Before PBO Initialization");
        glGenBuffers(1, &(newWindow->gl_PBO));
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, newWindow->gl_PBO);
        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, newWindow->arrWidth * newWindow->arrHeight * 3 * sizeof(float), NULL, GL_STREAM_COPY);

        CheckGL("Before Shader Initialization");
        // load shader program
        newWindow->gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);

        //glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        //glBindTexture(GL_TEXTURE_2D, 0);

        windows.push_back(newWindow);
        MakeContextCurrent(newWindow);

        CheckGL("At End of Create Window");
        return newWindow;
    }

    void DeleteWindow(WindowHandle window)
    {
        CheckGL("Before Delete Window");
        // Cleanup
        MakeContextCurrent(window);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glDeleteBuffers(1, &window->gl_PBO);
        glDeleteTextures(1, &window->gl_Tex);
        glDeleteProgramsARB(1, &window->gl_Shader);

        CheckGL("In Delete Window");

        // Delete GLEW context and GLFW window
        delete window->pGLEWContext;
        glfwDestroyWindow(window->pWindow);

        // Delete memory
        closedWindows.push_back(window->uiID);

        delete window;
        windows.erase(std::find(windows.begin(), windows.end(), window));
    }

    void ShutDown()
    {
        for(int i = 0; i < (int)windows.size(); i++) {
            DeleteWindow(windows[i]);
        }
        glfwTerminate();
    }

    int CopyAndDraw(const Array<float> &in, WindowHandle window)
    {
        if(!glfwWindowShouldClose(window->pWindow)) {
            MakeContextCurrent(window);
            CopyArrayToPBO(in);
            Draw();
            return window->uiID;
        } else {
            DeleteWindow(window);
            return -2;
        }
    }

    int image(const Array<float> &in, const int wId, const char* title,
              const dim_type disp_w, const dim_type disp_h)
    {
        WindowHandle window = NULL;
        int ret = -1;
        if(wId == -1) {
            window = CreateWindow(in.dims()[1], in.dims()[2], title, disp_w, disp_h);
            CopyAndDraw(in, window);
            ret = window->uiID;
        } else {
            for(int i = 0; i <= wId; i++) {
                if(windows[i]->uiID == wId) {
                    window = windows[i];
                    ret = CopyAndDraw(in, window);
                    break;
                }
            }
            if(ret == -1) {
                if(std::find(closedWindows.begin(), closedWindows.end(), wId) != closedWindows.end())
                    return -2;
                else
                    AF_ERROR("Invalid Window ID", AF_ERR_INVALID_ARG);
            }
        }
        return ret;
    }
}

#else   // WITH_GRAPHICS
// No Graphics
#include <Array.hpp>
#include <graphics.hpp>
#include <err_cpu.hpp>
#include <stdio.h>
namespace cpu
{
    int image(const Array<float> &in, const int wId, const char *title,
              const dim_type disp_w, const dim_type disp_h)
    {
        printf("Error: Graphics requirements not available. See https://github.com/arrayfire/arrayfire\n");
        AF_ERROR("Graphics not Available", AF_ERR_NOT_CONFIGURED);
    }
}
#endif  // WITH_GRAPHICS
