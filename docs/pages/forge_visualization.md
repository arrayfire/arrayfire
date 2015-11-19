##Visualizing af::arrays with Forge {#forge_visualization}
Arrayfire as a library aims to provide a robust and easy to use platform for high-performance, parallel and GPU computing. The goal of Forge, an OpenGL visualization library, is to provide equally robust visualizations that are interoperable between Arrayfire data-structures and an OpenGL context. Instead of wasting time copying and reformatting data from the GPU to the host and back to the GPU, we can draw directly from GPU-data to GPU-framebuffers! Let's see exactly what visuals forge can illuminate for us and how it handles data from af::arrays.  

Forge functions are tied to the af::Window class. First let's create a window:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
const static int WIDTH = 512, HEIGHT = 512;
af::Window window(WIDTH, HEIGHT, "2D plot example title");

do{

//drawing functions here

} while( !window.close() );
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
Now we can use Forge's drawing functions to draw to the window.

###image
The image() function can be used to plot grayscale or color images. To plot a grayscale image a 2d array should be passed into the function. Let's see this on a static noise example:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array img = constant(0, WIDTH, HEIGHT); //make a black image
array random = randu(WIDTH, HEIGHT);      //make random [0,1] distribution
img(random > 0.5) = 1; //set all pixels where distribution > 0.5 to white

window.image(img);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
<img src="forge_viz/noise.png" alt="Forge image plot of noise" width="20%" />
Tweaking the previous example by giving our image a depth of 3 for the RGB values allows us to generate colorful noise:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array img = 255 * randu(WIDTH, HEIGHT, 3);      //make random [0, 255] distribution
window.image( img.as(u8) );
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
<img src="forge_viz/color_noise.png" alt="Forge image plot of color noise" width="20%" />
Notice Forge automatically handles any af::array type passed from Arrayfire. In the first example we passed in an image of floats in the range [0, 1]. In the last example we cast our array to an unsigned byte array with the range [0, 255]. The type-handling properties are consistent for all Forge drawing functions.



###plot
The plot() function visualizes an array as a 2d-line plot. Let's see a simple example:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array X = seq(-af::Pi, af::Pi, 0.01);
array Y = sin(X);
window.plot(X, Y);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

<img src="forge_viz/sin_plot.png" alt="Forge 2d line plot of sin() function" width="30%" />
The plot function has the signature:  

**void plot( const array &X, const array &Y, const char *const title = NULL 
         );**  

Both the x and y coordinates of the points are required to plot. This allows for non-uniform, or parametric plots:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array t = seq(0, 100, 0.01);
array X = sin(t) * (exp(cos(t)) - 2*cos(4*t) - pow(sin(t/12), 5));
array Y = cos(t) * (exp(cos(t)) - 2*cos(4*t) - pow(sin(t/12), 5));
window.plot(X, Y);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

<img src="forge_viz/butterfly_plot.png" alt="Forge 2d line plot of butterfly function" width="30%" />

###plot3
The plot3() function will plot a curve in 3d-space. 
Its signature is:
**void plot3 (const array &in, const char * title = NULL);**

The input array expects xyz-triplets in sequential order. The points can be in a flattened one dimensional <i>(3n x 1)</i> array, or in one of <i>(3 x n),</i> <i>(n x 3)</i> matrix forms.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array Z = seq(0.1f, 10.f, 0.01);
array Y = sin(10*Z) / Z;
array X = cos(10*Z) / Z;

array Pts = join(1, X, Y, Z);
//Pts can be passed in as a matrix in the from n x 3, 3 x n
//or in the flattened xyz-triplet array with size 3n x 1
window.plot3(Pts);
//both of the following are equally valid
//window.plot3(transpose(Pts));
//window.plot3(flat(Pts));
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
<img src="forge_viz/spiral_plot3.png" alt="Forge 3d line plot" width="40%" />


###surface
The surface() function will plot af::arrays as a 3d surface.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array Z = randu(21, 21);
//window.surface(Z, "Random Surface");    //equal to next function call
window.surface( seq(-1, 1, 0.1), seq(-1, 1, 0.1), Z, "Random Surface");
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
<img src="forge_viz/rand_surface.png" alt="Forge random surface plot" width="30%" />
There are two overloads for the **surface()** function:  

* **void surface (const array & S, const char *const title )** -- accepts a 2d matrix with the z values of the surface  

* **void surface (const array &xVals, const array &yVals, const array &S, const char *const title)** -- accepts additional vectors that define the x,y coordinates for the surface points.  

The second overload has two options for the x, y coordinate vectors. Assuming a surface grid of size **m x n**:
 1. Short vectors defining the spacing along each axis. Vectors will have sizes **m x 1** and **n x 1**.
 2. Vectors containing the coordinates of each and every point. Each of the vectors will have length **mn x 1**. This can be used for completely non-uniform or parametric surfaces. 

