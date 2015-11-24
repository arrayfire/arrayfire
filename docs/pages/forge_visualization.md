##Visualizing af::arrays with Forge {#forge_visualization}
Arrayfire as a library aims to provide a robust and easy to use platform for high-performance, parallel and GPU computing. The goal of Forge, an OpenGL visualization library, is to provide equally robust visualizations that are interoperable between Arrayfire data-structures and an OpenGL context. Instead of wasting time copying and reformatting data from the GPU to the host and back to the GPU, we can draw directly from GPU-data to GPU-framebuffers! Furthermore, Arrayfire provides wrapper functions that handle all of the interoperability for us and leave us with a simple interface to visualize af::arrays. Let's see exactly what visuals we can illuminate with forge and how Arrayfire anneals the data between the two libraries.  

###Setup
Before we can call Forge functions, we need to set up the related "canvas" classes. Forge functions are tied to the af::Window class. First let's create a window:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
const static int WIDTH = 512, HEIGHT = 512;
af::Window window(WIDTH, HEIGHT, "2D plot example title");

do{

//drawing functions here

} while( !window.close() );
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
We also added a drawing loop, so now we can use Forge's drawing functions to draw to the window.
The drawing functions present in Forge are listed below.

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
<br> **void plot( const array &X, const array &Y, const char * const title = NULL );**  
<br> Both the x and y coordinates of the points are required to plot. This allows for non-uniform, or parametric plots:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array t = seq(0, 100, 0.01);
array X = sin(t) * (exp(cos(t)) - 2*cos(4*t) - pow(sin(t/12), 5));
array Y = cos(t) * (exp(cos(t)) - 2*cos(4*t) - pow(sin(t/12), 5));
window.plot(X, Y);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

<img src="forge_viz/butterfly_plot.png" alt="Forge 2d line plot of butterfly function" width="30%" />

###scatter
The scatter() function visualizes an array as a 2d-scatter plot. Let's see a example:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array X = seq(-af::Pi, af::Pi, 0.01);
array noise = randn(X.dims(0)) /3.0f;
array Y = sin(X) + noise;
window.scatter(X, Y);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
<img src="forge_viz/sin_scatter.png" alt="Forge 2d scatter plot" width="40%" />

The scatter function has the signature:  
<br> **void scatter( const array &X, const array &Y, af::markerType marker=AF\_MARKER\_POINT, const char * const title = NULL);**  
<br> The __af::markerType__ enum determines which marker will be drawn at each point. It can take on the values of:  

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
enum af::markerType {
    AF_MARKER_POINT
    AF_MARKER_CIRCLE
    AF_MARKER_SQUARE
    AF_MARKER_TRIANGLE
    AF_MARKER_CROSS
    AF_MARKER_PLUS
    AF_MARKER_STAR
};
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
<img src="forge_viz/marker_types.png" alt="Different marker types in forge" width="100%" />

###plot3
The plot3() function will plot a curve in 3d-space. 
Its signature is:  
<br> **void plot3 (const array &in, const char * title = NULL);**
<br> The input array expects xyz-triplets in sequential order. The points can be in a flattened one dimensional <i>(3n x 1)</i> array, or in one of the <i>(3 x n),</i> <i>(n x 3)</i> matrix forms.
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

###hist
The hist() function renders an input array as a histogram. In our example, the input array will be created with Arrayfire's histogram() function, which actually counts and bins each sample. The output from histogram() can directly be fed into the hist() rendering function.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
const int BINS = 128; SAMPLES = 9162;
array norm = randn(SAMPLES);
array hist_arr = histogram(norm, BINS);

win.hist(hist_arr, 0, BINS);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In addition to the histogram array with the number of samples in each bin, the hist() function takes two additional parameters -- the minimum and maximum values of all datapoints in the histogram array. This effectively sets the range of the binned data. The full signature of hist() is:<br>  
**void hist(const array & X, const double minval, const double maxval, const char * const title = NULL);**
<img src="forge_viz/norm_histogram.png" alt="Forge 3d scatter plot" width="40%" />


###scatter3
The scatter3() function visualizes an array as a 3d-scatter plot. 
Its signature is:
<br> **void scatter3 (const array &in, af::markerType marker=AF _ MARKER _ POINT, const char * title = NULL);**
<br> The input array expects xyz-triplets in sequential order. The points can be in a flattened one dimensional <i>(3n x 1)</i> array, or in one of the <i>(3 x n),</i> <i>(n x 3)</i> matrix forms.

<img src="forge_viz/norm_scatter3.png" alt="Forge 3d scatter plot" width="40%" />
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array X = randn(300);
array Y = randn(300);
array Z = randn(300);

array Pts = join(1, X, Y, Z);
window.scatter3(Pts);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The __af::markerType__ enum determines which marker will be drawn at each point. It can take on the values of:  

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
enum af::markerType {
    AF_MARKER_POINT
    AF_MARKER_CIRCLE
    AF_MARKER_SQUARE
    AF_MARKER_TRIANGLE
    AF_MARKER_CROSS
    AF_MARKER_PLUS
    AF_MARKER_STAR
};
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
<img src="forge_viz/marker_types.png" alt="Different marker types in forge" width="100%" />

###surface
The surface() function will plot af::arrays as a 3d surface.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array Z = randu(21, 21);
window.surface(Z, "Random Surface");    //equal to next function call
//window.surface( seq(-1, 1, 0.1), seq(-1, 1, 0.1), Z, "Random Surface");
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
<img src="forge_viz/rand_surface.png" alt="Forge random surface plot" width="30%" />
There are two overloads for the **surface()** function:  
* **void surface (const array & S, const char *const title )** -- accepts a 2d matrix with the z values of the surface  
* **void surface (const array &xVals, const array &yVals, const array &S, const char * const title)** -- accepts additional vectors that define the x,y coordinates for the surface points.  

The second overload has two options for the x, y coordinate vectors. Assuming a surface grid of size **m x n**:
 1. Short vectors defining the spacing along each axis. Vectors will have sizes **m x 1** and **n x 1**.
 2. Vectors containing the coordinates of each and every point. Each of the vectors will have length **mn x 1**. This can be used for completely non-uniform or parametric surfaces. 

###Conclusion
There is a fairly comprehensive collection of methods to visualize data in Arrayfire. Thanks to the high-performance gpu plotting library Forge, the provided Arrayfire functions not only make visualizations as simple as possible, but keep them as robust as the rest of the Arrayfire library.
