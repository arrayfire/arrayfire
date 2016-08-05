#pragma once
#include <vector>
#include <opencv2/core/core.hpp>
#include <arrayfire.h>

namespace afcv
{

/*
 * opencv -> arrayfire
 */
/**
    \ingroup opencv_interop
    @{
*/
/**
    This function will copy multiple cv::Mat from a std::vector<cv::Mat> to an \ref af::array

    \param[in]  input The data which will be loaded into the array
    \param[in]  transpose flag specifying if each Mat should be transposed during copy

    \note The number of channels of each cv::Mat should be the same for the whole set

    \returns an af::array with the contents of all cv::Mat in the std::vector
*/
static af::array array(const std::vector<cv::Mat>& input, bool transpose = true);

/**
   This function will copy data from a cv::Mat to an \ref af::array

   \param[in]  input The data which will be loaded into the array
   \param[in]  transpose flag specifying if the input should be transposed during copy

   \returns an af::array with the contents of the OpenCv Mat
*/
static af::array array(const cv::Mat& input, bool transpose = true);

/*
 * arrayfire -> opencv
 */
/**
   This function will copy data from an \ref af::array to a cv::Mat

   \param[in]  input The af::array from which data which will be loaded into cv::Mat
   \param[in]  type The resulting datatype of the output cv::Mat
   \param[in]  transpose flag specifying if the input should be transposed during copy

   \returns an cv::Mat with the contents of the input af::array
*/
static cv::Mat toMat(const af::array& input, int type = CV_32F, bool transpose = true);
/**
  @}
*/

//helper conversion function
static void copy_data(cv::Mat& input, af::array& output, bool transpose) {
    const unsigned w = input.cols;
    const unsigned h = input.rows;
    const unsigned nchannels = input.channels();
    if (nchannels > 4) { throw af::exception("mat to array error "); }
    if (nchannels == 1) {
        // bw, guaranteed to be continuous
        if (transpose) {
            output = af::array(w, h, input.ptr<float>(0)).T();
        } else {
            output = af::array(w, h, input.ptr<float>(0));
        }
    } else {
        if(input.isContinuous()) {
            std::vector<cv::Mat> channels; split(input, channels);
            size_t dims = channels.size();

            if (transpose) {
                output = af::array(h, w, dims);
                for(unsigned d=0; d<dims; ++d) {
                    output(af::span, af::span, d) = af::array(w, h, channels[d].ptr<float>(0)).T();
                }
            } else {
                output = af::array(w, h, dims);
                for(unsigned d=0; d<dims; ++d) {
                    output(af::span, af::span, d) = af::array(w, h, channels[d].ptr<float>(0));
                }
            }
        } else {
            throw af::exception("mat to array error ");
        }
    }
}


/*
 * opencv -> arrayfire
 */

// conversion for cv::Mat
static af::array array(const cv::Mat& input, bool transpose)
{
    af::array output;
    if (input.empty()) { return output; }
    cv::Mat tmp;
    if (input.channels() == 1)
        { input.convertTo(tmp, CV_32F); }
    else if (input.channels() == 2)
        { input.convertTo(tmp, CV_32FC2); }
    else if (input.channels() == 3)
        { input.convertTo(tmp, CV_32FC3); }
    else if (input.channels() == 4)
        { input.convertTo(tmp, CV_32FC4); }

    copy_data(tmp, output, transpose);
    return output;
}

static af::array array(const std::vector<cv::Mat>& input, bool transpose)
{
    af::array output;
    try {
        int h = input[0].rows;
        int w = input[0].cols;
        int c = input[0].channels();
        if (transpose) {
            output = af::array(h, w, c, input.size());
        } else {
            output = af::array(w, h, c, input.size());
        }
        for (unsigned i = 0; i < input.size(); i++) {
            if(input[i].channels() != c)
                throw af::exception("number of OpenCV matrix channels inconsistent");
            af::array tmp = array(input[i], transpose);
            output(af::span, af::span, af::span, i) = tmp;
        }
        return output;
    } catch (const af::exception& ex) {
        throw af::exception("std::vector<cv::Mat> to af::array error ");
    }
}


/*
 * arrayfire -> opencv
 */

// af::array conversion to cv::Mat
static cv::Mat toMat(const af::array& input, int type, bool transpose)
{
    cv::Mat output;
    const int channels = input.dims(2);
    int ndims = input.numdims();
    af::array input_;
    if (transpose) {
        if(channels == 1) {
            input_ = input;
        } else {
            input_ = af::constant(0, channels, input.dims(1), input.dims(0));
            for(int ii=0; ii<channels; ++ii) {
                input_(ii, af::span, af::span) = af::moddims(input(af::span, af::span, ii), 1, input.dims(1), input.dims(0));
            }
        }
        output = cv::Mat(input.dims(1), input.dims(0), CV_MAKETYPE(type, channels));
    } else {
        if(channels == 1) {
            input_ = input.T();
        } else {
            input_ = af::constant(0, channels, input.dims(0), input.dims(1));
            for(int ii=0; ii<channels; ++ii) {
                input_(ii, af::span, af::span) = af::moddims(input(af::span, af::span, ii).T(), 1, input.dims(0), input.dims(1));
            }
        }
        output = cv::Mat(input.dims(0), input.dims(1), CV_MAKETYPE(type, channels));
    }

    if (type == CV_32F) {
        float* data = output.ptr<float>(0);
        input_.host((void*)data);
    } else if (type == CV_32S) {
        int* data = output.ptr<int>(0);
        input_.as(s32).host((void*)data);
    } else if (type == CV_64F) {
        double* data = output.ptr<double>(0);
        input_.as(f64).host((void*)data);
    } else if (type == CV_8U) {
        uchar* data = output.ptr<uchar>(0);
        input_.as(b8).host((void*)data);
    } else {
        throw af::exception("array to mat error ");
    }
    return output;
}

}
