#pragma once
#include <vector>
#include <opencv2/core/core.hpp>
#include <arrayfire.h>

namespace afcv
{

/*
 * opencv -> arrayfire
 */
// conversion for cv::Mat
af::array array(const std::vector<cv::Mat>& input, bool transpose = true);
af::array array(const cv::Mat& input, bool transpose = true);

/*
 * arrayfire -> opencv
 */
// af::array conversion to cv::Mat
cv::Mat toMat(const af::array& input, int type = CV_32F, bool transpose = true);


//helper conversion function
void copy_data(Mat& input, af::array& output, bool transpose) {
    const unsigned w = input.cols;
    const unsigned h = input.rows;
    const unsigned nchannels = input.channels();
    if (nchannels > 4) { throw std::runtime_error(string("mat to array error ")); }
    if (nchannels == 1) {
        // bw, guaranteed to be continuous
        if (transpose) {
            output = af::array(w, h, input.ptr<float>(0)).T();
        } else {
            output = af::array(w, h, input.ptr<float>(0));
        }
    } else {
        if(input.isContinuous()) {
            vector<Mat> channels; split(input, channels);
            size_t dims = channels.size();

            if (transpose) {
                output = af::array(h, w, dims);
                for(unsigned d=0; d<dims; ++d) {
                    output(span, span, d) = af::array(w, h, channels[d].ptr<float>(0)).T();
                }
            } else {
                output = af::array(w, h, dims);
                for(unsigned d=0; d<dims; ++d) {
                    output(span, span, d) = af::array(w, h, channels[d].ptr<float>(0));
                }
            }
        } else {
            throw std::runtime_error(string("mat to array error "));
        }
    }
}


/*
 * opencv -> arrayfire
 */

// conversion for cv::Mat
af::array array(const cv::Mat& input, bool transpose)
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

af::array array(const std::vector<cv::Mat>& input, bool transpose)
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
                throw std::runtime_error("number of OpenCV matrix channels inconsistent");
            af::array tmp = array(input[i], transpose);
            output(span, span, span, i) = tmp;
            //todo: check dimensions?  channels? input.dims()
            //should work...
        }
        return output;
    } catch (const af::exception& ex) {
        throw std::runtime_error(string("std::vector<cv::Mat> to af::array error "));
    }
}


/*
 * arrayfire -> opencv
 */

// af::array conversion to cv::Mat
cv::Mat toMat(const af::array& input, int type, bool transpose)
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
                input_(ii, span, span) = af::moddims(input(span, span, ii), 1, input.dims(1), input.dims(0));
            }
        }
        output = cv::Mat(input.dims(1), input.dims(0), CV_MAKETYPE(type, channels));
    } else {
        if(channels == 1) {
            input_ = input.T();
        } else {
            input_ = af::constant(0, channels, input.dims(0), input.dims(1));
            for(int ii=0; ii<channels; ++ii) {
                input_(ii, span, span) = af::moddims(input(span, span, ii).T(), 1, input.dims(0), input.dims(1));
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
        throw std::runtime_error(string("array to mat error "));
    }
    return output;
}

}
