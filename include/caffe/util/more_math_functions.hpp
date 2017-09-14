#ifndef CAFFE_UTIL_MORE_MATH_FUNCTION_H_
#define CAFFE_UTIL_MORE_MATH_FUNCTION_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

template <typename Dtype>
Dtype caffe_cpu_max_element(const int n, const Dtype* x);

template <typename Dtype>
Dtype caffe_cpu_sum(const int n, const Dtype* x);

template <typename Dtype>
int caffe_cpu_threshold_bbox(Blob<Dtype>* cpg_blob, Blob<Dtype>* bboxes_blob,
                             const float fg_threshold, const int gt_label);

#ifndef CPU_ONLY  // GPU

template <typename Dtype>
void caffe_gpu_maximum(const int N, const Dtype* const X, const Dtype* const Y,
                       Dtype* const Z);

// This function finds the (smallest) index of the element of the maximum
// magnitude.
// Notice that the last equation reflects 1-based indexing used for
// compatibility with Fortran.
//
// Read more at: http://docs.nvidia.com/cuda/cublas/index.html#ixzz4YuGQxBLf
// Follow us: @GPUComputing on Twitter | NVIDIA on Facebook
template <typename Dtype>
void caffe_gpu_amax(const int n, const Dtype* x, int* y);

template <typename Dtype>
void caffe_gpu_or(const int N, const Dtype* const x, const Dtype* const y,
                  Dtype* const z);

template <typename Dtype>
void caffe_gpu_ceil(const int N, Dtype* const x);

template <typename Dtype>
void caffe_gpu_floor(const int N, Dtype* const x);

template <typename Dtype>
void caffe_gpu_without(const int N, Dtype* const x, const Dtype without,
                       const Dtype replace);

template <typename Dtype>
void caffe_gpu_threshold(const int N, const Dtype* const x, Dtype* const y,
                         const Dtype threshold, const Dtype replace,
                         const bool for_max);

template <typename Dtype>
void caffe_gpu_binary(const int N, const Dtype* const x, Dtype* const y,
                      const Dtype threshold);

#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_MORE_MATH_FUNCTIONS_H_
