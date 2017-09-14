#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/more_math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void maximum_kernel(const int N, const Dtype* const X,
                               const Dtype* const Y, Dtype* const Z) {
  CUDA_KERNEL_LOOP(index, N) {
    if (X[index] > Y[index])
      Z[index] = X[index];
    else
      Z[index] = Y[index];
  }
}

template <typename Dtype>
void caffe_gpu_maximum(const int N, const Dtype* const X, const Dtype* const Y,
                       Dtype* const Z) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  maximum_kernel<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
      (N, X, Y, Z);
}

template void caffe_gpu_maximum<int>(const int N, const int* const X,
                                     const int* const Y, int* const Z);
template void caffe_gpu_maximum<float>(const int N, const float* const X,
                                       const float* const Y, float* const Z);
template void caffe_gpu_maximum<double>(const int N, const double* const X,
                                        const double* const Y, double* const Z);

template <>
void caffe_gpu_amax<float>(const int n, const float* x, int* y) {
  CUBLAS_CHECK(cublasIsamax(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_amax<double>(const int n, const double* x, int* y) {
  CUBLAS_CHECK(cublasIdamax(Caffe::cublas_handle(), n, x, 1, y));
}

template <typename Dtype>
__global__ void or_kernel(const int N, const Dtype* const x,
                          const Dtype* const y, Dtype* const z) {
  CUDA_KERNEL_LOOP(index, N) {
    if (x[index] == 1 || y[index] == 1)
      z[index] = 1;
    else
      z[index] = 0;
  }
}

template <typename Dtype>
void caffe_gpu_or(const int N, const Dtype* const x, const Dtype* const y,
                  Dtype* const z) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  or_kernel<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
      (N, x, y, z);
}

template void caffe_gpu_or<int>(const int N, const int* const x,
                                const int* const y, int* z);
template void caffe_gpu_or<float>(const int N, const float* const x,
                                  const float* const y, float* z);
template void caffe_gpu_or<double>(const int N, const double* const x,
                                   const double* const y, double* z);

template <typename Dtype>
__global__ void ceil_kernel(const int N, Dtype* const x) {
  CUDA_KERNEL_LOOP(index, N) { x[index] = ceil(x[index]); }
}

template <typename Dtype>
void caffe_gpu_ceil(const int N, Dtype* const x) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  ceil_kernel<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>> (N, x);
}

template void caffe_gpu_ceil<float>(const int N, float* const x);
template void caffe_gpu_ceil<double>(const int N, double* const x);

template <typename Dtype>
__global__ void floor_kernel(const int n, Dtype* const x) {
  CUDA_KERNEL_LOOP(index, n) { x[index] = floor(x[index]); }
}

template <typename Dtype>
void caffe_gpu_floor(const int N, Dtype* const x) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  floor_kernel<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>> (N, x);
}

template void caffe_gpu_floor<float>(const int n, float* const x);
template void caffe_gpu_floor<double>(const int n, double* const x);

template <typename Dtype>
__global__ void without_kernel(const int N, Dtype* const x, const Dtype without,
                               const Dtype replace) {
  CUDA_KERNEL_LOOP(index, N) {
    if (x[index] == without) x[index] = replace;
  }
}

template <typename Dtype>
void caffe_gpu_without(const int N, Dtype* const x, const Dtype without,
                       const Dtype replace) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  without_kernel<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
      (N, x, without, replace);
}

template void caffe_gpu_without<float>(const int N, float* const x,
                                       const float without,
                                       const float replace);
template void caffe_gpu_without<double>(const int N, double* const x,
                                        const double without,
                                        const double replace);

template <typename Dtype>
__global__ void threshold_min_kernel(const int N, const Dtype* const x,
                                     Dtype* const y, const Dtype threshold,
                                     const Dtype replace) {
  CUDA_KERNEL_LOOP(index, N) {
    if (x[index] < threshold)
      y[index] = replace;
    else
      y[index] = x[index];
  }
}

template <typename Dtype>
__global__ void threshold_max_kernel(const int N, const Dtype* const x,
                                     Dtype* const y, const Dtype threshold,
                                     const Dtype replace) {
  CUDA_KERNEL_LOOP(index, N) {
    if (x[index] > threshold)
      y[index] = replace;
    else
      y[index] = x[index];
  }
}

template <typename Dtype>
void caffe_gpu_threshold(const int N, const Dtype* const x, Dtype* const y,
                         const Dtype threshold, const Dtype replace,
                         const bool for_max) {
  if (for_max) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    threshold_max_kernel<Dtype> << <CAFFE_GET_BLOCKS(N),
                                    CAFFE_CUDA_NUM_THREADS>>>
        (N, x, y, threshold, replace);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    threshold_min_kernel<Dtype> << <CAFFE_GET_BLOCKS(N),
                                    CAFFE_CUDA_NUM_THREADS>>>
        (N, x, y, threshold, replace);
  }
}

template void caffe_gpu_threshold<float>(const int N, const float* const x,
                                         float* const y, const float threshold,
                                         const float replace,
                                         const bool for_max);
template void caffe_gpu_threshold<double>(const int N, const double* const x,
                                          double* const y,
                                          const double threshold,
                                          const double replace,
                                          const bool for_max);

template <typename Dtype>
__global__ void binary_kernel(const int N, const Dtype* const x, Dtype* const y,
                              const Dtype threshold) {
  CUDA_KERNEL_LOOP(index, N) {
    if (x[index] >= threshold)
      y[index] = 1;
    else
      y[index] = 0;
  }
}

template <typename Dtype>
void caffe_gpu_binary(const int N, const Dtype* const x, Dtype* const y,
                      const Dtype threshold) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  binary_kernel<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
      (N, x, y, threshold);
}

template void caffe_gpu_binary<float>(const int N, const float* const x,
                                      float* const y, const float threshold);
template void caffe_gpu_binary<double>(const int N, const double* const x,
                                       double* const y, const double threshold);

}  // namespace caffe
