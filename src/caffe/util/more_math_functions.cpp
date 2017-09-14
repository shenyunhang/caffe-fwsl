#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/more_math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
Dtype caffe_cpu_max_element(const int N, const Dtype *x) {
  Dtype max_val = (*x);
  for (int i = 1; i < N; ++i) {
    ++x;
    if (max_val < (*x)) {
      max_val = (*x);
    }
  }
  return max_val;
}

template int caffe_cpu_max_element<int>(const int N, const int *x);
template float caffe_cpu_max_element<float>(const int N, const float *x);
template double caffe_cpu_max_element<double>(const int N, const double *x);

template <typename Dtype>
Dtype caffe_cpu_sum(const int N, const Dtype *x) {
  Dtype sum = (*x);
  for (int i = 1; i < N; ++i) {
    ++x;
    sum += (*x);
  }
  return sum;
}

template int caffe_cpu_sum<int>(const int N, const int *x);
template float caffe_cpu_sum<float>(const int N, const float *x);
template double caffe_cpu_sum<double>(const int N, const double *x);

/**
Warning: the diff of cpg_blob will be modified!
 **/
template <typename Dtype>
int caffe_cpu_threshold_bbox(Blob<Dtype> *cpg_blob, Blob<Dtype> *bboxes_blob,
                             const float fg_threshold, const int gt_label) {
  // CPG should be in the cpg_blob->gpu_data()
  // CPG:	1	1	heigh_im_	width_im
  caffe_set(bboxes_blob->count(), static_cast<Dtype>(-1),
            bboxes_blob->mutable_cpu_data());

  const int num = cpg_blob->shape(0);
  const int channels = cpg_blob->shape(1);
  const int height = cpg_blob->shape(2);
  const int width = cpg_blob->shape(3);
  CHECK_EQ(num, 1) << "we assume num equal 1";
  CHECK_EQ(channels, 1) << "we assume num equal 1";

  const int num_bboxes = bboxes_blob->shape(0);

  // caffe_copy(cpg_blob->count(), cpg_blob->gpu_data(),
  // cpg_blob->mutable_gpu_diff());
  memcpy(cpg_blob->mutable_cpu_diff(), cpg_blob->cpu_data(),
         cpg_blob->count() * sizeof(Dtype));

  // get maximum value
  // SYNC
  const Dtype *cpg_cpu = cpg_blob->cpu_diff();
  const Dtype max_value = caffe_cpu_max_element(cpg_blob->count(), cpg_cpu);
  const Dtype threshold = fg_threshold * (max_value);
  CHECK_GT(max_value, 0) << "max_value should be greater than 0. #cpg_cpg: "
                         << cpg_blob->count();
  /*LOG_IF(INFO, debug_info_) << "max_value: " << (max_value)*/
  /*<< " threshold: " << threshold;*/

  // id x1 y1 x2 y2
  int bbox_num = 0;
  bool is_trace = false;
  Dtype *bbox = bboxes_blob->mutable_cpu_data();
  const Dtype separate = std::min(height, width) / 5;

  Dtype *cpg_mcpu = cpg_blob->mutable_cpu_diff();
  while (true) {
    for (int n = 0; n < num; ++n) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          for (int c = 0; c < channels; ++c) {
            int index = cpg_blob->offset(n, c, h, w);
            if (cpg_mcpu[index] < threshold && cpg_mcpu[index] > -threshold)
              continue;
            if (is_trace && (w - bbox[2] > separate || h - bbox[3] > separate ||
                             bbox[0] - w > separate || bbox[1] - h > separate))
              continue;
            cpg_mcpu[index] = 0;
            if (!is_trace) {
              bbox[0] = width;
              bbox[1] = height;
              bbox[2] = 0;
              bbox[3] = 0;
              is_trace = true;
            }
            if (w < bbox[0]) bbox[0] = w;
            if (h < bbox[1]) bbox[1] = h;
            if (w > bbox[2]) bbox[2] = w;
            if (h > bbox[3]) bbox[3] = h;
          }
        }
      }
    }
    if (!is_trace) break;
    bbox_num++;
    if (bbox_num == num_bboxes) break;
    bbox += bboxes_blob->offset(1);
    is_trace = false;
  }

  /*if (debug_info_) {*/
  /*for (int i = 0; i < max_bb_per_cls_; ++i) {*/
  /*bbox = bboxes_->mutable_cpu_data() + bboxes_->offset(i);*/
  /*if (bbox[0] == -1) break;*/
  /*LOG(INFO) << "bbox label x1 y1 x2 y2: " << gt_label << " " << bbox[0]*/
  /*<< " " << bbox[1] << " " << bbox[2] << " " << bbox[3];*/
  /*}*/
  /*}*/

  return bbox_num;
}

template int caffe_cpu_threshold_bbox<float>(Blob<float> *cpg_blob,
                                             Blob<float> *bboxes_blob,
                                             const float fg_threshold,
                                             const int gt_label);
template int caffe_cpu_threshold_bbox<double>(Blob<double> *cpg_blob,
                                              Blob<double> *bboxes_blob,
                                              const float fg_threshold,
                                              const int gt_label);

}  // namespace caffe
