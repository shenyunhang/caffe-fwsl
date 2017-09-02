#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/roi_score_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RoIScorePoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  threshold_ = this->layer_param_.roi_score_pooling_param().threshold();
  LOG(INFO) << "----------------------------------------------";
  LOG(INFO) << "current only support one image per forward-backward.";
  CHECK_EQ(bottom.size(), 2) << "Pooling method need two input blob: "
                                "pool_size(X*1*1*1) and feature(N*C*H*W).";

  switch (this->layer_param_.roi_score_pooling_param().pool()) {
    case RoIScorePoolingParameter_PoolMethod_SUM:
      LOG(INFO) << "Using SUM pooling method.";
      break;
    case RoIScorePoolingParameter_PoolMethod_MAX:
      LOG(INFO) << "Using MAX pooling method.";
      break;
    case RoIScorePoolingParameter_PoolMethod_TSUM:
      LOG(INFO) << "Using threshold SUM pooling method.";
      LOG(INFO) << "threshold_: " << threshold_;
      break;
    case RoIScorePoolingParameter_PoolMethod_TAVEMAX:
      LOG(INFO) << "Using threshold AVE and MAX pooling method.";
      LOG(INFO) << "threshold_: " << threshold_;
      break;
    case RoIScorePoolingParameter_PoolMethod_MUL:
      LOG(INFO) << "Using MUL pooling method.";
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
  LOG(INFO) << "----------------------------------------------";

  pooling_axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.roi_score_pooling_param().axis());
}

template <typename Dtype>
void RoIScorePoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  num_roi_ = bottom[0]->num();
  num_class_ = bottom[0]->channels();
  num_img_ = bottom[1]->count();

  outer_num_ = bottom[0]->count(0, pooling_axis_);
  inner_num_ = bottom[0]->count(pooling_axis_ + 1);
  channels_ = bottom[0]->shape(pooling_axis_);
  dim_ = bottom[0]->count() / outer_num_;

  vector<int> top_dims = bottom[0]->shape();
  top_dims[pooling_axis_] = num_img_;
  top[0]->Reshape(top_dims);
  mask_idx_.Reshape(top_dims);
}

template <typename Dtype>
void RoIScorePoolingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // score	#roi	#class	1	1
  // number	X	1	1	1

  CHECK_EQ(bottom[0]->shape(pooling_axis_), bottom[1]->asum_data())
      << "size of score blob along pooling channel and sum of index blob must "
         "be the same.";

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* num_data = bottom[1]->cpu_data();

  const int top_count = top[0]->count();

  Dtype* top_data = top[0]->mutable_cpu_data();
  int* mask = mask_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, mask);

  for (int i = 0; i < num_img_; ++i) {
    // LOG(INFO)<<"RoI num: "<<num_data[i];
    CHECK_GT(num_data[i], 0) << "Need at least one RoI per image.";
  }

  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.roi_score_pooling_param().pool()) {
    case RoIScorePoolingParameter_PoolMethod_SUM: {
      // Initialize
      caffe_set(top_count, Dtype(0), top_data);

      // The main loop
      for (int i = 0; i < outer_num_; ++i) {
        for (int k = 0; k < inner_num_; ++k) {
          int idx_img = 0;
          int idx_img_roi = 0;
          for (int j = 0; j < channels_; ++j) {
            int pool_index = (i * num_img_ + idx_img) * inner_num_ + k;
            int index = i * dim_ + j * inner_num_ + k;
            top_data[pool_index] += bottom_data[index];

            idx_img_roi++;
            if (idx_img_roi == num_data[idx_img]) {
              idx_img++;
              idx_img_roi = 0;
            }
          }
        }
      }
    } break;
    case RoIScorePoolingParameter_PoolMethod_MAX: {
      // Initialize
      caffe_set(top_count, Dtype(-FLT_MAX), top_data);

      for (int i = 0; i < outer_num_; ++i) {
        for (int k = 0; k < inner_num_; ++k) {
          int idx_img = 0;
          int idx_img_roi = 0;
          Dtype max_value = -FLT_MAX;
          int max_value_index = -1;
          for (int j = 0; j < channels_; ++j) {
            const int index = i * dim_ + j * inner_num_ + k;
            Dtype in = bottom_data[index];
            if (in > max_value) {
              max_value = in;
              max_value_index = index;
            }

            idx_img_roi++;
            if (idx_img_roi == num_data[idx_img]) {
              CHECK_NE(max_value, Dtype(-FLT_MAX)) << "max value not found.";
              const int pool_index = (i * num_img_ + idx_img) * inner_num_ + k;
              top_data[pool_index] = max_value;
              mask[pool_index] = max_value_index;

              idx_img++;
              idx_img_roi = 0;
              max_value = -FLT_MAX;
              max_value_index = -1;
            }
          }
        }
      }
    } break;
    case RoIScorePoolingParameter_PoolMethod_TSUM: {
      // Initialize
      caffe_set(top_count, Dtype(0), top_data);

      // The main loop
      for (int i = 0; i < outer_num_; ++i) {
        for (int k = 0; k < inner_num_; ++k) {
          int idx_img = 0;
          int idx_img_roi = 0;
          int pool_size = 0;
          Dtype all_sum = 0;
          for (int j = 0; j < channels_; ++j) {
            const int index = i * dim_ + j * inner_num_ + k;
            const int pool_index = (i * num_img_ + idx_img) * inner_num_ + k;

            const Dtype in = bottom_data[index];
            all_sum += in;
            if (in < threshold_) {
            } else {
              top_data[pool_index] += in;
              pool_size++;
            }
            idx_img_roi++;

            if (idx_img_roi == num_data[idx_img]) {
              if (pool_size == 0) {
                top_data[pool_index] = all_sum;
              }
              mask[pool_index] = pool_size;

              idx_img++;
              idx_img_roi = 0;
              pool_size = 0;
              all_sum = 0;
            }
          }
        }
      }
    } break;
    case RoIScorePoolingParameter_PoolMethod_TAVEMAX: {
      // Initialize
      caffe_set(top_count, Dtype(0), top_data);

      for (int i = 0; i < outer_num_; ++i) {
        for (int k = 0; k < outer_num_; ++k) {
          int idx_img = 0;
          int idx_img_roi = 0;
          int pool_size = 0;
          Dtype max_value = -FLT_MAX;
          int max_value_index = -1;
          for (int j = 0; j < channels_; ++j) {
            const int pool_index = (i * num_img_ + idx_img) * inner_num_ + k;
            const int index = i * dim_ + j * inner_num_ + k;
            const Dtype in = bottom_data[index];
            if (in > max_value) {
              max_value = in;
              max_value_index = index;
            }
            if (in < threshold_) {
            } else {
              top_data[pool_index] += in;
              pool_size++;
            }

            idx_img_roi++;
            if (idx_img_roi == num_data[idx_img]) {
              if (pool_size == 0) {
                top_data[pool_index] = max_value;
                mask[pool_index] = max_value_index;
                mask[pool_index] *= -1;
              } else {
                top_data[pool_index] /= pool_size;
                mask[pool_index] = pool_size;
              }

              idx_img++;
              idx_img_roi = 0;
              pool_size = 0;
              max_value = -FLT_MAX;
              max_value_index = -1;
            }
          }
        }
      }
    } break;
    case RoIScorePoolingParameter_PoolMethod_MUL:
      NOT_IMPLEMENTED;
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void RoIScorePoolingLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // score	#roi	#class	1	1
  // index	#roi	1	1	1
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* num_data = bottom[1]->cpu_data();
  const int* mask = mask_idx_.cpu_data();

  // Initialize
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  // Different pooling methods. We explicitly do the switch outside the for loop
  // to save time, although this results in more codes.
  switch (this->layer_param_.roi_score_pooling_param().pool()) {
    case RoIScorePoolingParameter_PoolMethod_SUM: {
      for (int i = 0; i < outer_num_; ++i) {
        for (int k = 0; k < inner_num_; ++k) {
          int idx_img = 0;
          int idx_img_roi = 0;
          for (int j = 0; j < channels_; ++j) {
            const int pool_index = (i * num_img_ + idx_img) * inner_num_ + k;
            const int index = i * dim_ + j * inner_num_ + k;
            bottom_diff[index] = top_diff[pool_index];

            idx_img_roi++;
            if (idx_img_roi == num_data[idx_img]) {
              idx_img++;
              idx_img_roi = 0;
            }
          }
        }
      }
    } break;
    case RoIScorePoolingParameter_PoolMethod_MAX: {
      for (int i = 0; i < outer_num_; ++i) {
        for (int k = 0; k < inner_num_; ++k) {
          for (int idx_img = 0; idx_img < num_img_; idx_img++) {
            const int pool_index = (i * num_img_ + idx_img) * inner_num_ + k;
            const int bottom_index = mask[pool_index];
            bottom_diff[bottom_index] = top_diff[pool_index];
          }
        }
      }
    } break;
    case RoIScorePoolingParameter_PoolMethod_TSUM: {
      // The main loop
      for (int i = 0; i < outer_num_; ++i) {
        for (int k = 0; k < inner_num_; ++k) {
          int idx_img = 0;
          int idx_img_roi = 0;
          for (int j = 0; j < channels_; ++j) {
            const int pool_index = (i * num_img_ + idx_img) * inner_num_ + k;
            const int index = i * dim_ + j * inner_num_ + k;
            const int pool_size = mask[pool_index];

            if (pool_size == 0) {
              bottom_diff[index] = top_diff[pool_index];
            } else {
              if (bottom_data[index] < threshold_) {
              } else {
                bottom_diff[index] = top_diff[pool_index];
              }
            }

            idx_img_roi++;
            if (idx_img_roi == num_data[idx_img]) {
              idx_img++;
              idx_img_roi = 0;
            }
          }
        }
      }
    } break;
    case RoIScorePoolingParameter_PoolMethod_TAVEMAX: {
      for (int i = 0; i < outer_num_; ++i) {
        for (int k = 0; k < inner_num_; ++k) {
          int idx_img = 0;
          int idx_img_roi = 0;
          for (int j = 0; j < channels_; ++j) {
            const int pool_index = (i * num_img_ + idx_img) * inner_num_ + k;
            const int index = i * dim_ + j * inner_num_ + k;
            const int pool_size = mask[pool_index];

            if (pool_size > 0) {
              if (bottom_data[index] < threshold_) {
              } else {
                bottom_diff[index] = top_diff[pool_index] / pool_size;
              }
            } else {
              const int bottom_index = -1 * pool_size;
              bottom_diff[bottom_index] = top_diff[pool_index];
            }

            idx_img_roi++;
            if (idx_img_roi == num_data[idx_img]) {
              idx_img++;
              idx_img_roi = 0;
            }
          }
        }
      }
    } break;
    case RoIScorePoolingParameter_PoolMethod_MUL:
      NOT_IMPLEMENTED;
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
}

//#ifdef CPU_ONLY
// STUB_GPU(RoIScorePoolingLayer);
//#endif

INSTANTIATE_CLASS(RoIScorePoolingLayer);
REGISTER_LAYER_CLASS(RoIScorePooling);

}  // namespace caffe
