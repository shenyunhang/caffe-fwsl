#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/general_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GeneralPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  threshold_ = this->layer_param_.general_pooling_param().threshold();
  LOG(INFO) << "----------------------------------------------";
  LOG(INFO) << "current only support one image per forward-backward.";
  switch (this->layer_param_.general_pooling_param().pool()) {
    case GeneralPoolingParameter_PoolMethod_FMAX:
      CHECK_EQ(bottom.size(), 2)
          << "Using fliter MAX pooling method need two input blob.";
      LOG(INFO) << "Using MAX pooling method.";
      break;
    case GeneralPoolingParameter_PoolMethod_FAVE:
      CHECK_EQ(bottom.size(), 2)
          << "Using fliter AVE pooling method need two input blob.";
      LOG(INFO) << "Using fliter AVE pooling method.";
      break;
    case GeneralPoolingParameter_PoolMethod_SUM:
      LOG(INFO) << "Using SUM pooling method.";
      break;
    case GeneralPoolingParameter_PoolMethod_MAX:
      LOG(INFO) << "Using MAX pooling method.";
      break;
    case GeneralPoolingParameter_PoolMethod_FSUM:
      CHECK_EQ(bottom.size(), 2)
          << "Using fliter SUM pooling method need two input blob.";
      LOG(INFO) << "Using fliter SUM pooling method.";
      break;
    case GeneralPoolingParameter_PoolMethod_TSUM:
      LOG(INFO) << "Using threshold SUM pooling method.";
      LOG(INFO) << "threshold_: " << threshold_;
      break;
    case GeneralPoolingParameter_PoolMethod_FAVEMAX:
      CHECK_EQ(bottom.size(), 2)
          << "Using fliter AVE and MAX pooling method need two input blob.";
      LOG(INFO) << "Using fliter AVE and MAX pooling method.";
      LOG(INFO) << "threshold_: " << threshold_;
      break;
    case GeneralPoolingParameter_PoolMethod_TAVEMAX:
      LOG(INFO) << "Using threshold AVE and MAX pooling method.";
      LOG(INFO) << "threshold_: " << threshold_;
      break;
    case GeneralPoolingParameter_PoolMethod_MUL:
      CHECK_EQ(bottom.size(), 2)
          << "Using fliter MUL pooling method need two input blob.";
      LOG(INFO) << "Using flited NONE pooling method.";
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
  LOG(INFO) << "----------------------------------------------";

  pooling_axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.general_pooling_param().axis());
}

template <typename Dtype>
void GeneralPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 2) {
    CHECK_EQ(bottom[0]->num(), bottom[1]->num())
        << "first and second blob must have the same num.";
    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
        << "first and second blob must have the same channels.";
    CHECK_EQ(bottom[1]->count(), bottom[1]->num() * bottom[1]->channels())
        << "second blob height and width must 1.";
  }

  const int num_class = bottom[0]->channels();
  const int num_roi = bottom[0]->num();

  outer_num_ = bottom[0]->count(0, pooling_axis_);
  inner_num_ = bottom[0]->count(pooling_axis_ + 1);
  channels_ = bottom[0]->shape(pooling_axis_);
  dim_ = bottom[0]->count() / outer_num_;

  switch (this->layer_param_.general_pooling_param().pool()) {
    case GeneralPoolingParameter_PoolMethod_MUL:
      top[0]->Reshape(num_roi, num_class, 1, 1);
      break;
    default:
      vector<int> top_dims = bottom[0]->shape();
      top_dims[pooling_axis_] = 1;
      top[0]->Reshape(top_dims);
  }
  vector<int> mask_dims = bottom[0]->shape();
  mask_dims[pooling_axis_] = 1;
  mask_idx_.Reshape(mask_dims);
}

template <typename Dtype>
void GeneralPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  // score	#roi	#class	1	1
  // fliter	#roi	#class	1	1

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* fliter = NULL;
  if (bottom.size() == 2) fliter = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  const int top_count = top[0]->count();
  const int num_roi = bottom[0]->num();
  const int num_class = bottom[0]->channels();
  const int num_im = 1;

  int* mask = mask_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, mask);

  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.general_pooling_param().pool()) {
    case GeneralPoolingParameter_PoolMethod_FMAX:
      // Initialize
      caffe_set(top_count, Dtype(-FLT_MAX), top_data);

      // The main loop
      for (int n = 0; n < num_im; ++n) {
        for (int c = 0; c < num_class; ++c) {
          const int pool_index = n * num_class + c;
          int pool_size = 0;

          Dtype max_value = -FLT_MAX;
          int max_value_index = -1;

          for (int r = 0; r < num_roi; ++r) {
            const int index = r * num_class + c;
            const Dtype in = bottom_data[index];
            if (in > max_value) {
              max_value = in;
              max_value_index = index;
            }
            if (fliter[index] == Dtype(0)) continue;
            if (in > top_data[pool_index]) {
              top_data[pool_index] = in;
              mask[pool_index] = index;
            }
            pool_size++;
          }
          if (pool_size == 0) {
            top_data[pool_index] = max_value;
            mask[pool_index] = max_value_index;
          }
        }
      }
      break;
    case GeneralPoolingParameter_PoolMethod_FAVE:
      // Initialize
      caffe_set(top_count, Dtype(0), top_data);

      // The main loop
      for (int n = 0; n < num_im; ++n) {
        for (int c = 0; c < num_class; ++c) {
          const int pool_index = n * num_class + c;
          int pool_size = 0;
          Dtype sum_all = 0;

          for (int r = 0; r < num_roi; ++r) {
            const int index = r * num_class + c;
            sum_all += bottom_data[index];
            if (fliter[index] == Dtype(0)) continue;
            top_data[pool_index] += bottom_data[index];
            pool_size++;
          }
          if (pool_size == 0) {
            top_data[pool_index] = sum_all;
            pool_size = num_roi;
          }
          top_data[pool_index] /= pool_size;
          mask[pool_index] = pool_size;
        }
      }
      break;
    case GeneralPoolingParameter_PoolMethod_SUM: {
      // Initialize
      caffe_set(top_count, Dtype(0), top_data);

      for (int i = 0; i < outer_num_; ++i) {
        for (int k = 0; k < inner_num_; ++k) {
          int pool_index = i * inner_num_ + k;
          for (int j = 0; j < channels_; ++j) {
            int index = i * dim_ + j * inner_num_ + k;
            top_data[pool_index] += bottom_data[index];
          }
        }
      }

      // LOG(INFO)<<"top: "<<top[0]->asum_data();

    } break;
    case GeneralPoolingParameter_PoolMethod_MAX: {
      // Initialize
      caffe_set(top_count, Dtype(-FLT_MAX), top_data);

      for (int i = 0; i < outer_num_; ++i) {
        for (int k = 0; k < inner_num_; ++k) {
          int pool_index = i * inner_num_ + k;
          Dtype max_value = -FLT_MAX;
          int max_value_index = -1;
          for (int j = 0; j < channels_; ++j) {
            int index = i * dim_ + j * inner_num_ + k;
            Dtype in = bottom_data[index];
            if (in > max_value) {
              max_value = in;
              max_value_index = index;
            }
          }
          CHECK_NE(max_value, Dtype(-FLT_MAX)) << "max value not found.";
          top_data[pool_index] = max_value;
          mask[pool_index] = max_value_index;
        }
      }

    } break;
    case GeneralPoolingParameter_PoolMethod_FSUM:
      // Initialize
      caffe_set(top_count, Dtype(0), top_data);

      // The main loop
      for (int n = 0; n < num_im; ++n) {
        for (int c = 0; c < num_class; ++c) {
          const int pool_index = n * num_class + c;

          for (int r = 0; r < num_roi; ++r) {
            const int index = r * num_class + c;
            if (fliter[index] == Dtype(0)) continue;
            top_data[pool_index] += bottom_data[index];
          }
        }
      }
      break;
    case GeneralPoolingParameter_PoolMethod_TSUM:
      // Initialize
      caffe_set(top_count, Dtype(0), top_data);

      // The main loop
      for (int n = 0; n < num_im; ++n) {
        for (int c = 0; c < num_class; ++c) {
          const int pool_index = n * num_class + c;
          int pool_size = 0;
          Dtype all_sum = 0;

          for (int r = 0; r < num_roi; ++r) {
            const int index = r * num_class + c;
            const Dtype in = bottom_data[index];
            all_sum += in;
            if (in < threshold_) continue;
            top_data[pool_index] += in;
            pool_size++;
          }
          if (pool_size == 0) {
            top_data[pool_index] = all_sum;
          }
          mask[pool_index] = pool_size;
        }
      }
      break;
    case GeneralPoolingParameter_PoolMethod_FAVEMAX:
      // Initialize
      caffe_set(top_count, Dtype(0), top_data);

      // The main loop
      for (int n = 0; n < num_im; ++n) {
        for (int c = 0; c < num_class; ++c) {
          const int pool_index = n * num_class + c;
          int pool_size = 0;

          Dtype max_value = -FLT_MAX;
          int max_value_index = -1;
          Dtype max_value_fliter = -FLT_MAX;
          int max_value_index_fliter = -1;

          for (int r = 0; r < num_roi; ++r) {
            const int index = r * num_class + c;
            const Dtype in = bottom_data[index];
            if (in > max_value) {
              max_value = in;
              max_value_index = index;
            }
            if (fliter[index] == Dtype(0)) continue;
            if (in > max_value_fliter) {
              max_value_fliter = in;
              max_value_index_fliter = index;
            }
            if (in < threshold_) continue;
            top_data[pool_index] += in;
            pool_size++;
          }
          if (pool_size == 0) {
            if (max_value_fliter != -FLT_MAX) {
              top_data[pool_index] = max_value_fliter;
              mask[pool_index] = max_value_index_fliter;
            } else {
              top_data[pool_index] = max_value;
              mask[pool_index] = max_value_index;
            }
            mask[pool_index] *= -1;
          } else {
            top_data[pool_index] /= pool_size;
            mask[pool_index] = pool_size;
          }
        }
      }
      break;
    case GeneralPoolingParameter_PoolMethod_TAVEMAX: {
      // Initialize
      caffe_set(top_count, Dtype(0), top_data);

      for (int i = 0; i < outer_num_; ++i) {
        for (int k = 0; k < outer_num_; ++k) {
          int pool_index = i * inner_num_ + k;
          int pool_size = 0;
          Dtype max_value = -FLT_MAX;
          int max_value_index = -1;
          for (int j = 0; j < channels_; ++j) {
            const int index = i * dim_ + j * inner_num_ + k;
            const Dtype in = bottom_data[index];
            if (in > max_value) {
              max_value = in;
              max_value_index = index;
            }
            if (in < threshold_) continue;
            top_data[pool_index] += in;
            pool_size++;
          }
          if (pool_size == 0) {
            top_data[pool_index] = max_value;
            mask[pool_index] = max_value_index;
            mask[pool_index] *= -1;
          } else {
            top_data[pool_index] /= pool_size;
            mask[pool_index] = pool_size;
          }
        }
      }
    } break;
    case GeneralPoolingParameter_PoolMethod_MUL:
      caffe_mul(bottom[0]->count(), bottom_data, fliter, top_data);
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void GeneralPoolingLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // score	#roi	#class	1	1
  // fliter	#roi	#class	1	1
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* fliter = NULL;
  if (bottom.size() == 2) fliter = bottom[1]->cpu_data();

  const int num_roi = bottom[0]->num();
  const int num_class = bottom[0]->channels();
  const int num_im = 1;

  const int* mask = mask_idx_.cpu_data();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  switch (this->layer_param_.general_pooling_param().pool()) {
    case GeneralPoolingParameter_PoolMethod_FMAX:
      // The main loop
      for (int n = 0; n < num_im; ++n) {
        for (int c = 0; c < num_class; ++c) {
          const int index = n * num_class + c;
          const int bottom_index = mask[index];
          bottom_diff[bottom_index] = top_diff[index];
        }
      }
      break;
    case GeneralPoolingParameter_PoolMethod_FAVE:
      // The main loop
      for (int n = 0; n < num_im; ++n) {
        for (int c = 0; c < num_class; ++c) {
          const int pool_index = n * num_class + c;
          const int pool_size = mask[pool_index];
          const bool is_all = (pool_size == num_roi) ? true : false;
          if (is_all) {
            for (int r = 0; r < num_roi; ++r) {
              const int index = r * num_class + c;
              bottom_diff[index] += top_diff[pool_index] / pool_size;
            }
          } else {
            for (int r = 0; r < num_roi; ++r) {
              const int index = r * num_class + c;
              if (fliter[index] == Dtype(0)) continue;
              bottom_diff[index] = top_diff[pool_index] / pool_size;
            }
          }
        }
      }
      break;
    case GeneralPoolingParameter_PoolMethod_SUM: {
      for (int i = 0; i < outer_num_; ++i) {
        for (int k = 0; k < inner_num_; ++k) {
          int pool_index = i * inner_num_ + k;
          for (int j = 0; j < channels_; ++j) {
            int index = i * dim_ + j * inner_num_ + k;
            bottom_diff[index] = top_diff[pool_index];
          }
        }
      }
    } break;
    case GeneralPoolingParameter_PoolMethod_MAX: {
      for (int i = 0; i < outer_num_; ++i) {
        for (int k = 0; k < inner_num_; ++k) {
          int pool_index = i * inner_num_ + k;
          int bottom_index = mask[pool_index];
          bottom_diff[bottom_index] = top_diff[pool_index];
        }
      }
    } break;
    case GeneralPoolingParameter_PoolMethod_FSUM:
      // The main loop
      for (int n = 0; n < num_im; ++n) {
        for (int c = 0; c < num_class; ++c) {
          const int pool_index = n * num_class + c;
          for (int r = 0; r < num_roi; ++r) {
            const int index = r * num_class + c;
            if (fliter[index] == Dtype(0)) continue;
            bottom_diff[index] = top_diff[pool_index];
          }
        }
      }
      break;
    case GeneralPoolingParameter_PoolMethod_TSUM:
      // The main loop
      for (int n = 0; n < num_im; ++n) {
        for (int c = 0; c < num_class; ++c) {
          const int pool_index = n * num_class + c;
          const int pool_size = mask[pool_index];
          if (pool_size == 0) {
            for (int r = 0; r < num_roi; ++r) {
              const int index = r * num_class + c;
              bottom_diff[index] = top_diff[pool_index];
            }
          } else {
            for (int r = 0; r < num_roi; ++r) {
              const int index = r * num_class + c;
              if (bottom_data[index] < threshold_) continue;
              bottom_diff[index] = top_diff[pool_index];
            }
          }
        }
      }
      break;
    case GeneralPoolingParameter_PoolMethod_FAVEMAX:
      // The main loop
      for (int n = 0; n < num_im; ++n) {
        for (int c = 0; c < num_class; ++c) {
          const int pool_index = n * num_class + c;
          const int pool_size = mask[pool_index];
          if (pool_size > 0) {
            for (int r = 0; r < num_roi; ++r) {
              const int index = r * num_class + c;
              if (fliter[index] == Dtype(0)) continue;
              if (bottom_data[index] < threshold_) continue;
              bottom_diff[index] = top_diff[pool_index] / pool_size;
            }
          } else {
            const int bottom_index = -1 * pool_size;
            bottom_diff[bottom_index] = top_diff[pool_index];
          }
        }
      }
      break;
    case GeneralPoolingParameter_PoolMethod_TAVEMAX: {
      for (int i = 0; i < outer_num_; ++i) {
        for (int k = 0; k < inner_num_; ++k) {
          const int pool_index = i * inner_num_ + k;
          const int pool_size = mask[pool_index];

          if (pool_index > 0) {
            for (int j = 0; j < channels_; ++j) {
              const int index = i * dim_ + j * inner_num_ + k;
              if (bottom_data[index] < threshold_) continue;
              bottom_diff[index] = top_diff[pool_index] / pool_size;
            }
          } else {
            const int bottom_index = -1 * pool_size;
            bottom_diff[bottom_index] = top_diff[pool_index];
          }
        }
      }
    } break;
    case GeneralPoolingParameter_PoolMethod_MUL:
      NOT_IMPLEMENTED;
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
}

//#ifdef CPU_ONLY
// STUB_GPU(GeneralPoolingLayer);
//#endif

INSTANTIATE_CLASS(GeneralPoolingLayer);
REGISTER_LAYER_CLASS(GeneralPooling);

}  // namespace caffe
