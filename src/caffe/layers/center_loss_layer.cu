#include <cfloat>
#include <vector>

#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  if (max_num_im_center_ >= 0 && max_num_im_center_ <= pass_im_) {
    is_center_ = false;
  }
  if (!is_center_) {
    return;
  }
  LOG_IF(INFO, debug_info_) << "    [Forward] ";
  const Dtype* label_data = bottom[1]->cpu_data();
  const Dtype* score_data = bottom[2]->cpu_data();

  // get the top_k_ score RoI index
  for (int c = 0; c < num_class_; ++c) {
    roi_sets_[c].clear();
    if (label_data[c] != 1) continue;
    LOG_IF(INFO, debug_info_) << "class: " << c;
    num_gt_class_++;

    for (int k = 0; k < top_k_; ++k) {
      Dtype max_value = -FLT_MAX;
      int max_value_index = -1;
      for (int r = 0; r < num_roi_; ++r) {
        if (max_value < score_data[r * num_class_ + c]) {
          if (roi_sets_[c].find(r) == roi_sets_[c].end()) {
            max_value = score_data[r * num_class_ + c];
            max_value_index = r;
          }
        }
      }
      if (max_value_index == -1) {
        LOG(FATAL) << "can not find enought roi.";
        break;
      }
      roi_sets_[c].insert(max_value_index);
      LOG_IF(INFO, debug_info_) << "max_value_index: " << max_value_index
                                << " max_value: " << max_value;
    }
  }

  const Dtype* feature_data = bottom[0]->gpu_data();
  const Dtype* center_data = this->blobs_[0]->gpu_data();
  Dtype* diff_data = diff_.mutable_gpu_data();
  Dtype* diff_diff = diff_.mutable_gpu_diff();

  // TODO(YH): BUG, RoI score is forget to use.
  Dtype dot = 0;
  for (int c = 0; c < num_class_; ++c) {
    center_selector_[c] = -1;
    if (roi_sets_[c].size() == 0) continue;
    Dtype c_dot = FLT_MAX;
    for (int m = 0; m < num_center_; ++m) {
      set<int>::iterator it;
      int k;
      for (k = 0, it = roi_sets_[c].begin(); it != roi_sets_[c].end();
           ++k, ++it) {
        int r = *it;
        caffe_gpu_sub(dim_, feature_data + r * dim_,
                      center_data + (c * num_center_ + m) * dim_,
                      diff_diff + (c * top_k_ + k) * dim_);
      }
      Dtype cm_dot;
      caffe_gpu_dot(top_k_ * dim_, diff_diff + (c * top_k_ + 0) * dim_,
                    diff_diff + (c * top_k_ + 0) * dim_, &cm_dot);
      if (cm_dot < c_dot) {
        caffe_copy(top_k_ * dim_, diff_diff + (c * top_k_ + 0) * dim_,
                   diff_data + (c * top_k_ + 0) * dim_);
        c_dot = cm_dot;
        center_selector_[c] = m;
      }
    }
    num_update_class_[c][center_selector_[c]]++;
    accum_update_class_[c][center_selector_[c]]++;
    dot += c_dot;
  }

  Dtype loss = dot / num_gt_class_ / top_k_ / dim_ / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  accum_loss_ += loss;
  total_iter_++;
  if (total_iter_ % display_ == 0) {
    LOG(INFO) << this->layer_param().name() << " #iter_: " << total_iter_
              << " #loss_: " << accum_loss_
              << " AVE loss: " << accum_loss_ / display_;
    accum_loss_ = 0;

    for (int c = 0; c < num_class_; ++c) {
      std::cout << "(";
      for (int m = 0; m < num_center_; ++m) {
        std::cout << accum_update_class_[c][m] << " ";
        accum_update_class_[c][m] = 0;
      }
      std::cout << "\b)";
    }
    std::cout << std::endl;
  }
  LOG_IF(INFO, debug_info_) << this->blobs_[0]->asum_data() << " "
                            << diff_.asum_data();
  LOG_IF(INFO, debug_info_) << this->blobs_[0]->asum_diff() << " "
                            << diff_.asum_diff();

  pass_im_ += bottom[1]->num();
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom) {
  if (!is_center_) {
    for (size_t i = 0; i < bottom.size(); i++) {
      if (propagate_down[i]) {
        caffe_gpu_set(bottom[i]->count(), static_cast<Dtype>(0),
                      bottom[i]->mutable_gpu_diff());
      }
    }
    return;
  }

  LOG_IF(INFO, debug_info_) << "    [Backward] ";
  if (!propagate_down[0]) return;

  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());

  const Dtype alpha = top[0]->cpu_diff()[0] / num_gt_class_ / top_k_ / dim_;
  const Dtype* diff_data = diff_.gpu_data();
  LOG_IF(INFO, debug_info_) << "alpha: " << alpha
                            << " weight: " << top[0]->cpu_diff()[0];

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* center_diff = this->blobs_[0]->mutable_gpu_diff();

  for (int c = 0; c < num_class_; ++c) {
    LOG_IF(INFO, debug_info_) << "class: " << c;
    set<int>::iterator it;
    int k;
    for (k = 0, it = roi_sets_[c].begin(); it != roi_sets_[c].end();
         ++k, ++it) {
      int r = *it;
      LOG_IF(INFO, debug_info_) << "r: " << r;
      // feature diff
      caffe_gpu_axpby(dim_, alpha, diff_data + (c * top_k_ + k) * dim_,
                      Dtype(1), bottom_diff + r * dim_);

      // TODO(YH): whether center update is correct
      // center diff
      caffe_gpu_axpby(
          dim_, Dtype(-1), diff_data + (c * top_k_ + k) * dim_, Dtype(1),
          center_diff + (c * num_center_ + center_selector_[c]) * dim_);
    }
  }

  // update center
  if (total_iter_ % update_ == 0) {
    for (int c = 0; c < num_class_; ++c) {
      for (int m = 0; m < num_center_; ++m) {
        caffe_gpu_axpy(
            dim_, lr_ * Dtype(-1) / (num_update_class_[c][m] * top_k_ + 1),
            this->blobs_[0]->gpu_diff() + (c * num_center_ + m) * dim_,
            this->blobs_[0]->mutable_gpu_data() + (c * num_center_ + m) * dim_);
        num_update_class_[c][m] = 0;
      }
    }
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0),
                  this->blobs_[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CenterLossLayer);

}  // namespace caffe
