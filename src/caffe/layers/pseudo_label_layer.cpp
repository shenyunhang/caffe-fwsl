#include <cfloat>
#include <vector>

#include "caffe/layers/pseudo_label_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PseudoLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 2) << "shape missmatch.";

  CHECK_EQ(bottom[1]->num(), bottom[0]->num()) << "number is not consist.";
  CHECK_EQ(bottom[1]->channels(), 5) << "shape missmatch.";
  CHECK_EQ(bottom[1]->num_axes(), 2) << "shape missmatch.";

  CHECK_EQ(bottom[2]->channels(), bottom[0]->channels())
      << "channels not consist.";
  CHECK_EQ(bottom[2]->num_axes(), 2) << "shape missmatch.";
}

template <typename Dtype>
void PseudoLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  // roi_score
  // roi
  // label
  num_roi_ = bottom[0]->num();
  CHECK_EQ(bottom[0]->num_axes(), 2) << "shape missmatch.";

  CHECK_EQ(bottom[1]->num(), bottom[0]->num()) << "number is not consist.";
  CHECK_EQ(bottom[1]->channels(), 5) << "shape missmatch.";
  CHECK_EQ(bottom[1]->num_axes(), 2) << "shape missmatch.";

  num_img_ = bottom[2]->num();
  num_classes_ = bottom[2]->count(1);
  CHECK_EQ(bottom[2]->channels(), bottom[0]->channels())
      << "channels not consist.";
  CHECK_EQ(bottom[2]->num_axes(), 2) << "shape missmatch.";

  vector<int> top_shape(4);
  top_shape[0] = 1;
  top_shape[1] = 1;
  top_shape[2] = 0;
  top_shape[3] = 8;
  top[0]->Reshape(top_shape);

  // 假设每张图像每个类别有一个GT
  reserve_size_ = num_img_ * num_classes_;
}

template <typename Dtype>
void PseudoLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  const Dtype* score_data = bottom[0]->cpu_data();
  const Dtype* roi_data = bottom[1]->cpu_data();
  const Dtype* label_data = bottom[2]->cpu_data();

  vector<int> index_pgt_roi;
  vector<int> group_label_pgt_roi;
  vector<int> instance_id_pgt_roi;

  index_pgt_roi.reserve(reserve_size_);
  group_label_pgt_roi.reserve(reserve_size_);
  instance_id_pgt_roi.reserve(reserve_size_);

  for (int i = 0; i < num_img_; i++) {
    for (int j = 0; j < num_classes_; j++) {
      int label = label_data[i * num_classes_ + j];
      if (label != 1) {
        continue;
      }

      Dtype max_score = -FLT_MAX;
      vector<int> idx_score;
      idx_score.clear();
      for (int r = 0; r < num_roi_; r++) {
        if (roi_data[r * 5 + 0] != i) {
          continue;
        }
        Dtype score = score_data[r * num_classes_ + j];
        if (score > max_score) {
          max_score = score;
          idx_score.clear();
          idx_score.push_back(r);
        } else if (score == max_score) {
          idx_score.push_back(r);
        }
      }
      for (size_t t = 0; t < idx_score.size(); t++) {
        group_label_pgt_roi.push_back(j);
        instance_id_pgt_roi.push_back(t);
      }
      index_pgt_roi.insert(index_pgt_roi.begin(), idx_score.begin(),
                           idx_score.end());
    }
  }

  // Reshape the label and store the annotation.
  vector<int> top_shape(4);
  top_shape[0] = 1;
  top_shape[1] = 1;
  top_shape[2] = index_pgt_roi.size();
  top_shape[3] = 1;
  top[0]->Reshape(top_shape);
  Dtype* top_label = top[0]->mutable_cpu_data();

  int idx = 0;
  for (size_t t = 0; t < index_pgt_roi.size(); t++) {
    int r = index_pgt_roi[t];
    top_label[idx++] = roi_data[r * 5 + 0];
    top_label[idx++] = group_label_pgt_roi[t];
    top_label[idx++] = instance_id_pgt_roi[t];
    top_label[idx++] = roi_data[r * 5 + 1];
    top_label[idx++] = roi_data[r * 5 + 2];
    top_label[idx++] = roi_data[r * 5 + 3];
    top_label[idx++] = roi_data[r * 5 + 4];
    top_label[idx++] = Dtype(0);
  }
}

template <typename Dtype>
void PseudoLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
  for (size_t i = 0; i < bottom.size(); i++) {
    if (propagate_down[i]) {
      caffe_set(bottom[i]->count(), static_cast<Dtype>(0),
                bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(PseudoLabelLayer);
#endif

INSTANTIATE_CLASS(PseudoLabelLayer);
REGISTER_LAYER_CLASS(PseudoLabel);

}  // namespace caffe
