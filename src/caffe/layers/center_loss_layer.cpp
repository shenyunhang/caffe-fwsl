#include <vector>
#include <cfloat>

#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filler.hpp"

namespace caffe {

template <typename Dtype>
void CenterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  num_center_ = this->layer_param_.center_loss_param().num_center();
  top_k_ = this->layer_param_.center_loss_param().top_k();
  lr_ = this->layer_param_.center_loss_param().lr();
  debug_info_ = this->layer_param_.center_loss_param().debug_info();
  display_ = this->layer_param_.center_loss_param().display();
  update_ = this->layer_param_.center_loss_param().update();

  dim_ = bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
  num_class_ = bottom[1]->channels();

  for (int c = 0; c < num_class_; ++c) {
    set<int> r_s;
    roi_sets_.push_back(r_s);
    center_selector_.push_back(-1);

    vector<int> n_u_c;
    num_update_class_.push_back(n_u_c);
    vector<int> a_u_c;
    accum_update_class_.push_back(a_u_c);
    for (int m = 0; m < num_center_; ++m) {
      num_update_class_[c].push_back(0);
      accum_update_class_[c].push_back(0);
    }
  }

  accum_loss_ = 0;
  total_iter_ = 0;

  vector<int> center_dims;
  center_dims.push_back(num_class_);
  center_dims.push_back(num_center_);
  center_dims.push_back(dim_);
  this->blobs_.resize(1);
  this->blobs_[0].reset(new Blob<Dtype>(center_dims));

  // fill the center
  shared_ptr<Filler<Dtype> > weight_filler(
      GetFiller<Dtype>(this->layer_param_.center_loss_param().center_filler()));
  weight_filler->Fill(this->blobs_[0].get());
  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0),
                this->blobs_[0]->mutable_gpu_diff());

  vector<int> diff_dims;
  diff_dims.push_back(num_class_);
  diff_dims.push_back(top_k_);
  diff_dims.push_back(dim_);
  diff_.Reshape(diff_dims);
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  // LossLayer<Dtype>::Reshape(bottom, top);
  // 用下面两替代LossLayer<Dtype>::Reshape(bottom, top) 以此跳过里面的CHECK
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);

  num_gt_class_ = 0;
  num_roi_ = bottom[0]->num();
  CHECK_EQ(dim_, bottom[0]->channels()) << "Shape is not right.";

  CHECK_EQ(bottom[1]->num(), 1) << "Only support one image per forward.";
  CHECK_EQ(bottom[1]->channels(), num_class_) << "Shape is not right.";
  CHECK_EQ(bottom[1]->count(), num_class_) << "Shape is not right.";

  CHECK_EQ(bottom[2]->num(), num_roi_) << "Shape is not right.";
  CHECK_EQ(bottom[2]->channels(), num_class_) << "Shape is not right";
  CHECK_EQ(bottom[2]->height(), 1) << "Shape is not right";
  CHECK_EQ(bottom[2]->width(), 1) << "Shape is not right";

  caffe_gpu_set(diff_.count(), Dtype(0), diff_.mutable_gpu_data());
  caffe_gpu_set(diff_.count(), Dtype(0), diff_.mutable_gpu_diff());
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(CenterLossLayer);
#endif

INSTANTIATE_CLASS(CenterLossLayer);
REGISTER_LAYER_CLASS(CenterLoss);

}  // namespace caffe
