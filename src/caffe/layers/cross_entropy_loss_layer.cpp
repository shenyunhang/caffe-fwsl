#include <vector>

#include "caffe/layers/cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  CrossEntropyLossParameter this_layer_param =
      this->layer_param_.cross_entropy_loss_param();
  display_ = this_layer_param.display();
  ignore_label_ = this_layer_param.ignore_label();
  ignore_value_ = this_layer_param.ignore_value();

  accum_loss_ = 0;
  accum_iter_ = 0;
  accum_ignore_num_ = 0;
  num_class_ = bottom[0]->channels();
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  count_ = bottom[0]->count();
  num_im_ = bottom[0]->num();
  CHECK_EQ(num_class_, bottom[0]->channels()) << "channels had changed.";
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "CROSS_ENTROPY_LOSS layer inputs must have the same count.";
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();

  Dtype loss = 0;
  for (int i = 0; i < count_; ++i) {
    //LOG(INFO) << "i: " << i << " input_data[i]: " << input_data[i]
              //<< " target: " << target[i];
    if (target[i] == ignore_value_) {
      accum_ignore_num_++;
      continue;
    }
    if (i % num_class_ == ignore_label_) {
      accum_ignore_num_++;
      continue;
    }
    Dtype prob = std::max(input_data[i], Dtype(kLOG_THRESHOLD));
    Dtype one_prob = std::max(1 - input_data[i], Dtype(kLOG_THRESHOLD));
    loss -= (target[i] * log(prob) + (1 - target[i]) * log(one_prob));
  }
  top[0]->mutable_cpu_data()[0] = loss / num_im_;

  accum_loss_ += loss / num_im_;
  accum_iter_++;
  if (accum_iter_ % display_ == 0) {
    LOG(INFO) << this->layer_param().name() << " #iter_: " << accum_iter_
              << " #ignore: " << accum_ignore_num_ << " #loss_: " << accum_loss_
              << " AVE loss: " << accum_loss_ / accum_iter_;
    accum_loss_ = 0;
    accum_iter_ = 0;
    accum_ignore_num_ = 0;
  }
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  if (!propagate_down[0]) {
    return;
  }
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype scale = top[0]->cpu_diff()[0] / num_im_;
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int i = 0; i < count_; ++i) {
    if (i % num_class_ == ignore_label_) {
      bottom_diff[i] = 0;
      continue;
    }
    if (target[i] == ignore_value_) {
      bottom_diff[i] = 0;
      continue;
    }
    Dtype prob = std::max(input_data[i], Dtype(kLOG_THRESHOLD));
    Dtype one_prob = std::max(1 - input_data[i], Dtype(kLOG_THRESHOLD));
    bottom_diff[i] = std::min(
        scale * (-1 * target[i] / prob - (-1) * (1 - target[i]) / one_prob),
        Dtype(kDIFF_THRESHOLD));
  }
}

INSTANTIATE_CLASS(CrossEntropyLossLayer);
REGISTER_LAYER_CLASS(CrossEntropyLoss);

}  // namespace caffe
