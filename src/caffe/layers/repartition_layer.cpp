#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/repartition_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RepartitionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  CPGParameter this_layer_param = this->layer_param_.cpg_param();
  is_cpg_ = this_layer_param.is_cpg();
  max_num_im_cpg_ = this_layer_param.max_num_im_cpg();
  debug_info_ = this_layer_param.debug_info();
  is_order_ = this_layer_param.is_order();

  ignore_label_ = this_layer_param.ignore_label();

  predict_threshold_ = this_layer_param.predict_threshold();
  predict_order_ = this_layer_param.predict_order();
  crf_threshold_ = this_layer_param.crf_threshold();
  fg_threshold_ = this_layer_param.fg_threshold();
  bg_threshold_ = this_layer_param.bg_threshold();
  mass_threshold_ = this_layer_param.mass_threshold();
  density_threshold_ = this_layer_param.density_threshold();

  switch (this->layer_param_.cpg_param().mode()) {
    case CPGParameter_Mode_PRED:
      LOG(INFO) << "mode: PRED";
      break;
    case CPGParameter_Mode_CPG_POOLING:
      LOG(INFO) << "mode: CPG_POOLING";
      CHECK_EQ(is_order_, false)
          << "In CPG_POOLING mode, is_order_ should be false.";
      break;
    case CPGParameter_Mode_CRF:
      LOG(INFO) << "mode: CRF";
      break;
    default:
      LOG(FATAL) << "Unknown mode.";
  }

  bottom_index_["cpg"] = 0;
  bottom_index_["label"] = 1;
  bottom_index_["predict"] = 2;
  bottom_index_["rois"] = 3;
  bottom_index_["rois_score"] = 4;
  bottom_index_["filter"] = 5;
  bottom_index_["io"] = 6;

  num_class_ = bottom[bottom_index_["label"]]->channels();

  display_ = 1280;
  pass_im_ = 0;
  max_bb_per_im_ = 4;
  max_bb_per_cls_ = 4;

  if (is_order_) {
    order_K_ = 3;
    order_step_ = 10022 * 3;
    order_threshold_ = 1.0 * (order_K_ - 1) / order_K_;
    CHECK_EQ(top.size(), 3) << "In size order mode, #top should be 3!";
  }

  LOG(INFO) << "----------------------------------------------";
  LOG(INFO) << "is_cpg_: " << is_cpg_;
  LOG(INFO) << "is_order_: " << is_order_;
  LOG(INFO) << "debug_info_: " << debug_info_;
  LOG(INFO) << "predict_threshold_:" << predict_threshold_;
  LOG(INFO) << "predict_order_:" << predict_order_;
  LOG(INFO) << "fg_threshold_:" << fg_threshold_;
  LOG(INFO) << "bg_threshold_:" << bg_threshold_;
  LOG(INFO) << "mass_threshold_:" << mass_threshold_;
  LOG(INFO) << "density_threshold_:" << density_threshold_;
  LOG(INFO) << "max_bb_per_cls_:" << max_bb_per_cls_;
  LOG(INFO) << "----------------------------------------------";

  if (false) {
    crf_bottom_vec_.clear();
    crf_bottom_vec_.push_back(crf_cpg_.get());
    crf_bottom_vec_.push_back(crf_data_dim_.get());
    crf_bottom_vec_.push_back(crf_data_.get());
    crf_top_vec_.clear();
    crf_top_vec_.push_back(crf_output_.get());

    crf_data_->Reshape(1, 3, 1000, 1000);
    crf_cpg_->Reshape(1, 1, 1000, 1000);
    crf_data_dim_->Reshape(1, 2, 1, 1);
    crf_layer_->SetUp(crf_bottom_vec_, crf_top_vec_);
  }

  voc_label_.push_back("aeroplane");   // 0
  voc_label_.push_back("bicycle");     // 1
  voc_label_.push_back("bird");        // 2
  voc_label_.push_back("boat");        // 3
  voc_label_.push_back("bottle");      // 4
  voc_label_.push_back("bus");         // 5
  voc_label_.push_back("car");         // 6
  voc_label_.push_back("cat");         // 7
  voc_label_.push_back("chair");       // 8
  voc_label_.push_back("cow");         // 9
  voc_label_.push_back("diningtb");    // 10
  voc_label_.push_back("dog");         // 11
  voc_label_.push_back("horse");       // 12
  voc_label_.push_back("motorbike");   // 13
  voc_label_.push_back("person");      // 14
  voc_label_.push_back("potted");      // 15
  voc_label_.push_back("sheep");       // 16
  voc_label_.push_back("sofa");        // 17
  voc_label_.push_back("train");       // 18
  voc_label_.push_back("tvmonitor");   // 19
  voc_label_.push_back("background");  // 20
}

template <typename Dtype>
void RepartitionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.cpg_param().mode()) {
    case CPGParameter_Mode_PRED:
      break;
    case CPGParameter_Mode_CPG_POOLING:
      break;
    case CPGParameter_Mode_CRF:
      break;
    default:
      LOG(FATAL) << "Unknown mode.";
  }

  switch (this->layer_param_.cpg_param().mode()) {
    case CPGParameter_Mode_PRED:
    case CPGParameter_Mode_CPG_POOLING:
      channels_cpg_ = 1;
      height_im_ = bottom[bottom_index_["cpg"]]->height();
      width_im_ = bottom[bottom_index_["cpg"]]->width();
      size_cpg_ = height_im_ * width_im_;
      raw_data_->Reshape(1, 1, height_im_, width_im_);

      CHECK_EQ(bottom[bottom_index_["label"]]->num(),
               bottom[bottom_index_["cpg"]]->num())
          << "bottom nums are not the same.";
      LOG_IF(INFO, is_cpg_ && debug_info_)
          << "cpg info: channels: " << channels_cpg_
          << " height: " << height_im_ << " width: " << width_im_;
      break;
    case CPGParameter_Mode_CRF:
      LOG(FATAL) << "Not yet.";
      break;
    default:
      LOG(FATAL) << "Unknown mode.";
  }

  num_roi_ = bottom[bottom_index_["rois"]]->num();
  num_im_ = bottom[bottom_index_["label"]]->num();

  // shape top
  switch (this->layer_param_.cpg_param().mode()) {
    case CPGParameter_Mode_PRED:
    case CPGParameter_Mode_CPG_POOLING: {
      vector<int> top_dims;
      top_dims.push_back(num_roi_);
      top_dims.push_back(num_class_);
      top[0]->Reshape(top_dims);
      caffe_gpu_set(top[0]->count(), Dtype(1), top[0]->mutable_gpu_data());

      // In test model, the output number is one.
      if (top.size() == 3) {
        top[1]->CopyFrom(*bottom[bottom_index_["label"]], false, true);
        top[2]->ReshapeLike(*bottom[bottom_index_["label"]]);
        caffe_set(top[2]->count(), Dtype(0), top[2]->mutable_cpu_data());
      }
    } break;
    case CPGParameter_Mode_CRF:
      break;
    default:
      LOG(FATAL) << "Unknown mode.";
  }

  // shape filter
  vector<int> filter_dims;
  filter_dims.push_back(num_roi_);
  filter_dims.push_back(num_class_);
  filter_.Reshape(filter_dims);

  // shape bboxes_
  vector<int> bboxes_dims;
  bboxes_dims.push_back(max_bb_per_cls_);
  bboxes_dims.push_back(4);
  bboxes_->Reshape(bboxes_dims);

  // Do some check
  CHECK_EQ(num_im_, 1) << "Current only support one image per forward-backward";
  CHECK_EQ(bottom[bottom_index_["predict"]]->shape(0), num_im_)
      << "#im should be the same";
  CHECK_EQ(bottom[bottom_index_["predict"]]->shape(1), num_class_)
      << "#class should be the same";
  CHECK_EQ(bottom[bottom_index_["predict"]]->count(), num_im_ * num_class_)
      << "size should be the same";
  CHECK_EQ(bottom[bottom_index_["label"]]->count(), num_im_ * num_class_)
      << "size should be the same";

  if (bottom_index_.find("io") != bottom_index_.end() &&
      bottom.size() > bottom_index_["io"]) {
    CHECK_EQ(bottom[bottom_index_["filter"]]->shape(0), num_roi_)
        << "#roi should be the same";
    CHECK_EQ(bottom[bottom_index_["filter"]]->shape(1), num_class_)
        << "#class should be the same";
    CHECK_EQ(bottom[bottom_index_["io"]]->count(), 1) << "only need one IO ID";
  }
}

template <typename Dtype>
void RepartitionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void RepartitionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(RepartitionLayer);
#endif

INSTANTIATE_CLASS(RepartitionLayer);
REGISTER_LAYER_CLASS(Repartition);

}  // namespace caffe
