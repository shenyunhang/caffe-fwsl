#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>  // std::cout, std::fixed
#include <iomanip>   // std::setprecision

#include "caffe/layers/statistics_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void StatisticsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  bottom_index_["label"] = 0;
  bottom_index_["predict"] = 1;
  bottom_index_["predict_pos"] = 2;
  bottom_index_["predict_neg"] = 3;
  bottom_index_["filter"] = 4;

  display_ = 1280;
  predict_threshold_ = 0.7;
  num_class_ = bottom[bottom_index_["label"]]->channels();

  pass_im_ = 0;
  for (int c = 0; c < num_class_; ++c) {
    ori_.label.push_back(0);
    ori_.predict.push_back(0.0);
    ori_.predict_pos.push_back(0.0);
    ori_.predict_neg.push_back(0.0);
    ori_.roi.push_back(0);
    ori_.roi_left.push_back(0);

    cpg_.label.push_back(0);
    cpg_.predict.push_back(0.0);
    cpg_.predict_pos.push_back(0.0);
    cpg_.predict_neg.push_back(0.0);
    cpg_.roi.push_back(0);
    cpg_.roi_left.push_back(0);
  }

  ori_.accum_label = 0;
  ori_.accum_predict = 0;
  ori_.accum_predict_pos = 0;
  ori_.accum_predict_neg = 0;
  ori_.accum_roi = 0;
  ori_.accum_roi_left = 0;

  cpg_.accum_label = 0;
  cpg_.accum_predict = 0;
  cpg_.accum_predict_pos = 0;
  cpg_.accum_predict_neg = 0;
  cpg_.accum_roi = 0;
  cpg_.accum_roi_left = 0;

  voc_label_.push_back("aeroplane");    // 0
  voc_label_.push_back("bicycle");      // 1
  voc_label_.push_back("bird");         // 2
  voc_label_.push_back("boat");         // 3
  voc_label_.push_back("bottle");       // 4
  voc_label_.push_back("bus");          // 5
  voc_label_.push_back("car");          // 6
  voc_label_.push_back("cat");          // 7
  voc_label_.push_back("chair");        // 8
  voc_label_.push_back("cow");          // 9
  voc_label_.push_back("diningtable");  // 10
  voc_label_.push_back("dog");          // 11
  voc_label_.push_back("horse");        // 12
  voc_label_.push_back("motorbike");    // 13
  voc_label_.push_back("person");       // 14
  voc_label_.push_back("pottedplant");  // 15
  voc_label_.push_back("sheep");        // 16
  voc_label_.push_back("sofa");         // 17
  voc_label_.push_back("train");        // 18
  voc_label_.push_back("tvmonitor");    // 19
  voc_label_.push_back("background");   // 20
}

template <typename Dtype>
void StatisticsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void StatisticsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  num_im_ = bottom[bottom_index_["label"]]->num();
  num_roi_ = bottom[bottom_index_["filter"]]->num();

  CHECK_EQ(num_im_, 1)
      << "Current only support one image per forward and backward.";
  CHECK_EQ(num_class_, bottom[bottom_index_["label"]]->channels())
      << "number of class should not changed.";

  pass_im_ += num_im_;

  const Dtype* label = bottom[bottom_index_["label"]]->cpu_data();
  const Dtype* predict = bottom[bottom_index_["predict"]]->cpu_data();
  const Dtype* predict_pos = bottom[bottom_index_["predict_pos"]]->cpu_data();
  const Dtype* predict_neg = bottom[bottom_index_["predict_neg"]]->cpu_data();
  const Dtype* filter = bottom[bottom_index_["filter"]]->cpu_data();

  for (int c = 0; c < num_class_; ++c) {
    int index = c;
    if (this->phase_ == TEST) LOG(FATAL) << "Undone.";
    if (label[index] <= 0.5) continue;

    ori_.label[c]++;
    ori_.predict[c] += predict[index];
    ori_.roi[c] += num_roi_;
    ori_.accum_label++;
    ori_.accum_predict += predict[index];
    ori_.accum_roi += num_roi_;

    if (predict[index] < predict_threshold_) continue;

    cpg_.label[c]++;
    cpg_.predict[c] += predict[index];
    cpg_.predict_pos[c] += predict_pos[index];
    cpg_.predict_neg[c] += predict_neg[index];
    cpg_.roi[c] += num_roi_;
    cpg_.accum_label++;
    cpg_.accum_predict += predict[index];
    cpg_.accum_predict_pos += predict_pos[index];
    cpg_.accum_predict_neg += predict_neg[index];
    cpg_.accum_roi += num_roi_;
    for (int r = 0; r < num_roi_; ++r) {
      if (filter[r * num_class_ + c] > 0) {
        cpg_.roi_left[c]++;
        cpg_.accum_roi_left++;
      }
    }
  }

  if (pass_im_ % display_ != 0) return;

  std::cout << "#class\tprediction\t#roi\t#class\tpredition\tpos\t\tneg\t\t#"
               "roi\t#left\t%\t\tclass" << std::endl;

  for (int c = 0; c < num_class_; ++c) {
    std::cout << std::fixed << std::setprecision(7);
    if (ori_.label[c] == 0) {
      CHECK_EQ(ori_.predict[c], 0) << "There should no regions.";
      CHECK_EQ(ori_.roi[c], 0) << "There should no regions.";
      std::cout << "0\t0.000000\t000";
    } else {
      std::cout << ori_.label[c] << "\t" << ori_.predict[c] / ori_.label[c]
                << "\t" << ori_.roi[c] / ori_.label[c];
    }

    ori_.label[c] = 0;
    ori_.predict[c] = 0;
    ori_.roi[c] = 0;

    if (cpg_.label[c] == 0) {
      CHECK_EQ(cpg_.predict[c], 0) << "There should no regions.";
      CHECK_EQ(cpg_.predict_pos[c], 0) << "There should no regions.";
      CHECK_EQ(cpg_.predict_neg[c], 0) << "There should no regions.";
      CHECK_EQ(cpg_.roi[c], 0) << "There should no regions.";
      CHECK_EQ(cpg_.roi_left[c], 0) << "There should no regions.";
      std::cout << "\t0\t0.000000\t0.000000\t0.000000\t000\t000\t0.000000\t"
                << voc_label_[c] << std::endl;
    } else {
      std::cout << "\t" << cpg_.label[c] << "\t"
                << cpg_.predict[c] / cpg_.label[c] << "\t"
                << cpg_.predict_pos[c] / cpg_.label[c] << "\t"
                << cpg_.predict_neg[c] / cpg_.label[c] << "\t"
                << cpg_.roi[c] / cpg_.label[c] << "\t"
                << cpg_.roi_left[c] / cpg_.label[c] << "\t"
                << 1.0 * cpg_.roi_left[c] / cpg_.roi[c] << "\t" << voc_label_[c]
                << std::endl;
    }

    cpg_.label[c] = 0;
    cpg_.predict[c] = 0;
    cpg_.predict_pos[c] = 0;
    cpg_.predict_neg[c] = 0;
    cpg_.roi[c] = 0;
    cpg_.roi_left[c] = 0;
  }

  if (ori_.accum_label > 0 && cpg_.accum_label > 0) {
    std::cout << ori_.accum_label << "\t"
              << ori_.accum_predict / ori_.accum_label << "\t"
              << ori_.accum_roi / ori_.accum_label << "\t" << cpg_.accum_label
              << "\t" << cpg_.accum_predict / cpg_.accum_label << "\t"
              << cpg_.accum_predict_pos / cpg_.accum_label << "\t"
              << cpg_.accum_predict_neg / cpg_.accum_label << "\t"
              << cpg_.accum_roi / cpg_.accum_label << "\t"
              << cpg_.accum_roi_left / cpg_.accum_label << "\t"
              << 1.0 * cpg_.accum_roi_left / cpg_.accum_roi << "\t" << display_
              << std::endl;
  }

  ori_.accum_label = 0;
  ori_.accum_predict = 0;
  ori_.accum_predict_pos = 0;
  ori_.accum_predict_neg = 0;
  ori_.accum_roi = 0;
  ori_.accum_roi_left = 0;

  cpg_.accum_label = 0;
  cpg_.accum_predict = 0;
  cpg_.accum_predict_pos = 0;
  cpg_.accum_predict_neg = 0;
  cpg_.accum_roi = 0;
  cpg_.accum_roi_left = 0;
}

template <typename Dtype>
void StatisticsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom) {
  for (size_t i = 0; i < bottom.size(); i++) {
    if (propagate_down[i]) {
      caffe_gpu_set(bottom[i]->count(), static_cast<Dtype>(0),
                    bottom[i]->mutable_gpu_diff());
    }
  }
}

//#ifdef CPU_ONLY
// STUB_GPU(STATISTICSLayer);
//#endif

INSTANTIATE_CLASS(StatisticsLayer);
REGISTER_LAYER_CLASS(Statistics);

}  // namespace caffe
