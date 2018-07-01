#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif  // USE_OPENCV

#include <vector>

#include "caffe/layers/base_roi_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void Show_batch_forward(const Batch<Dtype>* batch,
                        const TransformationParameter transform_param,
                        const int num_roi_visualize_) {
  const Dtype* data = batch->data_.cpu_data();
  const Dtype* roi = batch->roi_.cpu_data();
  const Dtype* roi_score = batch->roi_score_.cpu_data();
  const Dtype* roi_num = batch->roi_num_.cpu_data();

  const int num = batch->data_.num();
  const int channels = batch->data_.channels();
  const int height = batch->data_.height();
  const int width = batch->data_.width();
  const int num_roi = batch->roi_.num();

  const string save_path = "";
  const string save_path_jet = "";
  vector<float> mean_values;
  for (int c = 0; c < transform_param.mean_value_size(); ++c) {
    mean_values.push_back(transform_param.mean_value(c));
  }

  int index, index_mat;
  int xmin, ymin, xmax, ymax;
  for (int n = 0; n < num; ++n) {
    //-----------------------------------------------------------------------
    cv::Mat img_mat;
    cv::Mat jet_mat;
    if (channels == 3) {
      img_mat = cv::Mat(height, width, CV_8UC3);
    } else if (channels == 1) {
      img_mat = cv::Mat(height, width, CV_8UC1);
    } else {
      LOG(FATAL) << "channels should 1 or 3";
    }

    uchar* img_mat_data = img_mat.data;
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          index = (c * height + h) * width + w;
          index_mat = (h * width + w) * channels + c;
          img_mat_data[index_mat] = data[index] + mean_values[c];
        }
      }
    }

    int num_item_roi = 0;
    for (int r = 0; r < num_roi; r++) {
      if (roi[r * 5 + 0] == n) {
      } else {
        continue;
      }
      num_item_roi++;
      if (num_item_roi > num_roi_visualize_) {
        continue;
      }
      xmin = roi[r * 5 + 1] * width;
      ymin = roi[r * 5 + 2] * height;
      xmax = roi[r * 5 + 3] * width;
      ymax = roi[r * 5 + 4] * height;
      LOG(INFO) << roi[r * 5 + 1] << " " << roi[r * 5 + 2] << " "
                << roi[r * 5 + 3] << " " << roi[r * 5 + 4] << " "
                << roi_score[r];
      // LOG(INFO) << xmin << " " << ymin << " " << xmax << " " << ymax;

      cv::Rect rec = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
      cv::rectangle(img_mat, rec, cv::Scalar(0, 0, 255), 1);
    }
    // LOG(INFO) << "out_RoI num: " << num_item_roi;
    CHECK_EQ(num_item_roi, roi_num[n]) << "num of roi is not consistent.";

    std::stringstream ss;
    ss << "forwad_" << n;
    cv::imshow(ss.str(), img_mat);

    // cv::imwrite(save_path, cpg_mat);
    // LOG(INFO) << "save_path: " << save_path;

    // cv::applyColorMap(cpg_mat, cpg_mat_jet, cv::COLORMAP_JET);
    // cv::imwrite(save_path_jet, cpg_mat_jet);

    //-----------------------------------------------------------------------

    //-----------------------------------------------------------------------
    // show the distubution of cpg_blob data
    // int total[26];
    // for (int i = 0; i < 26; ++i) {
    // total[i] = 0;
    //}
    // for (int e = 0; e < channels * height * width; e++) {
    // int level = int(data[e] / 10);
    // total[level]++;
    //}
    // for (int i = 0; i < 26; ++i) {
    // std::cout << i << ":" << total[i] << " ";
    //}
    // std::cout << std::endl;
    ////-----------------------------------------------------------------------
    data += batch->data_.offset(1);
    roi += num_item_roi * 5;
  }
  cv::waitKey(0);
}

template <typename Dtype>
void BasePrefetchingRoIDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->phase_ == TEST) {
    if (count_img % 100 == 0) {
      LOG(INFO) << "Processed images: " << count_img;
    }
    count_img++;
  }

  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");

  // if (true) {
  if (false) {
    const TransformationParameter& transform_param =
        this->layer_param_.transform_param();
    Show_batch_forward(batch, transform_param, 100);
  }

  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data.
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
             top[0]->mutable_gpu_data());

  // Reshape to loaded roi.
  top[1]->ReshapeLike(batch->roi_);
  // Copy the roi.
  caffe_copy(batch->roi_.count(), batch->roi_.gpu_data(),
             top[1]->mutable_gpu_data());

  // Reshape to loaded roi_score.
  top[2]->ReshapeLike(batch->roi_score_);
  // Copy the roi_score.
  caffe_copy(batch->roi_score_.count(), batch->roi_score_.gpu_data(),
             top[2]->mutable_gpu_data());

  // Reshape to loaded roi_num.
  top[3]->ReshapeLike(batch->roi_num_);
  // Copy the roi_num.
  caffe_copy(batch->roi_num_.count(), batch->roi_num_.gpu_data(),
             top[3]->mutable_gpu_data());

  // Reshape to loaded label.
  top[4]->ReshapeLike(batch->label_);
  // Copy the label.
  caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
             top[4]->mutable_gpu_data());

  // Reshape to loaded box.
  top[5]->ReshapeLike(batch->box_);
  // Copy the box.
  caffe_copy(batch->box_.count(), batch->box_.gpu_data(),
             top[5]->mutable_gpu_data());

  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingRoIDataLayer);

}  // namespace caffe
