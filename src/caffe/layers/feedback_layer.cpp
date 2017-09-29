#include <boost/filesystem.hpp>
#include <cfloat>
#include <opencv2/opencv.hpp>
#include <vector>

#include "caffe/layers/feedback_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/more_math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Show_blob(const Dtype* data, const int channels, const int height,
               const int width, const string save_path,
               const string save_path_jet, const float threshold_ratio,
               const bool radioactive = false, const int fill = 0) {
  int rec_size = std::max(height, width) * 0.01;
  Dtype maxval = caffe_cpu_max_element(channels * height * width, data);
  Dtype sum = caffe_cpu_sum(channels * height * width, data);
  Dtype mean = sum / channels / height / width;

  Dtype threshold_value;
  if (threshold_ratio >= 0) {
    threshold_value = maxval * threshold_ratio;
  } else {
    threshold_value = sum / channels / height / width;
  }
  Dtype scale_factor = 255.0 / threshold_value;

  //-----------------------------------------------------------------------
  cv::Mat img_mat;
  cv::Mat img_mat_jet;
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
        int index = (c * height + h) * width + w;
        int index_mat = (h * width + w) * channels + c;
        Dtype value = abs(data[index]);
        // Dtype value = data[index] > 0 ? data[index] : 0;
        if (value >= threshold_value) {
          img_mat_data[index_mat] = 255;
          //-----------------------------------------------------------------------
          if (radioactive) {
            for (int cc = 0; cc < channels; cc++) {
              for (int hh = std::max(h - rec_size, 0);
                   hh < std::min(h + rec_size, height); ++hh) {
                for (int ww = std::max(w - rec_size, 0);
                     ww < std::min(w + rec_size, width); ++ww) {
                  int index_mat_r = (hh * width + ww) * channels + cc;
                  img_mat_data[index_mat_r] = 255;

                  // int index_r = (cc * height + hh) * width + ww;
                  // Dtype value_r = abs(data[index_r]);
                  // if (value_r > threshold_value) {
                  // for (int ccc = 0; ccc < channels; ccc++) {
                  // for (int hhh = min(h, hh); hhh <= std::max(h, hh); ++hhh) {
                  // for (int www = min(w, ww); www <= std::max(w, ww); ++www) {
                  // int index_mat_r =
                  //(hhh * width + www) * channels + ccc;
                  // img_mat_data[index_mat_r] = 255;
                  //}
                  //}
                  //}
                  //}
                }
              }
            }
          }
          //-----------------------------------------------------------------------
        } else {
          if (fill >= 0) {
            img_mat_data[index_mat] = fill;
          } else {
            img_mat_data[index_mat] = scale_factor * value;
          }
        }
      }
    }
  }

  cv::imwrite(save_path, img_mat);
  LOG(INFO) << "radioactive: " << radioactive
            << " threshold_ratio: " << threshold_ratio
            << " threshold_value: " << threshold_value << " maxval: " << maxval
            << " mean: " << mean;
  LOG(INFO) << "save_path: " << save_path;

  // cv::applyColorMap(img_mat, img_mat_jet, cv::COLORMAP_JET);
  // cv::imwrite(save_path_jet, img_mat_jet);

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
}

template <typename Dtype>
void vis_det(const Blob<Dtype>* det_blob, const Blob<Dtype>* img_blob,
             Blob<Dtype>& score_map, int save_id_) {
  stringstream save_dir;
  save_dir << "tmp/";
  boost::filesystem::create_directories(save_dir.str());

  const int num_det = det_blob->height();
  const int num_img = img_blob->num();
  const int channels_img = img_blob->channels();
  const int height_img = img_blob->height();
  const int width_img = img_blob->width();
  vector<int> score_map_shape(3);
  score_map_shape[0] = channels_img;
  score_map_shape[1] = height_img;
  score_map_shape[2] = width_img;
  score_map.Reshape(score_map_shape);

  const Dtype* det_data = det_blob->cpu_data();
  for (int i = 0; i < num_img; i++) {
    caffe_set(score_map.count(), Dtype(0.0), score_map.mutable_cpu_data());
    Dtype* map_data = score_map.mutable_cpu_data();
    for (int d = 0; d < num_det; d++) {
      //[image_id, label, confidence, xmin, ymin, xmax, ymax]
      const Dtype* det = det_data + d * 7;

      if (i != det[0]) {
        continue;
      }
      const Dtype det_score = det[2];
      det += 3;
      for (int x = det[0] * width_img; x <= det[2] * width_img; x++) {
        for (int y = det[1] * height_img; y <= det[3] * height_img; y++) {
          if (x < 0 || x >= width_img || y < 0 || y >= height_img) {
            continue;
          }
          map_data[y * width_img + x] += det_score;
        }
      }
    }

    stringstream save_path;
    stringstream save_path_jet;
    save_path << save_dir.str() << save_id_ << "_det_score.png";
    save_path_jet << save_dir.str() << save_id_ << "_jet_det_score.png";
    Show_blob(score_map.cpu_data(), 1, height_img, width_img, save_path.str(),
              save_path_jet.str(), 1, false, -1);

    caffe_set(score_map.count(), Dtype(0.0), score_map.mutable_cpu_data());
    map_data = score_map.mutable_cpu_data();
    const Dtype* image_data = img_blob->cpu_data();
    for (int h = 0; h < height_img; h++) {
      for (int w = 0; w < width_img; w++) {
        map_data[(0 * height_img + h) * width_img + w] =
            image_data[(0 * height_img + h) * width_img + w] + 103;
        map_data[(1 * height_img + h) * width_img + w] =
            image_data[(1 * height_img + h) * width_img + w] + 116;
        map_data[(2 * height_img + h) * width_img + w + 0] =
            image_data[(2 * height_img + h) * width_img + w] + 124;
      }
    }

    save_path.str(std::string());
    save_path_jet.str(std::string());
    save_path << save_dir.str() << save_id_ << ".png";
    save_path_jet << save_dir.str() << save_id_ << "_jet.png";
    Show_blob(score_map.cpu_data(), channels_img, height_img, width_img,
              save_path.str(), save_path_jet.str(), 1, false, -1);

    for (int d = 0; d < num_det; d++) {
      //[image_id, label, confidence, xmin, ymin, xmax, ymax]
      const Dtype* det = det_data + d * 7;

      if (i != det[0]) {
        continue;
      }
      det += 3;
      for (int x = det[0] * width_img; x <= det[2] * width_img; x++) {
        for (int y = det[1] * height_img; y <= det[1] * height_img; y++) {
          if (x < 0 || x >= width_img || y < 0 || y >= height_img) {
            continue;
          }
          map_data[(0 * height_img + y) * width_img + x] = 0;
          map_data[(1 * height_img + y) * width_img + x] = 0;
          map_data[(2 * height_img + y) * width_img + x] = 255;
        }
        for (int y = det[3] * height_img; y <= det[3] * height_img; y++) {
          if (x < 0 || x >= width_img || y < 0 || y >= height_img) {
            continue;
          }
          map_data[(0 * height_img + y) * width_img + x] = 0;
          map_data[(1 * height_img + y) * width_img + x] = 0;
          map_data[(2 * height_img + y) * width_img + x] = 255;
        }
      }
      for (int y = det[1] * height_img; y <= det[3] * height_img; y++) {
        for (int x = det[0] * width_img; x <= det[0] * width_img; x++) {
          if (x < 0 || x >= width_img || y < 0 || y >= height_img) {
            continue;
          }
          map_data[(0 * height_img + y) * width_img + x] = 0;
          map_data[(1 * height_img + y) * width_img + x] = 0;
          map_data[(2 * height_img + y) * width_img + x] = 255;
        }
        for (int x = det[2] * width_img; x <= det[2] * width_img; x++) {
          if (x < 0 || x >= width_img || y < 0 || y >= height_img) {
            continue;
          }
          map_data[(0 * height_img + y) * width_img + x] = 0;
          map_data[(1 * height_img + y) * width_img + x] = 0;
          map_data[(2 * height_img + y) * width_img + x] = 255;
        }
      }
    }

    save_path.str(std::string());
    save_path_jet.str(std::string());
    save_path << save_dir.str() << save_id_ << "_det_roi.png";
    save_path_jet << save_dir.str() << save_id_ << "_jet_det_roi.png";
    Show_blob(score_map.cpu_data(), channels_img, height_img, width_img,
              save_path.str(), save_path_jet.str(), 1, false, -1);

    save_id_++;
  }
}

template <typename Dtype>
void FeedbackLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), 3) << "top size should be 3.";

  CHECK_EQ(bottom[0]->num(), 1) << "num of 1-th blob should be 1";
  CHECK_EQ(bottom[0]->channels(), 1) << "channels of 1-th blob should be 1";
  CHECK_EQ(bottom[0]->width(), 7) << "width of 1-th blob should be 7";

  save_id_ = 0;
}

template <typename Dtype>
void FeedbackLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  // detection_out
  // data

  // roi_normlized
  // roi_score
  // roi_num

  CHECK_EQ(top.size(), 3) << "top size should be 3.";

  CHECK_EQ(bottom[0]->num(), 1) << "num of 1-th blob should be 1";
  CHECK_EQ(bottom[0]->channels(), 1) << "channels of 1-th blob should be 1";
  num_det_ = bottom[0]->height();
  CHECK_EQ(bottom[0]->width(), 7) << "width of 1-th blob should be 7";

  vector<int> top0_shape(2);
  top0_shape[0] = num_det_;
  top0_shape[1] = 5;
  top[0]->Reshape(top0_shape);

  vector<int> top1_shape(1);
  top1_shape[0] = num_det_;
  top[1]->Reshape(top1_shape);

  vector<int> top2_shape(1);
  top2_shape[0] = num_det_;
  top[2]->Reshape(top2_shape);
}

template <typename Dtype>
void FeedbackLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  const Dtype* det_data = bottom[0]->cpu_data();
  Dtype* roi_data = top[0]->mutable_cpu_data();
  Dtype* roi_score_data = top[1]->mutable_cpu_data();

  int idx_roi = 0;
  int idx_img = 0;
  vector<int> num_det_img;
  while (true) {
    int cur_num_det_img = 0;
    for (int d = 0; d < num_det_; d++) {
      //[image_id, label, confidence, xmin, ymin, xmax, ymax]
      const Dtype* det = det_data + d * 7;

      if (det[0] != idx_img) {
        continue;
      }

      // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
      Dtype* roi = roi_data + idx_roi * 5;

      roi[0] = det[0];
      roi[1] = det[3];
      roi[2] = det[4];
      roi[3] = det[5];
      roi[4] = det[6];

      roi_score_data[idx_roi] = Dtype(1.0);

      idx_roi++;
      cur_num_det_img++;
    }

    if (cur_num_det_img > 0) {
      num_det_img.push_back(cur_num_det_img);
      idx_img++;
    } else {
      break;
    }
  }

  CHECK_EQ(idx_img, num_det_img.size()) << "something error happen.";

  vector<int> top2_shape(1);
  top2_shape[0] = num_det_img.size();
  top[2]->Reshape(top2_shape);

  Dtype* roi_num_data = top[2]->mutable_cpu_data();
  for (size_t t = 0; t < num_det_img.size(); t++) {
    roi_num_data[t] = num_det_img[t];
  }

  vis_det(bottom[0], bottom[1], score_map_, save_id_);
  save_id_ += num_det_img.size();
}

template <typename Dtype>
void FeedbackLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
  for (size_t i = 0; i < bottom.size(); i++) {
    if (propagate_down[i]) {
      caffe_set(bottom[i]->count(), static_cast<Dtype>(0),
                bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_CLASS(FeedbackLayer);
REGISTER_LAYER_CLASS(Feedback);

}  // namespace caffe
