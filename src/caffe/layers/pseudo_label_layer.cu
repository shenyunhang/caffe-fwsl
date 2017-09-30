#include <boost/filesystem.hpp>
#include <cfloat>
#include <opencv2/opencv.hpp>
#include <vector>

#include "caffe/layers/pseudo_label_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/more_math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PseudoLabelLayer<Dtype>::Show_blob(const Dtype* data, const int channels,
                                        const int height, const int width,
                                        const string save_path,
                                        const string save_path_jet,
                                        const float threshold_ratio,
                                        const int fill) {
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
        Dtype value = fabs(data[index]);
        // Dtype value = data[index] > 0 ? data[index] : 0;
        if (value >= threshold_value) {
          img_mat_data[index_mat] = 255;
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
  LOG(INFO) << " threshold_ratio: " << threshold_ratio
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
void PseudoLabelLayer<Dtype>::vis_roi(const Blob<Dtype>* roi_blob,
                                      const Blob<Dtype>* roi_score_blob,
                                      const Blob<Dtype>* img_blob,
                                      Blob<Dtype>& score_map, int save_id_) {
  stringstream save_dir;
  save_dir << "tmp/";
  boost::filesystem::create_directories(save_dir.str());

  const int num_roi = roi_blob->num();
  const int num_cls = roi_score_blob->channels();
  const int num_img = img_blob->num();
  const int channels_img = img_blob->channels();
  const int height_img = img_blob->height();
  const int width_img = img_blob->width();
  vector<int> score_map_shape(3);
  score_map_shape[0] = channels_img;
  score_map_shape[1] = height_img;
  score_map_shape[2] = width_img;
  score_map.Reshape(score_map_shape);

  const Dtype* roi_data = roi_blob->cpu_data();
  const Dtype* roi_score_data = roi_score_blob->cpu_data();
  for (int i = 0; i < num_img; i++) {
    caffe_set(score_map.count(), Dtype(0.0), score_map.mutable_cpu_data());
    Dtype* map_data = score_map.mutable_cpu_data();
    for (int r = 0; r < num_roi; r++) {
      // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
      const Dtype* roi = roi_data + r * 5;

      if (i != roi[0]) {
        continue;
      }
      const Dtype* roi_score = roi_score_data + r * num_cls;
      roi += 1;
      for (int x = roi[0] * width_img; x <= roi[2] * width_img; x++) {
        for (int y = roi[1] * height_img; y <= roi[3] * height_img; y++) {
          if (x < 0 || x >= width_img || y < 0 || y >= height_img) {
            continue;
          }
          for (int c = 0; c < num_cls; c++) {
            map_data[y * width_img + x] += roi_score[c];
          }
        }
      }
    }

    stringstream save_path;
    stringstream save_path_jet;
    save_path << save_dir.str() << save_id_ << "_roi_score.png";
    save_path_jet << save_dir.str() << save_id_ << "_jet_roi_score.png";
    Show_blob(score_map.cpu_data(), 1, height_img, width_img, save_path.str(),
              save_path_jet.str(), 1, -1);

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
              save_path.str(), save_path_jet.str(), 1, -1);

    for (int r = 0; r < num_roi; r++) {
      // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
      const Dtype* roi = roi_data + r * 5;

      if (i != roi[0]) {
        continue;
      }
      roi += 1;
      for (int x = roi[0] * width_img; x <= roi[2] * width_img; x++) {
        for (int y = roi[1] * height_img; y <= roi[1] * height_img; y++) {
          if (x < 0 || x >= width_img || y < 0 || y >= height_img) {
            continue;
          }
          map_data[(0 * height_img + y) * width_img + x] = 0;
          map_data[(1 * height_img + y) * width_img + x] = 0;
          map_data[(2 * height_img + y) * width_img + x] = 255;
        }
        for (int y = roi[3] * height_img; y <= roi[3] * height_img; y++) {
          if (x < 0 || x >= width_img || y < 0 || y >= height_img) {
            continue;
          }
          map_data[(0 * height_img + y) * width_img + x] = 0;
          map_data[(1 * height_img + y) * width_img + x] = 0;
          map_data[(2 * height_img + y) * width_img + x] = 255;
        }
      }
      for (int y = roi[1] * height_img; y <= roi[3] * height_img; y++) {
        for (int x = roi[0] * width_img; x <= roi[0] * width_img; x++) {
          if (x < 0 || x >= width_img || y < 0 || y >= height_img) {
            continue;
          }
          map_data[(0 * height_img + y) * width_img + x] = 0;
          map_data[(1 * height_img + y) * width_img + x] = 0;
          map_data[(2 * height_img + y) * width_img + x] = 255;
        }
        for (int x = roi[2] * width_img; x <= roi[2] * width_img; x++) {
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
    save_path << save_dir.str() << save_id_ << "_roi_roi.png";
    save_path_jet << save_dir.str() << save_id_ << "_jet_roi_roi.png";
    Show_blob(score_map.cpu_data(), channels_img, height_img, width_img,
              save_path.str(), save_path_jet.str(), 1, -1);

    save_id_++;
  }
}

template <typename Dtype>
void PseudoLabelLayer<Dtype>::vis_roi_score(const Blob<Dtype>* roi_score_blob,
                                            Blob<Dtype>& score_map,
                                            int save_id_) {
  stringstream save_dir;
  save_dir << "tmp/";
  boost::filesystem::create_directories(save_dir.str());

  const int num_roi = roi_score_blob->num();
  const int num_cls = roi_score_blob->channels();
  vector<int> score_map_shape(2);
  score_map_shape[0] = num_roi;
  score_map_shape[1] = num_cls;
  score_map.Reshape(score_map_shape);

  const Dtype* roi_score_data = roi_score_blob->cpu_data();
  caffe_set(score_map.count(), Dtype(0.0), score_map.mutable_cpu_data());
  Dtype* map_data = score_map.mutable_cpu_data();
  for (int r = 0; r < num_roi; r++) {
    const Dtype* roi_score = roi_score_data + r * num_cls;
    // LOG(INFO) << det[0] << " " << det[1] << " " << det[2] << " " << det[3]
    //<< " " << det[4] << " " << det[5] << " " << det[6];
    for (int c = 0; c < num_cls; c++) {
      map_data[r * num_cls + c] = roi_score[c];
    }
  }

  stringstream save_path;
  stringstream save_path_jet;
  save_path << save_dir.str() << save_id_ << "_score.png";
  save_path_jet << save_dir.str() << save_id_ << "_jet_score.png";
  Show_blob(score_map.cpu_data(), 1, num_roi, num_cls, save_path.str(),
            save_path_jet.str(), 1, -1);
}

template <typename Dtype>
__global__ void Get_roi_det_blob(const int nthreads, const Dtype* roi_data,
                                 const Dtype* det_data, const int num_roi,
                                 const int num_det, Dtype* roi_det_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int roi_idx = index / num_det;
    const int det_idx = index % num_det;

    // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
    const Dtype* roi = roi_data + roi_idx * 5;
    //[image_id, label, confidence, xmin, ymin, xmax, ymax]
    const Dtype* det = det_data + det_idx * 7;

    if (roi[0] != det[0]) {
      roi_det_data[roi_idx * num_det + det_idx] = 0;
      continue;
    }

    roi += 1;
    det += 3;
    if (roi[0] > det[2] || roi[2] < det[0] || roi[1] > det[3] ||
        roi[3] < det[1]) {
      roi_det_data[roi_idx * num_det + det_idx] = 0;
      continue;
    }

    const Dtype intersect_x1 = max(roi[0], det[0]);
    const Dtype intersect_y1 = max(roi[1], det[1]);
    const Dtype intersect_x2 = min(roi[2], det[2]);
    const Dtype intersect_y2 = min(roi[3], det[3]);

    const Dtype intersect_h = intersect_y2 - intersect_y1;
    const Dtype intersect_w = intersect_x2 - intersect_x1;
    if (intersect_h == 0 && intersect_w == 0) {
      roi_det_data[roi_idx * num_det + det_idx] = 0;
      continue;
    }
    const Dtype intersect_size = intersect_h * intersect_w;
    const Dtype roi_size = (roi[2] - roi[0]) * (roi[3] - roi[1]);
    const Dtype det_size = (det[2] - det[0]) * (det[3] - det[1]);

    roi_det_data[roi_idx * num_det + det_idx] =
        intersect_size / (roi_size + det_size - intersect_size);
  }
}

template <typename Dtype>
void PseudoLabelLayer<Dtype>::top1forward(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  Get_roi_det_blob<
      Dtype><<<CAFFE_GET_BLOCKS(num_roi_ * num_det_), CAFFE_CUDA_NUM_THREADS>>>(
      num_roi_ * num_det_, bottom[1]->gpu_data(), bottom[3]->gpu_data(),
      num_roi_, num_det_, roi_det_.mutable_gpu_data());

  // LOG(INFO) << "roi_det_: " << roi_det_.asum_data();
  // LOG(INFO) << "roi_det_: " << roi_det_.count();
  // LOG(INFO) << "roi_det_: " << roi_det_.asum_data() / roi_det_.count();

  caffe_gpu_set(det_cls_.count(), Dtype(0.), det_cls_.mutable_gpu_data());
  Dtype* det_cls_data = det_cls_.mutable_cpu_data();
  //[image_id, label, confidence, xmin, ymin, xmax, ymax]
  const Dtype* det_data = bottom[3]->cpu_data();
  for (int i = 0; i < num_det_; ++i) {
    const Dtype* det = det_data + i * 7;
    const int label = det[1];
    if (label == -1 && num_det_ == 1) {
      caffe_gpu_set(top[1]->count(), Dtype(1.), top[1]->mutable_gpu_data());
      return;
    }
    // LOG(INFO)  << det[0]<<" "<< det[1]<< " "<<det[2]<< " "<<det[3]<< "
    // "<<det[4]<< " "<<det[5]<<" "<< det[6];
    CHECK_GT(label, 0) << "found background label in detection result.";
    CHECK_LT(label, num_cls_ + 1) << "label id is wrong.";
    det_cls_data[i * num_cls_ + label - 1] = det[2];
  }

  // LOG(INFO) << "det_cls_: " << det_cls_.asum_data();
  // LOG(INFO) << "det_cls_: " << det_cls_.count();
  // LOG(INFO) << "det_cls_: " << det_cls_.asum_data() / det_cls_.count();

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_roi_, num_cls_,
                        num_det_, Dtype(1.), roi_det_.gpu_data(),
                        det_cls_.gpu_data(), Dtype(0.),
                        top[1]->mutable_gpu_data());

  // LOG(INFO) << "top[1]: " << top[1]->asum_data();
  // LOG(INFO) << "top[1]: " << top[1]->count();
  // LOG(INFO) << "top[1]: " << top[1]->asum_data() / top[1]->count();

  const int top1_num = top[1]->num();
  const int top1_channels = top[1]->channels();

  Dtype* top1_data = top[1]->mutable_cpu_data();
  for (int j = 0; j < top1_channels; j++) {
    Dtype max_val = -FLT_MAX;
    for (int i = 0; i < top1_num; i++) {
      if (top1_data[i * top1_channels + j] > max_val) {
        max_val = top1_data[i * top1_channels + j];
      }
    }
    CHECK_GE(max_val, 0) << "max_val should >0.";
    if (max_val == 0) {
      continue;
    }
    for (int i = 0; i < top1_num; i++) {
      top1_data[i * top1_channels + j] /= max_val;
    }
  }

  caffe_gpu_powx(top[1]->count(), top[1]->gpu_data(), Dtype(0.5),
                 top[1]->mutable_gpu_data());

  // LOG(INFO) << "maxval: " << maxval;
  // LOG(INFO) << "top[1]: " << top[1]->asum_data();
  // LOG(INFO) << "top[1]: " << top[1]->count();
  // LOG(INFO) << "top[1]: " << top[1]->asum_data() / top[1]->count();
}

template <typename Dtype>
void PseudoLabelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  top0forward(bottom, top);
  if (bottom.size() == 5) {
    top1forward(bottom, top);
  }

  vis_roi(bottom[1], bottom[0], bottom[3], score_map_, save_id_);
  vis_roi_score(bottom[0], score_map_, save_id_);
  save_id_ += num_img_;
}

template <typename Dtype>
void PseudoLabelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
  for (size_t i = 0; i < bottom.size(); i++) {
    if (propagate_down[i]) {
      caffe_set(bottom[i]->count(), static_cast<Dtype>(0),
                bottom[i]->mutable_cpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PseudoLabelLayer);

}  // namespace caffe
