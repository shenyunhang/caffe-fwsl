#include <stdlib.h> /* srand, rand */
#include <boost/filesystem.hpp>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/repartition_layer.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/more_math_functions.hpp"

namespace caffe {

template <typename Dtype>
cv::Scalar gray2jet(Dtype f) {
  /*plot short rainbow RGB*/
  Dtype a = f / 0.25;            // invert and group
  int X = floor(a);              // this is the integer part
  int Y = floor(255 * (a - X));  // fractional part from 0 to 255
  int Z = floor(128 * (a - X));  // fractional part from 0 to 128
  int r, g, b;
  switch (X) {
    case 0:
      r = 0;
      g = Y;
      b = 128 - Z;
      break;
    case 1:
      r = Y;
      g = 255;
      b = 0;
      break;
    case 2:
      r = 255;
      g = 255 - Z;
      b = 0;
      break;
    case 3:
      r = 255;
      g = 128 - Z;
      b = 0;
      break;
    case 4:
      r = 255;
      g = 0;
      b = 0;
      break;
  }
  // opencv is bgr, not rgb
  return cv::Scalar(b, g, r);
}

template <typename Dtype>
Dtype GetDtypePrecision(Dtype value, Dtype precision) {
  return (floor((value * pow(10, precision) + 0.5)) / pow(10, precision));
}

template <typename Dtype>
void Show_rois(const Dtype *rois, const Dtype *scores, const Dtype *label,
               const int save_id, const int num_im, const int num_class,
               const int num_roi, const vector<string> voc_label,
               const string info, const float predict_threshold,
               const bool jet = false) {
  const int each_page_num = 10000;
  const int line_width = 4;

  cv::RNG rng(12345);
  stringstream save_path;
  stringstream save_dir;

  // load new image
  stringstream load_path;
  load_path << "tmp/" << save_id << "_.png";
  LOG(INFO) << "load png path: " << load_path.str();
  cv::Mat im_mat = cv::imread(load_path.str());
  cv::Mat im_mat_o = im_mat.clone();
  const int width_im = im_mat.cols;
  const int height_im = im_mat.rows;

  for (int c = 0; c < num_class; ++c) {
    if (label[c] <= predict_threshold) {
      continue;
    }

    // if (c == 7) {
    //} else {
    // continue;
    //}

    save_dir.str(std::string());
    save_dir << "tmp/" << voc_label[c] << "/" << save_id << "/";
    boost::filesystem::create_directories(save_dir.str());

    vector<int> show_ix;
    Dtype score_scale;
    for (int t = 0; t < num_roi; ++t) {
      Dtype max_roi_score = kMIN_SCORE;
      int max_roi_ix = -1;
      for (int r = 0; r < num_roi; ++r) {
        if (std::find(show_ix.begin(), show_ix.end(), r) == show_ix.end()) {
        } else {
          continue;
        }
        Dtype rois_score = scores[r * num_class + c];
        if (rois_score > max_roi_score) {
          max_roi_score = rois_score;
          max_roi_ix = r;
        }
      }
      show_ix.push_back(max_roi_ix);
      if (t == 0) {
        score_scale = max_roi_score;
      }
    }

    cv::Mat add_mat(im_mat.rows, im_mat.cols, CV_32FC1);
    cv::Mat count_mat(im_mat.rows, im_mat.cols, CV_32FC1, cv::Scalar(0));
    cv::Mat score_mat(im_mat.rows, im_mat.cols, CV_32FC1, cv::Scalar(0));
    cv::Mat norma_mat(im_mat.rows, im_mat.cols, CV_32FC1, cv::Scalar(0));
    // for (int r = 0, page = 0; r < num_roi; ++r) {
    for (int t = show_ix.size() - 1, page = 0; t >= 0; --t) {
      int r = show_ix[t];

      // rois: n x1 y1 x2 y2
      // rec: x y w h

      Dtype wstart = round(rois[5 * r + 1]);
      Dtype hstart = round(rois[5 * r + 2]);
      Dtype wend = round(rois[5 * r + 3]);
      Dtype hend = round(rois[5 * r + 4]);

      // caculate the inner and outer RoI coordinate
      Dtype width_roi = wend - wstart;
      Dtype height_roi = hend - hstart;
      Dtype context_scale = 1.8;
      // Dtype context_scale = sqrtf(2.0);
      Dtype width_roi_inner = width_roi / context_scale;
      Dtype height_roi_inner = height_roi / context_scale;
      Dtype width_roi_outer = width_roi * context_scale;
      Dtype height_roi_outer = height_roi * context_scale;
      Dtype wcenter = (wend + wstart) / 2.0;
      Dtype hcenter = (hend + hstart) / 2.0;

      Dtype wstart_inner = wcenter - width_roi_inner / 2.0;
      Dtype hstart_inner = hcenter - height_roi_inner / 2.0;
      Dtype wend_inner = wcenter + width_roi_inner / 2.0;
      Dtype hend_inner = hcenter + height_roi_inner / 2.0;

      Dtype wstart_outer = max(wcenter - width_roi_outer / 2.0, 0.0);
      Dtype hstart_outer = max(hcenter - height_roi_outer / 2.0, 0.0);
      Dtype wend_outer = min(wcenter + width_roi_outer / 2.0, width_im * 1.0);
      Dtype hend_outer = min(hcenter + height_roi_outer / 2.0, height_im * 1.0);

      cv::Rect rec = cv::Rect(wstart, hstart, wend - wstart, hend - hstart);
      cv::Rect rec_inner =
          cv::Rect(wstart_inner, hstart_inner, wend_inner - wstart_inner,
                   hend_inner - hstart_inner);
      cv::Rect rec_outer =
          cv::Rect(wstart_outer, hstart_outer, wend_outer - wstart_outer,
                   hend_outer - hstart_outer);

      Dtype rois_score = scores[r * num_class + c];
      //--------------------------------------------------------------------------
      // draw rectangle
      if (rois_score < 0) {
        rois_score = 0;
      }

      {
        rois_score = rois_score / score_scale;
        cv::rectangle(im_mat_o, rec, gray2jet(abs(rois_score)), line_width);
      }

      // cv::rectangle(im_mat_o, rec, gray2jet(abs(rois_score)), line_width);
      // cv::rectangle(im_mat_o, rec_inner, gray2jet(abs(rois_score)),
      // line_width);
      // cv::rectangle(im_mat_o, rec_outer, gray2jet(abs(rois_score)),
      // line_width);

      // 如果r+1整除each_page_num或者r是最后一个
      // if ((r + 1) % each_page_num == 0 || r == num_roi - 1) {
      if ((show_ix.size() - t) % each_page_num == 0 || t == 0) {
        save_path.str(std::string());
        save_path.precision(4);
        save_path << save_dir.str() << (rois_score > 0 ? "+" : "-")
                  << std::fixed << abs(rois_score) << "_" << page << info
                  << ".png";
        cv::imwrite(save_path.str(), im_mat_o);
        LOG(INFO) << "save_path: " << save_path.str();

        page++;
        im_mat.copyTo(im_mat_o);
      }

      //--------------------------------------------------------------------------
      // 计算 heat map
      cv::Mat mask_mat(im_mat.rows, im_mat.cols, CV_8UC1, cv::Scalar(0));
      cv::Mat roi_mat = mask_mat(rec);
      roi_mat = 1;

      add_mat = 1;
      // CV_32FC1 CV_32FC1 CV_32FC1 CV_8UC1
      cv::add(count_mat, add_mat, count_mat, mask_mat);

      add_mat = rois_score > 0 ? rois_score : 0;
      // CV_32FC1 CV_32FC1 CV_32FC1 CV_8UC1
      cv::add(score_mat, add_mat, score_mat, mask_mat);
    }

    //----------------------------------------------------------------------------
    if (!jet) continue;
    double maxVal, minVal, alpha, beta;
    cv::Mat u8_mat;
    cv::Mat cm_mat;

    // 保存 count map
    cv::minMaxLoc(count_mat, &minVal, &maxVal);
    alpha = 255.0 / (maxVal - minVal);
    beta = alpha * minVal;
    LOG(INFO) << "maxVal: " << maxVal << " minVal: " << minVal;

    count_mat.convertTo(u8_mat, CV_8UC1, alpha, beta);
    save_path.str(std::string());
    save_path << save_dir.str() << info << "rois_c.png";
    cv::imwrite(save_path.str(), u8_mat);

    cv::applyColorMap(u8_mat, cm_mat, cv::COLORMAP_JET);
    save_path.str(std::string());
    save_path << save_dir.str() << info << "rois_cj.png";
    cv::imwrite(save_path.str(), cm_mat);

    // 保存 score map
    cv::minMaxLoc(score_mat, &minVal, &maxVal);
    alpha = 255.0 / (maxVal - minVal);
    beta = alpha * minVal;
    LOG(INFO) << "maxVal: " << maxVal << " minVal: " << minVal;

    score_mat.convertTo(u8_mat, CV_8UC1, alpha, beta);
    save_path.str(std::string());
    save_path << save_dir.str() << info << "rois_s.png";
    cv::imwrite(save_path.str(), u8_mat);

    cv::applyColorMap(u8_mat, cm_mat, cv::COLORMAP_JET);
    save_path.str(std::string());
    save_path << save_dir.str() << info << "rois_sj.png";
    cv::imwrite(save_path.str(), cm_mat);

    // 保存 norm map
    cv::divide(score_mat, count_mat, norma_mat);
    cv::minMaxLoc(norma_mat, &minVal, &maxVal);
    alpha = 255.0 / (maxVal - minVal);
    beta = alpha * minVal;
    LOG(INFO) << "maxVal: " << maxVal << " minVal: " << minVal;

    norma_mat.convertTo(u8_mat, CV_8UC1, alpha, beta);
    save_path.str(std::string());
    save_path << save_dir.str() << info << "rois_n.png";
    cv::imwrite(save_path.str(), u8_mat);

    cv::applyColorMap(u8_mat, cm_mat, cv::COLORMAP_JET);
    save_path.str(std::string());
    save_path << save_dir.str() << info << "rois_nj.png";
    cv::imwrite(save_path.str(), cm_mat);
  }
}

template <typename Dtype>
void Show_blob(const Dtype *data, const int channels, const int height,
               const int width, const string save_cpg_path,
               const float threshold_, const int fill = 0) {
  Dtype maxval = -FLT_MAX;
  Dtype sum = 0;
  for (int c = 0; c < channels; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        int index = (c * height + h) * width + w;
        /*Dtype value = abs(data[index]);*/
        Dtype value = data[index] > 0 ? data[index] : 0;
        if (value > maxval) {
          maxval = value;
        }
        sum += value;
      }
    }
  }
  LOG(INFO) << "raw mean: " << sum / channels / height / width;

  if (threshold_ > 0) {
    maxval = maxval * threshold_;
  } else {
    maxval = sum / channels / height / width;
  }
  Dtype scale_factor = 255.0 / maxval;

  //-----------------------------------------------------------------------
  cv::Mat cpg_mat;
  if (channels == 3) {
    cpg_mat = cv::Mat(height, width, CV_8UC3);
  } else if (channels == 1) {
    cpg_mat = cv::Mat(height, width, CV_8UC1);
  } else {
    LOG(FATAL) << "channels should 1 or 3";
  }

  sum = 0;
  uchar *cpg_mat_data = cpg_mat.data;
  for (int c = 0; c < channels; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        int index = (c * height + h) * width + w;
        int index_mat = (h * width + w) * channels + c;
        /*Dtype value = abs(data[index]);*/
        Dtype value = data[index] > 0 ? data[index] : 0;
        if (value > maxval) {
          cpg_mat_data[index_mat] = 255;
          sum += maxval;
        } else {
          if (fill >= 0) {
            cpg_mat_data[index_mat] = fill;
          } else {
            cpg_mat_data[index_mat] = scale_factor * value;
          }
          sum += value;
        }
      }
    }
  }

  LOG(INFO) << "max_value: " << maxval
            << " mean: " << sum / channels / height / width;

  cv::imwrite(save_cpg_path, cpg_mat);

  //-----------------------------------------------------------------------
  /*const Dtype* cpg_cpu=cpg_blob->cpu_data();*/
  /*int total[26];*/
  /*for(int i=0;i<26;++i){*/
  /*total[i]=0;*/
  /*}*/
  /*for(int e=0;e<cpg_blob->count();e++){*/
  /*int level=int(cpg_cpu[e]/10);*/
  /*total[level]++;*/
  /*}*/
  /*for(int i=0;i<26;++i){*/
  /*std::cout << i<<":"<<total[i]<<" ";*/
  /*}*/
  /*std::cout<<std::endl;*/
  //-----------------------------------------------------------------------
}

template <typename Dtype>
__global__ void get_above_mask(const int N, const Dtype *const data,
                               Dtype *const mask, const Dtype threshold) {
  CUDA_KERNEL_LOOP(index, N) {
    if (data[index] >= threshold)
      mask[index] = 1;
    else
      mask[index] = 0;
  }
}

template <typename Dtype>
void RepartitionLayer<Dtype>::Score_map_crf() {
  crf_data_->ReshapeLike(*raw_data_);
  caffe_copy(crf_data_->count(), raw_data_->cpu_data(),
             crf_data_->mutable_cpu_data());

  //-----------------------------------------------------------------------
  const vector<string> layer_names;
  //= net_->layer_names();
  // int conv4_index = -1;
  // int conv5_index = -1;
  for (size_t i = 0; i < layer_names.size(); i++) {
    if (layer_names[i].compare("conv4_3") == 0) {
      // conv4_index = i;
    }
    if (layer_names[i].compare("conv5_3") == 0) {
      // conv5_index = i;
    }
  }

  const vector<int> conv4_bottom_ids;
  //= net_->bottom_ids(conv4_index);
  shared_ptr<Blob<Dtype> > conv4_blob;
  //= net_->blobs()[conv4_bottom_ids[0]];
  const vector<int> conv5_bottom_ids;
  //= net_->bottom_ids(conv5_index);
  shared_ptr<Blob<Dtype> > conv5_blob;
  //= net_->blobs()[conv5_bottom_ids[0]];

  shared_ptr<Blob<Dtype> > feature_blob = conv5_blob;

  //-----------------------------------------------------------------------
  vector<int> crf_cpg_shape = raw_data_->shape();
  crf_cpg_shape[1] = 2;
  crf_cpg_->Reshape(crf_cpg_shape);
  caffe_set(crf_cpg_->count(), Dtype(0), crf_cpg_->mutable_cpu_data());

  Blob<Dtype> fusion_blob;
  vector<int> fusion_shape = feature_blob->shape();
  fusion_shape[1] = 1;
  fusion_blob.Reshape(fusion_shape);
  caffe_set(fusion_blob.count(), Dtype(0), fusion_blob.mutable_cpu_data());

  const int a_offset = fusion_blob.offset(0, 1, 0, 0);
  for (int c = 0; c < feature_blob->channels(); ++c) {
    /*caffe_abs(a_offset, feature_blob->cpu_data() + c * a_offset,
     * crf_cpg_->mutable_cpu_diff());*/
    /*caffe_add(a_offset, crf_cpg_->cpu_diff(), crf_cpg_->cpu_data(),
     * crf_cpg_->mutable_cpu_data());*/
    caffe_add(a_offset, feature_blob->cpu_data() + c * a_offset,
              fusion_blob.cpu_data(), fusion_blob.mutable_cpu_data());
  }

  caffe_cpu_interp2<Dtype, false>(
      1, fusion_blob.cpu_data(), 0, 0, fusion_shape[2], fusion_shape[3],
      fusion_shape[2], fusion_shape[3], crf_cpg_->mutable_cpu_data(), 0, 0,
      crf_cpg_shape[2], crf_cpg_shape[3], crf_cpg_shape[2], crf_cpg_shape[3]);

  const Dtype max_value = max_element_(crf_cpg_->cpu_data(), crf_cpg_->count());
  const Dtype scale_factor = 1 / (max_value);
  crf_cpg_->scale_data(scale_factor);
  Dtype *crf_cpg = crf_cpg_->mutable_cpu_data();
  for (int i = 0; i < crf_cpg_->count(); ++i) {
    if (crf_cpg[i] < 0.0) {
      crf_cpg[i] = 0;
    }
  }

  if (debug_info_) {
    LOG(INFO) << "max_value: " << (max_value);
  }

  //-----------------------------------------------------------------------
  crf_data_dim_->Reshape(1, 2, 1, 1);
  crf_data_dim_->mutable_cpu_data()[0] = raw_data_->shape(2);
  crf_data_dim_->mutable_cpu_data()[1] = raw_data_->shape(3);

  //-----------------------------------------------------------------------
  crf_layer_->Forward(crf_bottom_vec_, crf_top_vec_);

  stringstream save_path;
  save_path << "tmp/" << pass_im_ << "_feat.png";
  /*Show_blob(crf_output, false, n, rows, cols, channels,
   * save_crf_cpg_path.str());*/
  Show_blob(crf_output_->cpu_data(), 1, crf_output_->height(),
            crf_output_->width(), save_path.str(), 1);

  stringstream save_fusion_path;
  save_fusion_path << "tmp/" << pass_im_ << "_fusion.png";
  Show_blob(crf_cpg_->cpu_data(), 1, crf_cpg_->height(), crf_cpg_->width(),
            save_fusion_path.str(), 1);
}

template <typename Dtype>
void RepartitionLayer<Dtype>::Repartition_crf(const int label) {
  const vector<int> start_bottom_ids;
  //= net_->bottom_ids(start_index_);
  shared_ptr<Blob<Dtype> > im_blob;
  //= net_->blobs()[start_bottom_ids[0]];

  crf_data_->ReshapeLike(*im_blob);
  caffe_copy(crf_data_->count(), im_blob->cpu_data(),
             crf_data_->mutable_cpu_data());

  vector<int> cpg_shape = im_blob->shape();
  cpg_shape[1] = 2;
  crf_cpg_->Reshape(cpg_shape);

  /*caffe_copy(crf_cpg_->count(), im_blob->cpu_diff(),*/
  /*crf_cpg_->mutable_cpu_data());*/

  const int a_offset = crf_cpg_->offset(0, 1, 0, 0);
  caffe_abs(a_offset, im_blob->cpu_diff(), crf_cpg_->mutable_cpu_data());
  caffe_abs(a_offset, im_blob->cpu_diff() + 1 * a_offset,
            crf_cpg_->mutable_cpu_diff());
  caffe_add(a_offset, crf_cpg_->cpu_diff(), crf_cpg_->cpu_data(),
            crf_cpg_->mutable_cpu_data());
  caffe_abs(a_offset, im_blob->cpu_diff() + 2 * a_offset,
            crf_cpg_->mutable_cpu_diff());
  caffe_add(a_offset, crf_cpg_->cpu_diff(), crf_cpg_->cpu_data(),
            crf_cpg_->mutable_cpu_data());

  const Dtype *max_value = std::max_element(
      crf_cpg_->cpu_data(), crf_cpg_->cpu_data() + crf_cpg_->count());

  if (debug_info_) {
    LOG(INFO) << "Repartition_crf max_value: " << *max_value;
  }
  const Dtype scale_factor = 1 / (*max_value);
  crf_cpg_->scale_data(scale_factor);
  Dtype *crf_cpg = crf_cpg_->mutable_cpu_data();
  for (int i = 0; i < crf_cpg_->count(); ++i) {
    if (crf_cpg[i] < crf_threshold_) {
      crf_cpg[i] = 0;
    }
  }

  /*caffe_cpu_axpby(a_offset, Dtype(-1), crf_cpg_->cpu_data(), Dtype(0),
   * crf_cpg_->mutable_cpu_data() + a_offset);*/
  /*caffe_add_scalar(a_offset, Dtype(1), crf_cpg_->mutable_cpu_data() +
   * a_offset);*/

  crf_data_dim_->Reshape(1, 2, 1, 1);
  crf_data_dim_->mutable_cpu_data()[0] = im_blob->shape(2);
  crf_data_dim_->mutable_cpu_data()[1] = im_blob->shape(3);

  crf_layer_->Forward(crf_bottom_vec_, crf_top_vec_);

  stringstream save_crf_cpg_path;
  save_crf_cpg_path << "tmp/" << pass_im_ << "_" << voc_label_[label]
                    << "_crf.png";
  /*Show_blob(crf_output, false, n, rows, cols, channels,
   * save_crf_cpg_path.str());*/
  Show_blob(crf_output_->cpu_data(), 1, crf_output_->height(),
            crf_output_->width(), save_crf_cpg_path.str(), 1);
}

template <typename Dtype>
__global__ void InitFilter_Test(const int count, const Dtype *const label_data,
                                const int num_class, Dtype *const top_data) {
  CUDA_KERNEL_LOOP(index, count) {
    const int c = index % num_class;
    // TODO(YH): What is the correct threshold in test
    if (label_data[c] > 0.00001) {
      top_data[index] = 1;
    } else {
      top_data[index] = 0;
    }
  }
}

template <typename Dtype>
void RepartitionLayer<Dtype>::InitFilter(const Dtype *const label_gpu_data,
                                         Dtype *const filter_gpu_data) {
  switch (this->layer_param_.cpg_param().mode()) {
    case CPGParameter_Mode_PRED:
    case CPGParameter_Mode_CPG_POOLING:
      if (this->phase_ == TRAIN) {
        caffe_gpu_set(num_roi_ * num_class_, Dtype(1), filter_gpu_data);
      } else {
        // NOLINT_NEXT_LINE(whitespace/operators)
        InitFilter_Test<Dtype><<<CAFFE_GET_BLOCKS(num_roi_ * num_class_),
                                 CAFFE_CUDA_NUM_THREADS>>>(
            num_roi_ * num_class_, label_gpu_data, num_class_, filter_gpu_data);
      }
      break;
    case CPGParameter_Mode_CRF:
      break;
    default:
      LOG(FATAL) << "Unknown mode.";
  }
}

template <typename Dtype>
__global__ void ScoreBBoxes(const int num_roi, const Dtype *const cpg_data,
                            const int num, const int channels, const int height,
                            const int width, const Dtype *const rois_data,
                            const int num_class, const int label,
                            const Dtype threshold, const Dtype min_density,
                            const Dtype all_mass, Dtype *const top_data,
                            const int r = 5) {
  CUDA_KERNEL_LOOP(index, num_roi) {
    const int rois_index = index;

    const Dtype *const roi = rois_data + 5 * rois_index;
    const int wstart = max(int(roi[1]), 0);
    const int hstart = max(int(roi[2]), 0);
    const int wend = min(int(roi[3]), width);
    const int hend = min(int(roi[4]), height);

    /*Dtype sum = 0;*/
    /*Dtype maxval = -FLT_MAX;*/
    Dtype mass = 0;
    for (int c = 0; c < channels; ++c) {
      const Dtype *gradient = cpg_data + c * height * width;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          /*sum += gradient[h * width + w];*/
          /*sum += exp(max(r * gradient[h * width + w], kLOG_THRESHOLD));*/
          /*if (maxval < gradient[h * width + w]) {*/
          /*maxval = gradient[h * width + w];*/
          /*}*/
          if (threshold < gradient[h * width + w]) {
            mass++;
          }
        }
      }
    }
    Dtype s = (hend - hstart) * (wend - wstart);
    Dtype density = 1.0 * mass / s / channels;
    top_data[rois_index * num_class + label] = density + 1.0 * mass / all_mass;
  }
}

template <typename Dtype>
__global__ void CPGPooling(const int num_roi, const Dtype *cpg_data,
                           const int height_im, const int width_im,
                           const Dtype *rois_data, const int num_class,
                           const int cls_id, const Dtype min_density,
                           const Dtype min_mass, Dtype *const top_data) {
  CUDA_KERNEL_LOOP(index, num_roi) {
    int rois_index = index;

    rois_data += 5 * rois_index;
    int wstart = round(rois_data[1]);
    int hstart = round(rois_data[2]);
    int wend = round(rois_data[3]);
    int hend = round(rois_data[4]);

    // Check RoI
    if (wstart >= 0 && hstart >= 0 && wstart < wend && hstart < hend &&
        wend < width_im && hend < height_im) {
    } else {
      top_data[rois_index * num_class + cls_id] = kMIN_SCORE;
      // 这里面是for循环，用return会中断后续的循环
      continue;
    }

    // caculate the inner and outer RoI coordinate
    Dtype width_roi = wend - wstart;
    Dtype height_roi = hend - hstart;
    Dtype context_scale = 1.8;
    // Dtype context_scale = sqrtf(2.0);
    Dtype width_roi_inner = 1.0 * width_roi / context_scale;
    Dtype height_roi_inner = 1.0 * height_roi / context_scale;
    Dtype width_roi_outer = 1.0 * width_roi * context_scale;
    Dtype height_roi_outer = 1.0 * height_roi * context_scale;
    Dtype wcenter = 1.0 * (wend + wstart) / 2.0;
    Dtype hcenter = 1.0 * (hend + hstart) / 2.0;

    int wstart_inner = round(wcenter - width_roi_inner / 2.0);
    int hstart_inner = round(hcenter - height_roi_inner / 2.0);
    int wend_inner = round(wcenter + width_roi_inner / 2.0);
    int hend_inner = round(hcenter + height_roi_inner / 2.0);

    int wstart_outer = round(max(wcenter - width_roi_outer / 2.0, 0.0));
    int hstart_outer = round(max(hcenter - height_roi_outer / 2.0, 0.0));
    int wend_outer =
        round(min(wcenter + width_roi_outer / 2.0, width_im - 1.0));
    int hend_outer =
        round(min(hcenter + height_roi_outer / 2.0, height_im - 1.0));

    width_roi = wend - wstart + 1;
    height_roi = hend - hstart + 1;
    width_roi_inner = wend_inner - wstart_inner + 1;
    height_roi_inner = hend_inner - hstart_inner + 1;
    width_roi_outer = wend_outer - wstart_outer + 1;
    height_roi_outer = hend_outer - hstart_outer + 1;

    // a1-a2-a3+a4
    Dtype a1, a2, a3, a4;

    // CPG sum of RoI
    a1 = cpg_data[hend * width_im + wend];
    a2 = (wstart - 1 >= 0) ? cpg_data[hend * width_im + (wstart - 1)] : 0;
    a3 = (hstart - 1 >= 0) ? cpg_data[(hstart - 1) * width_im + wend] : 0;
    a4 = (hstart - 1 >= 0 && wstart - 1 >= 0)
             ? cpg_data[(hstart - 1) * width_im + (wstart - 1)]
             : 0;
    Dtype sum_roi = a1 - a2 - a3 + a4;

    // CPG sum of inner RoI
    a1 = cpg_data[hend_inner * width_im + wend_inner];
    a2 = (wstart_inner - 1 >= 0)
             ? cpg_data[hend_inner * width_im + (wstart_inner - 1)]
             : 0;
    a3 = (hstart_inner - 1 >= 0)
             ? cpg_data[(hstart_inner - 1) * width_im + wend_inner]
             : 0;
    a4 = (hstart_inner - 1 >= 0 && wstart_inner - 1 >= 0)
             ? cpg_data[(hstart_inner - 1) * width_im + (wstart_inner - 1)]
             : 0;
    Dtype sum_inner = a1 - a2 - a3 + a4;

    // CPG sum of outer RoI
    a1 = cpg_data[hend_outer * width_im + wend_outer];
    a2 = (wstart_outer - 1 >= 0)
             ? cpg_data[hend_outer * width_im + (wstart_outer - 1)]
             : 0;
    a3 = (hstart_outer - 1 >= 0)
             ? cpg_data[(hstart_outer - 1) * width_im + wend_outer]
             : 0;
    a4 = (hstart_outer - 1 >= 0 && wstart_outer - 1 >= 0)
             ? cpg_data[(hstart_outer - 1) * width_im + (wstart_outer - 1)]
             : 0;
    Dtype sum_outer = a1 - a2 - a3 + a4;

    // area size
    Dtype area_roi = height_roi * width_roi;
    Dtype area_inner = height_roi_inner * width_roi_inner;
    Dtype area_outer = height_roi_outer * width_roi_outer;

    Dtype area_frame = max(area_roi - area_inner, Dtype(1));
    Dtype area_context = max(area_outer - area_roi, Dtype(1));

    //-----------------------------------------------------------------------
    // current best
    Dtype score = (sum_roi - sum_inner) / sqrt(area_frame) -
                  (sum_outer - sum_roi) / sqrt(area_context);

    // bad at test debug
    // Dtype score = (sum_roi - sum_inner) - (sum_outer - sum_roi);

    // (msra 0223):
    // Dtype score = ((sum_roi - 2.0 * (sum_outer - sum_roi)) *
    //(2.0 * (sum_roi - sum_inner) - sum_inner)) /
    // area_roi;
    // if ((sum_roi - 2.0 * (sum_outer - sum_roi)) < 0 &&
    //(2.0 * (sum_roi - sum_inner) - sum_inner) < 0) {
    // score = -1.0 * score;
    //}

    // (msra 0101): bad
    // Dtype score = sqrt((sum_roi - sum_inner) / area_frame) -
    //               sqrt((sum_outer - sum_roi) / area_context);

    // (msra 12.30): very bad
    // Dtype score =
    //    (sum_roi - sum_inner) / area_frame - (sum_outer - sum_roi) /
    // area_context;

    // (msra 12.29): bad
    // Dtype score = ((sum_roi - sum_inner) - (sum_outer - sum_roi)) /
    // area_frame;

    // (msra 0105): bad than (msra 12.29)
    // Dtype score = ((sum_roi - sum_inner) - (sum_outer - sum_roi)) /
    // sqrt(area_frame);

    //-----------------------------------------------------------------------

    // if (sum_roi < min_mass) score = kMIN_SCORE;

    top_data[rois_index * num_class + cls_id] = score;
  }
}

template <typename Dtype>
void integral_cpu(const Dtype *src, Dtype *sum, const int height,
                  const int width) {
  Dtype s = 0;
  for (int x = 0; x < width; x++) {
    s += src[x];
    sum[x] = s;
  }
  src += width;
  sum += width;
  for (int y = 1; y < height; y++, src += width, sum += width) {
    s = 0;
    for (int x = 0; x < width; x++) {
      s += src[x];
      sum[x] = sum[x - width] + s;
    }
  }
}

template <typename Dtype>
bool RepartitionLayer<Dtype>::Need_Order(const int cls_id, const Dtype label,
                                         const Dtype predict) {
  if (cls_id == ignore_label_) return false;
  // assum score is betwween 0 ~ 1
  if (this->phase_ == TRAIN) {
    if (label < 0.5) return false;
    if (is_order_ && predict > predict_order_) {
      return true;
    } else {
      return false;
    }
  } else if (this->phase_ == TEST) {
    return false;
  } else {
    LOG(FATAL) << "unkown phase: " << this->phase_;
  }
  LOG(FATAL) << "We should not arrive here!";
  return false;
}

template <typename Dtype>
void RepartitionLayer<Dtype>::After() {
  // this should not in the Reshape function
  // as Reshape function will be call before start
  pass_im_ += num_im_;

  if (max_num_im_cpg_ > 0 && max_num_im_cpg_ <= pass_im_) {
    is_cpg_ = false;
  }

  if (pass_im_ % display_ == 0) {
    LOG(INFO) << "is_cpg: " << is_cpg_;
  }

  if (pass_im_ % display_ == 0 && this->phase_ != TEST) {
    if (is_order_) {
      order_threshold_ =
          1.0 - 1.0 * (int(1.0 * pass_im_ / order_step_) + 1) / order_K_;
      if (order_threshold_ < 0) order_threshold_ = 0;
      LOG(INFO) << "#im:" << pass_im_
                << " order_threshold_: " << order_threshold_;
    }
  }
}

template <typename Dtype>
void RepartitionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  if (!is_cpg_) {
    return;
  }

  InitFilter(bottom[bottom_index_["label"]]->gpu_data(),
             filter_.mutable_gpu_data());

  LOG_IF(INFO, debug_info_) << "------------------start-----------------------";
  LOG_IF(INFO, debug_info_) << "pass_im_: " << pass_im_;
  //-----------------------------------------------------------------------
  //-----------------------------------------------------------------------
  const Dtype *bottom_label = bottom[bottom_index_["label"]]->cpu_data();
  const Dtype *bottom_predict = bottom[bottom_index_["predict"]]->cpu_data();
  const Dtype *rois_score = bottom[bottom_index_["rois_score"]]->cpu_data();

  Dtype *pos_label_data;
  Dtype *neg_label_data;
  if (is_order_ && top.size() == 3) {
    pos_label_data = top[1]->mutable_cpu_data();
    neg_label_data = top[2]->mutable_cpu_data();
  }

  int re_num = 0;
  for (size_t gt_id = 0; gt_id < gt_class_.size(); ++gt_id) {
    int cls_id = gt_class_[gt_id];
    int index = cls_id;
    LOG_IF(INFO, debug_info_) << "class: " << voc_label_[cls_id]
                              << "\t\tlabel: " << bottom_label[index]
                              << " score: " << bottom_predict[index];

    //-----------------------------------------------------------------------
    // propocess data
    switch (this->layer_param_.cpg_param().mode()) {
      case CPGParameter_Mode_PRED:
      case CPGParameter_Mode_CPG_POOLING: {
        caffe_gpu_set(raw_data_->count(), Dtype(0),
                      raw_data_->mutable_gpu_data());
        caffe_gpu_set(raw_data_->count(), Dtype(0),
                      raw_data_->mutable_gpu_diff());
        caffe_gpu_abs(size_cpg_,
                      bottom[bottom_index_["cpg"]]->gpu_data() +
                          bottom[bottom_index_["cpg"]]->offset(0, cls_id, 0, 0),
                      raw_data_->mutable_gpu_data());

        // TODO(YH): order_threshold_
        if (order_threshold_ > 0 &&
            Need_Order(cls_id, bottom_label[index], bottom_predict[index])) {
          caffe_cpu_threshold_bbox(raw_data_, bboxes_, fg_threshold_, cls_id);
          Dtype max_size = 0;
          const Dtype *bbox = bboxes_->cpu_data();
          for (int box_id = 0; box_id < max_bb_per_cls_; ++box_id) {
            if (bbox[0] == -1) break;
            Dtype size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
            if (max_size < size) max_size = size;
            bbox += bboxes_->offset(1);
          }
          Dtype max_scale = max_size / (height_im_ * width_im_);
          if (max_scale < order_threshold_) {
            pos_label_data[cls_id] = -1;
            neg_label_data[cls_id] = -1;
            continue;
          }
        }
      } break;
      case CPGParameter_Mode_CRF:
        break;
      default:
        LOG(FATAL) << "Unknown mode.";
    }

    ++re_num;

    switch (this->layer_param_.cpg_param().mode()) {
      case CPGParameter_Mode_PRED: {
        const Dtype maxval =
            caffe_cpu_max_element(size_cpg_, raw_data_->cpu_data());
        const Dtype threshold = maxval * fg_threshold_;

        // NOLINT_NEXT_LINE(whitespace/operators)
        get_above_mask<
            Dtype><<<CAFFE_GET_BLOCKS(size_cpg_), CAFFE_CUDA_NUM_THREADS>>>(
            size_cpg_, raw_data_->gpu_data(), raw_data_->mutable_gpu_diff(),
            threshold);
        Dtype im_mass;
        caffe_gpu_asum(size_cpg_, raw_data_->gpu_diff(), &im_mass);
        const Dtype im_density = 1.0 * im_mass / height_im_ / width_im_;

        LOG_IF(INFO, debug_info_)
            << "maxval: " << maxval << " threshold: " << threshold
            << " im_mass: " << im_mass << " im_density: " << im_density;
        LOG_IF(INFO, debug_info_) << "ScoreBBoxes:";
        const Dtype min_density = im_density * density_threshold_;
        // NOLINT_NEXT_LINE(whitespace/operators)
        ScoreBBoxes<
            Dtype><<<CAFFE_GET_BLOCKS(num_roi_), CAFFE_CUDA_NUM_THREADS>>>(
            num_roi_, raw_data_->gpu_data(), 1, 1, height_im_, width_im_,
            bottom[bottom_index_["rois"]]->gpu_data(), num_class_, cls_id,
            threshold, min_density, im_mass, filter_.mutable_gpu_data());
      } break;
      case CPGParameter_Mode_CPG_POOLING: {
        int max_value_index;
        caffe_gpu_amax(size_cpg_, raw_data_->gpu_data(), &max_value_index);
        max_value_index--;
        const Dtype maxval = raw_data_->cpu_data()[max_value_index];
        const Dtype threshold = maxval * fg_threshold_;

        caffe_gpu_binary(size_cpg_, raw_data_->gpu_data(),
                         raw_data_->mutable_gpu_diff(), threshold);

        Dtype im_mass;
        caffe_gpu_asum(size_cpg_, raw_data_->gpu_diff(), &im_mass);
        const Dtype im_density = 1.0 * im_mass / height_im_ / width_im_;

        // CHECK_GE(maxval, 0) << "maxval should be greater than 0.";
        LOG_IF(INFO, debug_info_)
            << "maxval: " << maxval << " threshold: " << threshold
            << " im_mass: " << im_mass << " im_density: " << im_density;
        LOG_IF(INFO, debug_info_) << "SumBBoxes:";

        integral_cpu(raw_data_->cpu_diff(), raw_data_->mutable_cpu_data(),
                     height_im_, width_im_);

        CHECK_EQ(raw_data_->cpu_data()[size_cpg_ - 1], im_mass)
            << "Should be equal.";

        // NOLINT_NEXT_LINE(whitespace/operators)
        CPGPooling<
            Dtype><<<CAFFE_GET_BLOCKS(num_roi_), CAFFE_CUDA_NUM_THREADS>>>(
            num_roi_, raw_data_->gpu_data(), height_im_, width_im_,
            bottom[bottom_index_["rois"]]->gpu_data(), num_class_, cls_id,
            im_density * density_threshold_, im_mass * mass_threshold_,
            filter_.mutable_gpu_data());

        Dtype re_predict = 0;

        // normalization max value to |1|
        if (true) {
          Dtype *filter_data = filter_.mutable_cpu_data();
          Dtype max_value = 0;
          Dtype min_value = 0;
          for (int roi_id = 0; roi_id < num_roi_; roi_id++) {
            Dtype value = filter_data[roi_id * num_class_ + cls_id];
            if (value > max_value) {
              max_value = value;
            }
            if (value < min_value && value != kMIN_SCORE) {
              min_value = value;
            }
          }
          // CHECK_GE(max_value, 0) << "max_value should be greater than 0.";
          // CHECK_GE(min_value, -1) << "min_value should be -1.";
          if (max_value > 0 && min_value < 0) {
            for (int roi_id = 0; roi_id < num_roi_; roi_id++) {
              Dtype value = filter_data[roi_id * num_class_ + cls_id];
              if (value == kMIN_SCORE) {
                value = -1;
              } else {
                value = value > 0 ? value / max_value : value / (-min_value);
              }
              // value = value > 0 ? value / max_value : -1;
              filter_data[roi_id * num_class_ + cls_id] = value;

              re_predict +=
                  value > 0 ? value * rois_score[roi_id * num_class_ + cls_id]
                            : 0;
            }
          } else if (max_value > 0 && min_value == 0) {
            for (int roi_id = 0; roi_id < num_roi_; roi_id++) {
              Dtype value = filter_data[roi_id * num_class_ + cls_id];
              if (value == kMIN_SCORE) {
                value = -1;
              } else {
                value = value / max_value;
              }
              filter_data[roi_id * num_class_ + cls_id] = value;

              re_predict += value * rois_score[roi_id * num_class_ + cls_id];
            }
          } else {
            for (int roi_id = 0; roi_id < num_roi_; roi_id++) {
              filter_data[roi_id * num_class_ + cls_id] = 1.0;
              re_predict += 1 * rois_score[roi_id * num_class_ + cls_id];
            }
          }
        }

        // drop
        if (false) {
          if (this->phase_ == TRAIN) {
            double secret;
            caffe_rng_uniform(1, 0.0, 1.0, &secret);

            if (secret < min(max(bottom_predict[index] - re_predict, 0.0) +
                                 1.0 * pass_im_ / (5011 * 2 * 20),
                             1.0)) {
              Dtype *filter_data = filter_.mutable_cpu_data();
              for (int roi_id = 0; roi_id < num_roi_; roi_id++) {
                filter_data[roi_id * num_class_ + cls_id] = 1;
              }
            }
          }
        }

        if (debug_info_) {
          const Dtype *filter_data = filter_.cpu_data();
          const Dtype *rois_data = bottom[bottom_index_["rois"]]->cpu_data();
          int a0 = 0;
          int b0 = 0;
          int e0 = 0;
          for (int roi_id = 0; roi_id < num_roi_; roi_id++) {
            Dtype value = filter_data[roi_id * num_class_ + cls_id];
            std::cout << value << "(" << rois_data[roi_id * 5 + 1] << " "
                      << rois_data[roi_id * 5 + 2] << " "
                      << rois_data[roi_id * 5 + 3] << " "
                      << rois_data[roi_id * 5 + 4] << ") ";
            if (value > 0)
              a0++;
            else if (value < 0)
              b0++;
            else
              e0++;
          }
          std::cout << std::endl;
          std::cout << "a0: " << a0 << " b0: " << b0 << " e0: " << e0
                    << std::endl;
          std::cout << "re_predict: " << re_predict << std::endl;
        }
      } break;
      case CPGParameter_Mode_CRF:
        break;
      default:
        LOG(FATAL) << "Unknown mode.";
    }
  }

  //----------------------------------------------------------------------
  // Show patch
  //----------------------------------------------------------------------
  if (debug_info_) {
    Show_rois(bottom[bottom_index_["rois"]]->cpu_data(), filter_.cpu_data(),
              bottom_label, pass_im_, num_im_, num_class_, num_roi_, voc_label_,
              "_w_", predict_threshold_, true);

    Show_rois(bottom[bottom_index_["rois"]]->cpu_data(), rois_score,
              bottom_label, pass_im_, num_im_, num_class_, num_roi_, voc_label_,
              "_s_", predict_threshold_, true);

    caffe_gpu_mul(num_class_ * num_roi_, filter_.gpu_data(),
                  bottom[bottom_index_["rois_score"]]->gpu_data(),
                  filter_.mutable_gpu_diff());
    Show_rois(bottom[bottom_index_["rois"]]->cpu_data(), filter_.cpu_diff(),
              bottom_label, pass_im_, num_im_, num_class_, num_roi_, voc_label_,
              "_ws_", predict_threshold_, true);
  }

  // get the final output from filter
  switch (this->layer_param_.cpg_param().mode()) {
    case CPGParameter_Mode_PRED:
      top[0]->CopyFrom(filter_, false, false);
      if (bottom.size() == bottom_index_["io"] + 1) {
        if (re_num > 0) {
          int save_id = int(bottom[bottom_index_["io"]]->cpu_data()[0]);
          LOG_IF(INFO, debug_info_) << "save_id: " << save_id;

          BlobProto save_blob;
          top[0]->ToProto(&save_blob, false);
          stringstream save_path;
          save_path << "data/cpg_cache/" << save_id;
          WriteProtoToBinaryFile(save_blob, save_path.str());
        }

        caffe_gpu_or(num_roi_ * num_class_,
                     bottom[bottom_index_["filter"]]->gpu_data(),
                     top[0]->gpu_data(), top[0]->mutable_gpu_data());
      }
      break;
    case CPGParameter_Mode_CPG_POOLING:
      top[0]->CopyFrom(filter_, false, false);
      if (bottom.size() == bottom_index_["io"] + 1) {
        if (re_num > 0) {
          int save_id = int(bottom[bottom_index_["io"]]->cpu_data()[0]);

          BlobProto save_blob;
          top[0]->ToProto(&save_blob, false);
          stringstream save_path;
          save_path << "data/cpg_cache/" << save_id;
          WriteProtoToBinaryFile(save_blob, save_path.str());

          LOG_IF(INFO, debug_info_) << "save_id: " << save_id
                                    << " save_path: " << save_path;
        }
        caffe_gpu_maximum(num_roi_ * num_class_,
                          bottom[bottom_index_["filter"]]->gpu_data(),
                          top[0]->gpu_data(), top[0]->mutable_gpu_data());

        // caffe_gpu_add(num_roi_ * num_class_,
        // bottom[bottom_index_["filter"]]->gpu_data(),
        // top[0]->gpu_data(), top[0]->mutable_gpu_data());
        // caffe_gpu_threshold(num_roi_ * num_class_, top[0]->gpu_data(),
        // top[0]->mutable_gpu_data(), Dtype(1), true);
      }
      break;
    case CPGParameter_Mode_CRF:
      break;
    default:
      LOG(FATAL) << "Unknown mode.";
  }

  //-----------------------------------------------------------------------
  LOG_IF(INFO, debug_info_) << " top: " << top[0]->asum_data();

  After();
}

template <typename Dtype>
void RepartitionLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  for (size_t i = 0; i < bottom.size(); i++) {
    if (propagate_down[i]) {
      caffe_gpu_set(bottom[i]->count(), static_cast<Dtype>(0),
                    bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RepartitionLayer);

}  // namespace caffe
