pub mod align_edge;
pub mod base;
pub mod utils;
pub mod fast_mcd;
pub mod draw;
pub mod foreign;
pub mod deviation;

use std::collections::HashMap;
use std::ops::BitOr;
use std::sync::Mutex;

use base::Mat2Array_2;
use ndarray::ArrayBase;
use ndarray::ArrayViewMut;
use ndarray::Axis;
use ndarray::Dim;
use ndarray::OwnedRepr;
use ndarray::s;
use toml::*;
use toml::map::Map;
use align_edge::ret;
use time::*;

use linfa::DatasetBase;
use linfa::prelude::*;
use linfa_clustering::GaussianMixtureModel;

use rand_xoshiro::Xoshiro256Plus;


#[derive(Debug)]
pub struct ctx{
    pub C: Map<String, Value>,
    pub name: String,
    pub img_name: String,
    pub tpl_dir: String,
    pub board: String,
    pub cam_id: String,

    pub save_crop: String,
    pub save_filter: String,
    pub save_resize_align: String,
    pub save_transform: String,
    pub save_aligned_images: String,
    pub save_seg_gmm: String,
    pub save_seg_ood: String,
    pub save_foreigns: String,
    pub save_deviations: String,

    pub verify_transform: String,
    pub verify_seg_gmm: String,
    pub verify_seg_ood: String,
    pub verify_foreigns: String,
    pub verify_deviations: String,

    pub zoom: i32,
}

use opencv::imgcodecs::imwrite;
use opencv::prelude::*;
use opencv::core::*;
use opencv::imgproc::*;
use opencv::prelude::*;

pub fn process_crop(img:&Mat,tpl_cx:f64,tpl_cy:f64,tpl_h:i32,tpl_w:i32,scale:f64)->Mat{
    let mut grey_img = Mat::default();
    let code = ColorConversionCodes::COLOR_BGR2GRAY as i32;
    cvt_color(img, &mut grey_img, code, 0).expect("convert img to gray error");

    let scalar=Scalar::new(0.0,0.0,0.0,0.0);
    let mut resize_image = Mat::default();
    if scale < 1.0 {
        resize(&grey_img,&mut resize_image,
            Size_ { width: (img.cols() as f64*scale) as i32, height: (img.rows() as f64*scale) as i32 },
            0.0,0.0,3).unwrap();
        let mut edges = Mat::default();
        let threshold1 =100.0;
        let threshold2 =200.0;
        let aperture_size = 3;
        canny(&resize_image,&mut edges,threshold1,threshold2,aperture_size,false).expect("get edges error");

        let mut non_zero_points = Vector::<Point_<f64>>::new();
        find_non_zero(&edges, &mut non_zero_points).unwrap();

        let mut p_x:f64 = 0.0;
        let mut p_y:f64 = 0.0;
        for i in non_zero_points.iter() {
            p_x += i.x;
            p_y += i.y;
        }
        let cx = (p_x/non_zero_points.len() as f64)/scale;
        let cy = (p_y/non_zero_points.len() as f64)/scale;

        let mut dx = cx - tpl_cx;
        let mut dy = cy - tpl_cy;
        if dx < 0.0{
            dx=0.0;
        }
        if dy < 0.0{
            dy=0.0;
        }
        let mut width = tpl_w;
        let mut height = tpl_h;
        if width + dx as i32>= img.cols(){
            width = img.cols() - dx as i32;
        }
        if height +dy as i32 >= img.rows(){
            height = img.rows() -dy as i32;
        }
        let rect = Rect_{x:dx as i32,y:dy as i32,width:width,height:height};
    
        let crop_img = Mat::roi(img,rect).unwrap();
        return crop_img;
    }
    let mut edges = Mat::default();
    let threshold1 =100.0;
    let threshold2 =200.0;
    let aperture_size = 3;
    canny(&grey_img,&mut edges,threshold1,threshold2,aperture_size,false).expect("get edges error");

    let mut non_zero_points = Vector::<Point_<f64>>::new();
    find_non_zero(&edges, &mut non_zero_points).unwrap();

    let mut p_x:f64 = 0.0;
    let mut p_y:f64 = 0.0;
    for i in non_zero_points.iter() {
        p_x += i.x;
        p_y += i.y;
    }
    let cx = (p_x/non_zero_points.len() as f64)/scale;
    let cy = (p_y/non_zero_points.len() as f64)/scale;

    let mut dx = cx - tpl_cx;
    let mut dy = cy - tpl_cy;
    if dx < 0.0{
        println!("1111");
        dx=0.0;
    }
    if dy < 0.0{
        dy=0.0;
    }
    let mut width = tpl_w;
    let mut height = tpl_h;
    if width >= img.cols(){
        width = img.cols();
    }
    if height >= img.rows(){
        height = img.rows();
    }
    let rect = Rect_{x:dx as i32,y:dy as i32,width:width,height:height};

    let crop_img = Mat::roi(img,rect).unwrap();
    crop_img

}

pub fn process_filter(img:&Mat,d:i64,sigma_color:i64,sigma_space:i64)->Mat{
    let mut filter_img = Mat::default();
    bilateral_filter(img,&mut filter_img,d as i32,
        sigma_color as f64,sigma_space as f64,BORDER_DEFAULT as i32).expect("filter failed");
    filter_img
}

pub fn process_align_camera(img:&Mat,tpl_distmap:&Mat,init_bbox:[i64; 4],params:align_edge::align_params)->ret{
    let gray_img = base::get_gray_image(img);
    let tpl_w = tpl_distmap.cols();

    let x0 = init_bbox[0];
    let y0 = init_bbox[1];
    let x1 = init_bbox[2];
    let y1 = init_bbox[3];

    //是否越界？ init_box 改成f32
    let roi = Rect::new(x0 as i32,y0 as i32,(x1-x0) as i32,(y1-y0) as i32);
    let mut roi_img = Mat::default();
    if gray_img.empty() {
        roi_img = Mat::roi(img, roi).expect("use img roi error");
    }
    else{
        roi_img = Mat::roi(&gray_img, roi).expect("use grey img error");
    }
    
    let ret = align_edge::align_edge(roi_img,&tpl_distmap,Some(init_bbox),Mat::default(),None,
    params.tform_tp,false,params.optim,params.lr,1,1,params.max_iters,
    params.max_patience,0.9,10,200,0.001,Option::None,
    params.msg_prefix,"cuda:0".to_string(),true);
   
    ret
}

pub fn get_gmm_img(img:&Mat,feature:&Vec<Value>) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>{
    //考虑labget_gmm_img
    let mut lab_img = Mat::default();
    for i in feature.iter() {
        if i.as_str() == Some("lab.*") {
            cvt_color(&img, &mut lab_img, ColorConversionCodes::COLOR_BGR2Lab as i32, 0).expect("convert img to color error");
        }
        else{
            panic!("feature error:don't have lab");
        }
    }

    let array = base::Mat2Array_3(lab_img);
    let img_array = array.mapv(|x|x as f64);
    
    img_array
}
use ndarray::{Array2, Zip};
use rand::prelude::*;

fn sample_points_from_mask(mask: &Array2<bool>, nr: Option<usize>, seed: Option<u64>) -> Vec<(usize, usize)> {
    let mut rng = match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let (ys, xs) = mask.indexed_iter().filter(|(_, &v)| v).map(|((y, x), _)| (y, x)).unzip::<_, _, Vec<_>, Vec<_>>();
    let mut idxs: Vec<_> = (0..ys.len()).collect();
    idxs.shuffle(&mut rng);

    let idxs = match nr {
        Some(nr) => &idxs[..nr],
        None => &idxs,
    };

    let xs:Vec<usize> = idxs.iter().map(|&i| xs[i]).collect();
    let ys:Vec<usize> = idxs.iter().map(|&i| ys[i]).collect();
    xs.into_iter().zip(ys.into_iter()).collect()
}
pub fn process_gmm_seg(img:&ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,classes:&Map<String, Value>,
    mut blank_mask:ArrayBase<OwnedRepr<bool>, Dim<[usize; 2]>>,sample_nr:i32,chull_scale:f64,chull_erosion:i32,
    random_seed:i32,ys_init:usize) ->  ArrayBase<OwnedRepr<u8>, Dim<[usize; 2]>>{

        let imgh = img.shape()[0];
        let imgw = img.shape()[1];

        let b:ArrayBase<OwnedRepr<bool>, Dim<[usize; 2]>> = blank_mask.slice_mut(s![..,..]).mapv(|x| !x).to_owned();
        let mut tmp_coords = sample_points_from_mask(&b,Some(sample_nr as usize),Some(random_seed as u64));
        
        let samples = Array2::from_shape_fn((tmp_coords.len(),3), 
            |(i,j)|{
                let (x,y) = tmp_coords[i];
                img[(y,x,j)]
            });
        
        
        let _img = img.to_shape((imgh*imgw,3)).unwrap().to_owned();

        let dataset = DatasetBase::from(samples);
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let n_clusters = 2;
        let gmm = GaussianMixtureModel::params(n_clusters)
                    .n_runs(10)
                    .tolerance(1e-3)
                    .with_rng(rng)
                    .fit(&dataset).expect("GMM fitting error!!!");     
        println!(" train finished ");
    
        let predict_data =gmm.predict(_img);
        println!("predicted finished");
        let label = predict_data.targets().to_shape((imgh,imgw)).unwrap();

        println!(" predict data label is {:?}",label);
        println!(" predict finished ");
        println!("GMM means = {:?}", gmm.means());
        println!("GMM covariances = {:?}", gmm.covariances());
        
        let mut copper_label:Option<usize> = None;
        let mut bg_label:Option<usize> = None;
        for i in 0..n_clusters {
            let mask = label.mapv(|x|x==i);
            let (ys, xs) = mask.indexed_iter()
                .filter(|(_, &v)| v).map(|((y, x), _)| (y, x))
                .unzip::<_, _, Vec<_>, Vec<_>>();
            let ys_min = *ys.iter().min().unwrap();
            if ys_min < ys_init{
                bg_label = Some(i);
            }
            else{
                copper_label = Some(i);
            }
        }

        let mut segmap: ArrayBase<OwnedRepr<u8>, Dim<[usize; 2]>> = Array2::zeros((imgh,imgw));
        let copper_mask = label.mapv(|x|Some(x)==copper_label);
        let bg_mask = label.mapv(|x|Some(x)==bg_label);

        segmap.assign(&copper_mask.mapv(|x| 
            if x {
                classes.get("copper").unwrap().get("label").unwrap().as_integer().unwrap() as u8
            }
            else{
                classes.get("bg").unwrap().get("label").unwrap().as_integer().unwrap() as u8
            }
        ));

        let copper_mask_u8 = copper_mask.mapv(|x| if x{255 as u8} else {0 as u8});
        let mat = base::Array2Mat(&copper_mask_u8);
        let vaild_mask_tmp = get_valid_chull_mask(&mat,chull_scale,chull_erosion);
        
        let params:Vector<i32> = Vector::new();
        imwrite("vaild_mask.jpg", &vaild_mask_tmp, &params).unwrap();

        let vaild_mask = base::Mat2Array_2(vaild_mask_tmp).mapv(|x|if x==255 {true} else {false});
        
        let or_result = blank_mask.bitor(!vaild_mask);

        let mut segmap_view = segmap.view_mut();
        let or_result_view = or_result.view();
        for ((i,j),value) in or_result_view.indexed_iter() {
            if *value {
                segmap_view[[i,j]] = 0;
            }
        }
        let segmap = segmap_view.to_owned();
        segmap
}
pub fn process_gmm_seg_2(img_:&Mat,img:&ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,classes:&Map<String, Value>,
    mut blank_mask:ArrayBase<OwnedRepr<bool>, Dim<[usize; 2]>>,sample_nr:i32,chull_scale:f64,chull_erosion:i32,
    random_seed:i32,ys_init:usize) ->  ArrayBase<OwnedRepr<u8>, Dim<[usize; 2]>>{

        let imgh = img.shape()[0];
        let imgw = img.shape()[1];

        let start_time = time::now();
        let grey_img = base::get_gray_image(&img_);
        
        let mut adaptive_img = Mat::default();
        opencv::imgproc::threshold(&grey_img, &mut adaptive_img, 0.0, 255.0 ,THRESH_OTSU).unwrap();
   
        let label = Mat2Array_2(adaptive_img).mapv(|x|if x==255 {1} else {0});
    
        let mut copper_label:Option<u8> = None;
        let mut bg_label:Option<u8> = None;
        let n_clusters = 2;
        for i in 0..n_clusters {
            let mask = label.mapv(|x|x==i);
            let (ys, xs) = mask.indexed_iter()
                .filter(|(_, &v)| v).map(|((y, x), _)| (y, x))
                .unzip::<_, _, Vec<_>, Vec<_>>();
            let ys_min = *ys.iter().min().unwrap();
            if ys_min < ys_init{
                bg_label = Some(i);
            }
            else{
                copper_label = Some(i);
            }
        }
        // println!("bg_label = {:?}", bg_label);
        // println!("copper_label = {:?}",copper_label);
        let mut segmap: ArrayBase<OwnedRepr<u8>, Dim<[usize; 2]>> = Array2::zeros((imgh,imgw));
        let copper_mask = label.mapv(|x|Some(x)==copper_label);

        segmap.assign(&copper_mask.mapv(|x| 
            if x {
                classes.get("copper").unwrap().get("label").unwrap().as_integer().unwrap() as u8
            }
            else{
                classes.get("bg").unwrap().get("label").unwrap().as_integer().unwrap() as u8
            }
        ));

        
        let copper_mask_u8 = copper_mask.mapv(|x| if x{255 as u8} else {0 as u8});
        let mat = base::Array2Mat(&copper_mask_u8);
        let vaild_mask_tmp = get_valid_chull_mask(&mat,chull_scale,chull_erosion);
        
        
        let vaild_mask = base::Mat2Array_2(vaild_mask_tmp).mapv(|x|if x==255 {true} else {false});
        
        let or_result = blank_mask.bitor(!vaild_mask);
        //println!("a is {:?}",or_result);

        let mut segmap_view = segmap.view_mut();
        let or_result_view = or_result.view();
        for ((i,j),value) in or_result_view.indexed_iter() {
            if *value {
                segmap_view[[i,j]] = 0;
            }
        }
        let segmap = segmap_view.to_owned();
        let end_time = time::now();
        println!("process gmm time over is {:?}",end_time - start_time);
      
        segmap
}
use opencv::{
    core::{Scalar, Vec2i},
    imgproc::{
        adaptive_threshold, contour_area, convex_hull, draw_contours, fill_poly, find_contours,
        morphology_default_border_value, morphology_ex, MORPH_ELLIPSE, RETR_EXTERNAL,
        THRESH_BINARY,
    },
    prelude::*,
    types::{VectorOfPoint, VectorOfVec4i,VectorOfMat},
};

fn get_valid_chull_mask(mask: &Mat, scale: f64, erosion_radius: i32) -> Mat{

    let mut resize_mask = base::rescale(mask, scale, INTER_NEAREST);
    println!("resize mask is {:#?}", resize_mask);
    let mut color_mask = Mat::default();
    cvt_color(&resize_mask, &mut color_mask, ColorConversionCodes::COLOR_GRAY2BGR as i32, 0).unwrap();

    // Find contours
    let mut contours:Vector<Vector<Point>> = Vector::new();
    find_contours(&resize_mask, &mut contours, 
        RETR_TREE, CHAIN_APPROX_SIMPLE,Point::default()).unwrap();

    
    let mut tmp = VectorOfPoint::new();
    for i in contours.iter() {
        for j in i.iter() {
            tmp.push(j);
        }
    }

    println!("find contours is shape {:#?}", contours.len());
    let mut vec = Vector::<VectorOfPoint>::new();
    let mut chull_mask =  Mat::new_rows_cols_with_default(resize_mask.rows(), resize_mask.cols(), opencv::core::CV_8UC1,Scalar::all(0.0)).unwrap();
    let mut hull_points =  VectorOfPoint::new();
    convex_hull(&tmp, &mut hull_points, false, true).unwrap();
    fill_convex_poly(&mut chull_mask, &hull_points, Scalar::all(255.0), LINE_8, 0).unwrap();
    
    let erosion_kernel = get_structuring_element(MORPH_ELLIPSE,Size::new(erosion_radius as i32 * 2 + 1, erosion_radius as i32 * 2 + 1), Point::default()).unwrap();
    let mut chull_mask_ = Mat::default();
    erode(&chull_mask, &mut chull_mask_, &erosion_kernel,Point_ { x: -1, y: -1 },1,BORDER_CONSTANT,morphology_default_border_value().unwrap()).unwrap();

    // Resize mask to original size
    let mut chull_mask_rescale = Mat::default();
    let h = mask.rows() as f64;
    let w = mask.cols() as f64;

    resize(&chull_mask_,&mut chull_mask_rescale,
        Size_ { width: w as i32, height: h as i32 },
        0.0,0.0,INTER_NEAREST).unwrap();
    chull_mask_rescale
}

use base::disk;
pub fn process_ood_seg(img:&Mat,mask:&Mat,cfg:&Value,shadow_mask:Option<&Mat>,zoom:i32)
 ->(ArrayBase<OwnedRepr<bool>, Dim<[usize; 2]>>,ArrayBase<OwnedRepr<bool>, Dim<[usize; 2]>>){

    let shrink = utils::get_zoomed_len(cfg.get("segmap_shrink").unwrap(), zoom);
    let edge_region_radius = utils::get_zoomed_len(cfg.get("edge_region_radius").unwrap(), zoom);
    let random_seed = utils::get_zoomed_len(cfg.get("random_seed").unwrap(), zoom);
    let sample_nr_all = utils::get_zoomed_len(cfg.get("sample_nr").unwrap().get("all").unwrap(), zoom);
    let sample_nr_edge = utils::get_zoomed_len(cfg.get("sample_nr").unwrap().get("edge").unwrap(), zoom);
    let mut sample_nr_shadow = 0 as i64;
    
    let mut mask_erode = Mat::default();
    let kernel=disk(shrink as i32);
    erode(&mask, &mut mask_erode, &kernel, Point_{x:-1,y:-1}, 1, BORDER_CONSTANT, morphology_default_border_value().unwrap()).expect("erode failed");

    let mut edge_erode = Mat::default();
    let kernel=disk(edge_region_radius as i32);
    erode(&mask_erode, &mut edge_erode, &kernel, Point_{x:-1,y:-1}, 1, BORDER_CONSTANT, morphology_default_border_value().unwrap()).expect("erode failed");

    let mut edge_mask = Mat::default();
    bitwise_xor(&mask_erode, &edge_erode, &mut edge_mask, &no_array()).unwrap();

    let mut sample_masks = Vec::new();
    if let Some(mat) = shadow_mask {
        let mut shadow_mask = Mat::default();
        bitwise_and(&mask_erode, mat, &mut shadow_mask, &no_array()).unwrap();
        let array = Mat2Array_2(shadow_mask).mapv(|x|if x==255{true} else {false});
        sample_masks.push(array);
        sample_nr_shadow = utils::get_zoomed_len(cfg.get("sample_nr").unwrap().get("shadow").unwrap(), zoom);
    }
    let mask_erode = Mat2Array_2(mask_erode).mapv(|x|if x==255{true} else {false});
    let edge_mask = Mat2Array_2(edge_mask).mapv(|x|if x==255{true} else {false});
    sample_masks.push(mask_erode);
    sample_masks.push(edge_mask);
    

    let feat = cfg.get("feature").unwrap().as_array().unwrap();
    let dist_th = cfg.get("dist_th").unwrap().as_integer().unwrap();
    let sample_nr = [sample_nr_shadow,sample_nr_all,sample_nr_edge];
    let support_frac = cfg.get("support_frac").unwrap().as_float().unwrap();

    let mut coords = Vec::new();
    for i in 0..sample_masks.len(){
        let mut _coords = sample_points_from_mask(&sample_masks[i], Some(sample_nr[1] as usize), Some(random_seed as u64));
        coords.append(&mut _coords);
    }

    let img_array = get_gmm_img(img,feat);
    let train_xs = Array2::from_shape_fn((coords.len(),3), //train_xs
    |(i,j)|{
        let (x,y) = coords[i];
        img_array[(y,x,j)]
    });

    use fast_mcd::*;
    use base::empirical_covariance;
    let p_fn = empirical_covariance as *const ();
    let cov_computation_method:fn(&Array2<f64>) -> Array2<f64>  = unsafe{std::mem::transmute(p_fn)};
    let start_time = time::now();
    let (location, covariance, support, dist) = 
        fast_mcd::fast_mcd(train_xs, Some(support_frac), cov_computation_method, None).unwrap();
    let end_time = time::now();

    let rows = img_array.shape()[0] * img_array.shape()[1];
    let test_xs = img_array.to_shape((rows,3)).unwrap().to_owned();
    let (n_samples, n_features) = test_xs.dim();
    let precision = pinvh(&covariance,1e-12);
    let x_centered = test_xs - &location.broadcast((n_samples, n_features)).unwrap();
    let dists = (&x_centered.dot(&precision) * &x_centered).sum_axis(Axis(1));
    
    let dists_ = dists.to_shape((img_array.shape()[0] , img_array.shape()[1])).unwrap().to_owned();

    let mut lbl = dists_.mapv(|x|if x < dist_th as f64 {true} else {false});

    if let None = shadow_mask{
        for ((i,j),value) in sample_masks[0].indexed_iter(){//copper/bg mask
            if !value {
                lbl[(i,j)] = false;
            }
            else{
                lbl[(i,j)] = lbl[(i,j)];
            }
        }        
        (lbl,sample_masks[0].clone())
    }
    else{
        for ((i,j),value) in sample_masks[1].indexed_iter(){
            if !value {
                lbl[(i,j)] = false;
            }
            else{
                lbl[(i,j)] = lbl[(i,j)];
            }
        }
        (lbl,sample_masks[1].clone())
    }
    

}