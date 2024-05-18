use deeppcb::align_edge;
use deeppcb::base;
use deeppcb::deviation;
use deeppcb::draw;
use deeppcb::foreign;
use deeppcb::utils;

use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use std::sync::Mutex;
use crossbeam_channel::{unbounded, Receiver, Sender};
use draw::{verify_align_camera, transform_img};
use json::{object, array};
use rayon::prelude::ParallelIterator;
use rayon::{self, ThreadPool};
use serde_derive::{Serialize, Deserialize};
use std::fs::File;
use std::io::{self};
use std::path::{Path, PathBuf};
use opencv::imgcodecs::{imread, imwrite, IMWRITE_JPEG_QUALITY};
use std::fmt::format;
use std::{fs, thread};
use base::*;
use base::{find_file,imsave};
use deeppcb::*;
use align_edge::*;
use foreign::*;
use deviation::detect_deviations;

use std::io::{Read, Write}; 
use toml::{Table, Value};
use toml::map::Map;
use std::env;
use opencv::prelude::*;
use opencv::core::*;
use opencv::imgproc::{resize, ColorConversionCodes, cvt_color, COLOR_BGR2RGB};
use rayon::iter::IntoParallelRefIterator;
use ndarray::{ArrayViewMut,s, ArrayBase,OwnedRepr,Dim};

// use rusty_pool::Builder;
// use rusty_pool::ThreadPool as Threadpool;

const N_WORKERS: usize = 4;
#[macro_use]
extern crate lazy_static;
lazy_static!{
    static ref thread_pool:ThreadPool = rayon::ThreadPoolBuilder::new().
        num_threads(N_WORKERS).build().unwrap();

    // static ref thread_pool_:Threadpool = Threadpool::default();
        
    static ref filter_channel:FILTER_CHANNEL = {
        let (tx,rx) = unbounded::<Mat>();
        let filter_struct = FILTER_CHANNEL{tx:tx,rx:rx};
        filter_struct
    };

    static ref tform_channel:TFORM_CHANNEL = {
        let (tx,rx) = unbounded::<ret>();
        let tfrom_struct = TFORM_CHANNEL{tx:tx,rx:rx};
        tfrom_struct
    };
    static ref ALIGN_IMAGE:Mutex<Mat> = Mutex::new(Mat::default());
    //static ref align_img:ALIGN_IMG = ALIGN_IMG{img:Mat::default()};
}


struct FILTER_CHANNEL{
    tx:Sender<Mat>,
    rx:Receiver<Mat>,
}
struct TFORM_CHANNEL{
    tx:Sender<ret>,
    rx:Receiver<ret>,
}

fn run_stage(ctx:&ctx,img:&mut Mat){

    let (crop_tx_ctx,crop_rx_ctx) = unbounded();
    let (crop_tx_mat,crop_rx_mat) = unbounded();
    crop_tx_ctx.send(ctx).unwrap();
    crop_tx_mat.send(img).unwrap();
    thread_pool.install(move ||{
        run_stage_crop(crop_rx_ctx.recv().unwrap(), crop_rx_mat.recv().unwrap());
    })
}
fn run_next_stage_1(ctx:&ctx,img:&mut Mat,function:fn(&ctx,&mut Mat)){
    let (next_tx_ctx,next_rx_ctx) = unbounded();
    let (next_tx_mat,next_rx_mat) = unbounded();
    next_tx_ctx.send(ctx).unwrap();
    next_tx_mat.send(img).unwrap();
    thread_pool.install(||{
        function(next_rx_ctx.recv().unwrap(), next_rx_mat.recv().unwrap());
    })
}
fn run_next_stage_(ctx:&ctx,function:fn(&ctx)){
    let (next_tx_ctx,next_rx_ctx) = unbounded();
    next_tx_ctx.send(ctx).unwrap();
  
    thread_pool.install(||{
        function(next_rx_ctx.recv().unwrap());
    })
}
fn run_next_stage_2(ctx:&ctx,img:&mut Mat,function:fn(&ctx,&mut Mat,&mut Mat),img2:&mut Mat){
    let (next_tx_ctx,next_rx_ctx) = unbounded();
    let (next_tx_mat,next_rx_mat) = unbounded();
    let (next_tx_mat2,next_rx_mat2) = (next_tx_mat.clone(),next_rx_mat.clone());
    next_tx_ctx.send(ctx).unwrap();
    next_tx_mat.send(img).unwrap();
    next_tx_mat2.send(img2).unwrap();
    thread_pool.install(||{
        function(next_rx_ctx.recv().unwrap(), next_rx_mat.recv().unwrap(), next_rx_mat2.recv().unwrap());
    })
}
fn run_stage_crop(ctx:&ctx,img:&mut Mat){
    let start_time = time::now();
    println!("Running stage crop ");
    let stage = "crop".to_string();
    let name = &ctx.name;
    let C = &ctx.C;

    let save_dir =ctx.save_crop.clone();
    let save_path = format!("{}/{}",ctx.save_crop,ctx.img_name);

    let mut path = PathBuf::new();
    path.push(".");
    if base::find_file(&path, &save_dir){
        
        path.push(&save_dir);
        if base::find_file(&path, &ctx.img_name.clone()){
            let mut crop_img = imread(&save_path,opencv::imgcodecs::IMREAD_COLOR).unwrap();
            let p_fn = run_stage_filter as *const ();
            let run_stage_filter:fn(&ctx,&mut Mat) = unsafe{std::mem::transmute(p_fn)};
            run_next_stage_1(ctx,&mut crop_img,run_stage_filter);
            drop(img);
            return;
        }
    }
    
    let tpl_edge_path = format!("{}/target_{}x/edges/{}.png",ctx.tpl_dir,ctx.zoom,ctx.cam_id);
    let tpl_edge = imread(&tpl_edge_path,opencv::imgcodecs::IMREAD_ANYCOLOR).unwrap();

    let mut non_zero_points = Vector::<Point_<f64>>::new();
    find_non_zero(&tpl_edge, &mut non_zero_points).unwrap();

    let mut p_x:f64 = 0.0;
    let mut p_y:f64 = 0.0;
    for i in non_zero_points.iter() {
        p_x += i.x;
        p_y += i.y;
    }
    let h = tpl_edge.rows();
    let w = tpl_edge.cols();
    let tpl_cx = p_x/non_zero_points.len() as f64;
    let tpl_cy =p_y/non_zero_points.len() as f64;

    let mut crop_image = deeppcb::process_crop(img, tpl_cx, tpl_cy, h, w, 0.5);

    if save_path.is_empty() == false{
        println!("save...");
        if base::find_file(&path, &save_dir)==false{
            fs::create_dir_all(save_dir).unwrap();
        }
        imsave(&save_path, &crop_image,100);
    }
    let end_time = time::now();
    println!("crop time over {:?}",end_time-start_time);
    
    let p_fn = run_stage_filter as *const ();
    let run_stage_filter:fn(&ctx,&mut Mat) = unsafe{std::mem::transmute(p_fn)};
    run_next_stage_1(ctx,&mut crop_image,run_stage_filter);
    drop(img);

}
fn run_stage_filter(ctx:&ctx,img:&mut Mat){
    println!("Running filter  ");
    let start_time = time::now();
    let stage = "filter".to_string();
    let name = &ctx.name;
    let C = &ctx.C;

    let save_dir =ctx.save_filter.clone();
    let save_path = format!("{}/{}",save_dir,ctx.img_name);
    println!("save_path is {}",save_path);
    println!("save_dir is {}",save_dir);
    let mut path = PathBuf::new();
    path.push(".");
    if find_file(&path, &save_dir){
        path.push(&save_dir);
        if find_file(&path, &ctx.img_name.clone()){
            let mut filter_img = imread(&save_path,opencv::imgcodecs::IMREAD_COLOR).unwrap();
            let p_fn = run_stage_resize_align as *const ();
            let run_stage_resize_align:fn(&ctx,&mut Mat) = unsafe{std::mem::transmute(p_fn)};
            filter_channel.tx.send(filter_img.clone()).unwrap();
            run_next_stage_1(ctx,&mut filter_img,run_stage_resize_align);
            
            drop(img);
            return;
        }
    }
    let cfg = C.get("target").unwrap().get("filter").unwrap();
    let d =utils::get_zoomed_len(cfg.get("d").unwrap(),ctx.zoom);
    let sigma_space =utils::get_zoomed_len(cfg.get("d").unwrap(),ctx.zoom);
    let sigma_color = cfg.get("sigma_color").unwrap().as_integer().unwrap();

    
    let mut filter_img =deeppcb::process_filter(img, d, sigma_color, sigma_space);

    if save_path.is_empty() == false{
        println!("save...");
        if find_file(&path, &save_dir)==false{
            fs::create_dir_all(save_dir).unwrap();
        }
        imsave(&save_path, &filter_img,100);
    }

    let end_time = time::now();
    println!("filter time over {:?}",end_time-start_time);
    
    let p_fn = run_stage_resize_align as *const ();
    let run_stage_resize_align:fn(&ctx,&mut Mat) = unsafe{std::mem::transmute(p_fn)};
    filter_channel.tx.send(filter_img.clone()).unwrap();
    run_next_stage_1(ctx,&mut filter_img,run_stage_resize_align);
    
    drop(img);


}
fn run_stage_resize_align(ctx:& ctx,img:&mut Mat){
    println!("Running run_stage_resize_align ");
    let start_time = time::now();
    let C = &ctx.C;
    //println!("name : {}",name);
    //println!("C : {:#?}",C);
    let save_dir =ctx.save_resize_align.clone();
    let save_path = format!("{}/{}",save_dir,ctx.img_name);

    let mut path = PathBuf::new();
    path.push(".");
    if find_file(&path, &save_dir){
        path.push(&save_dir);
        if find_file(&path, &ctx.img_name.clone()){
            let mut img = imread(&save_path,opencv::imgcodecs::IMREAD_COLOR).unwrap();
            let p_fn = run_stage_estimate_camera_align as *const ();
            let run_stage_estimate_camera_align:fn(&ctx,&mut Mat) = unsafe{std::mem::transmute(p_fn)};
            run_next_stage_1(ctx,&mut img,run_stage_estimate_camera_align);
            drop(img);
            return;
        }
    }
    let mut radio:f64 = 0.0;
    if ctx.zoom == 1{
        radio = C.get("base").unwrap().get("align_scale").unwrap().as_float().unwrap();
    }
    else{
        radio = ctx.zoom as f64;
    }
    
    let scalar=Scalar::new(0.0,0.0,0.0,0.0);
    let mut resize_image = Mat::default();
    if radio < 1.0{
        resize(img,&mut resize_image,
            Size_ { width: (img.cols() as f64*radio) as i32, height: (img.rows() as f64*radio) as i32 },
            0.0,0.0,3).unwrap();

        if save_path.is_empty() == false{
            println!("save...");
            if find_file(&path, &save_dir)==false{
                fs::create_dir_all(save_dir).unwrap();
            }
            imsave(&save_path, &resize_image,100);
        }
        let p_fn = run_stage_estimate_camera_align as *const ();
        let run_stage_estimate_camera_align:fn(&ctx,&mut Mat) = unsafe{std::mem::transmute(p_fn)};
        run_next_stage_1(ctx,&mut resize_image,run_stage_estimate_camera_align);
        drop(img);
    }
    else{
        if save_path.is_empty() == false{
            println!("save...");
            if find_file(&path, &save_dir)==false{
                fs::create_dir_all(save_dir).unwrap();
            }
            imsave(&save_path, img,100);
        }
        //没有进行resize
        //ctx.resize_align__img = ctx.filter__img.clone();
        let p_fn = run_stage_estimate_camera_align as *const ();
        let run_stage_estimate_camera_align:fn(&ctx,&mut Mat) = unsafe{std::mem::transmute(p_fn)};
        run_next_stage_1(ctx,img,run_stage_estimate_camera_align);
        
    }
    
    let end_time = time::now();
    println!("resize align time over {:?}",end_time-start_time);
    
}
fn run_stage_estimate_camera_align(ctx:& ctx,img:&mut Mat){
    println!("Running estimate_camera_align ");       

    let start_time = time::now();
    let stage = "estimate_camera_align".to_string();
    let name = &ctx.name;
    let C = &ctx.C;
    println!("name : {}",name);
    //println!("C : {:#?}",C);
    let save_dir  =ctx.save_transform.clone();
    let save_path = format!("{}/{}.json",save_dir,name);
    println!("save_path is {}",save_path);
    println!("save_dir is ...........................................................{}",save_dir);
    let mut path = PathBuf::new();
    path.push(".");
    if find_file(&path, &save_dir){
        path.push(&save_dir);
        //save json not img
        if find_file(&path, &format!("{}.json",name)){
            let file = std::fs::File::open(save_path).unwrap();
            
            #[derive(Debug, Serialize, Deserialize)]
            struct PyParams{
                tform_tp:String,
                sim_no_scale:bool,
                lr:f64,
                max_iters:i32,
                max_patience:i32,
                max_dist:i32,
                H_scale:[i32;2],
            }
            #[derive(Debug, Serialize, Deserialize)]
            struct Value{
                H_20:Vec<Vec<f32>>,
                H_21:Vec<Vec<f32>>,
                error:f32,
                params:PyParams,
                tgt_shape:[i32;2],
                tpl_shape:[i32;2],
            }
            let _ret_tmp: Value = serde_json::from_reader(file).unwrap();
            println!("This is align_region :{:#?}",_ret_tmp);
            let params = py_params{
                tform_tp:_ret_tmp.params.tform_tp,
                sim_no_scale:_ret_tmp.params.sim_no_scale,
                lr:_ret_tmp.params.lr ,
                max_iters:_ret_tmp.params.max_iters,
                max_patience:_ret_tmp.params.max_patience,
                max_dist:_ret_tmp.params.max_dist,
                H_scale:_ret_tmp.params.H_scale,

            };
            let ret = ret{
                H_20:_ret_tmp.H_20,
                H_21:_ret_tmp.H_21,
                err:_ret_tmp.error,
                tpl_h:_ret_tmp.tpl_shape[0],
                tpl_w:_ret_tmp.tpl_shape[1],
                params:params,
            };
            tform_channel.tx.send(ret).unwrap();
            let p_fn = run_stage_align_camera as *const ();
            let run_stage_align_camera:fn(&ctx) = unsafe{std::mem::transmute(p_fn)};
            run_next_stage_(ctx,run_stage_align_camera);
            return;
        }
    }
    let cfg = C.get("target").unwrap().get("align_camera").unwrap();
    let padding = cfg.get("padding").unwrap().as_integer().unwrap();
    let tform_tp = cfg.get("tform_tp").unwrap().as_str().unwrap().to_string();
    let msg_prefix = format!("{}:",name);
    println!{"tform_tp is {}",tform_tp};


    //read_jason
    let tpl_dir = format!("./{}/target_2x",ctx.tpl_dir);
    let align_json_dir = format!("{}/align_region.json",tpl_dir);
    println!("align_json_dir is {}",align_json_dir);
    let file = std::fs::File::open(align_json_dir).unwrap();
    let align_region_tmp: serde_json::Value = serde_json::from_reader(file).unwrap();
    let align_region = align_region_tmp.get(ctx.cam_id.clone()).unwrap().get("align_bbox").unwrap().
    as_array().unwrap();
    println!("This is align_region :{:#?}",align_region);
    //read_dismap
    let dismap_dir = format!("{}/distmaps/{}.jpg",tpl_dir,ctx.cam_id.clone());
    println!("dismap_dir is :{}",dismap_dir);
    let tpl_distmap = imread(&dismap_dir,opencv::imgcodecs::IMREAD_GRAYSCALE).unwrap();

    let x0 = align_region[0].as_i64().unwrap() + padding;
    let y0 = align_region[1].as_i64().unwrap() + padding;
    let x1 = align_region[2].as_i64().unwrap() - padding;
    let y1 = align_region[3].as_i64().unwrap() - padding;
    let init_bbox = [x0,y0,x1,y1];
    println!("init_bbox is {:?}", init_bbox);

    //params
    let params = align_edge::align_params{
        tform_tp :tform_tp,
        msg_prefix : msg_prefix,
        optim : String::from("adam"),
        lr : 0.002,
        max_patience : 20,
        max_iters : 500,
    };
    let ret = deeppcb::process_align_camera(img, &tpl_distmap, init_bbox, params);
    let ret_ = ret.clone();
    tform_channel.tx.send(ret_).unwrap();
    let H_21:Vec<Vec<f32>> = ret.H_21;
    let H_20:Vec<Vec<f32>> = ret.H_20;
    if save_path.is_empty() == false {
        println!("save path {}",save_path);
        let tgt_shape = img.size().unwrap();
        let tpl_shape = tpl_distmap.size().unwrap();
        println!("tgt_shape is {:?}",tgt_shape);
        println!("tpl_shape is {:?}",tpl_shape);
    
        
        let error = ret.err;
        let mut params = json::JsonValue::new_object();
        
        params["tform_tp"] = ret.params.tform_tp.into();
        params["max_iters"] = ret.params.max_iters.into();
        params["sim_no_scale"] = ret.params.sim_no_scale.into();
        params["max_dist"] = ret.params.max_dist.into();
        params["lr"] = ret.params.lr.into();
        params["max_patience"] = ret.params.max_patience.into();
        let mut H_scale = json::JsonValue::new_array();
        H_scale.push(ret.params.H_scale[0]);
        H_scale.push(ret.params.H_scale[1]);
        params["H_scale"] = H_scale.into();
        println!("params: {:?}",params);
        let save_data = object!{
            "tgt_shape" => array![tgt_shape.height,tgt_shape.width],
            "tpl_shape" => array![tpl_shape.height,tpl_shape.width],
            "H_21" => H_21.clone(),
            "H_20" => H_20,
            "error" =>error,
            "tpl_h" =>tpl_shape.height,
            "tpl_w" =>tpl_shape.width,
            "params" => params
        };
        let response = save_data.dump();
        if find_file(&path, &save_dir)==false{
            fs::create_dir_all(save_dir).unwrap();
        }
        let mut file = File::create(save_path).unwrap();
        file.write_all(response.as_bytes()).unwrap();
    }

   
    let p_fn = run_stage_align_camera as *const ();
    let run_stage_align_camera:fn(&ctx) = unsafe{std::mem::transmute(p_fn)};
    run_next_stage_(ctx,run_stage_align_camera);



    let verify = &ctx.verify_transform;
    
    if verify.is_empty() ==false{
        if find_file(&path, verify) == false{
            fs::create_dir_all(verify).unwrap();
        }
    }
    let tpl_img_dir = format!("{}/images/{}.jpg",tpl_dir,ctx.cam_id.clone());
    //println!("tpl_img_dir is {}",tpl_img_dir);
    let tpl_img = imread(&tpl_img_dir,opencv::imgcodecs::IMREAD_ANYCOLOR).unwrap();   
    //println!("tpl_img is {:#?}",tpl_img);       

    // let tpl_img = Mat::roi(&tpl_img,Rect_{x:x0 as i32,y:y0 as i32,width:(x1-x0) as i32,height:(y1-y0) as i32}).unwrap();
    // let img = Mat::roi(&img,Rect_{x:x0 as i32,y:y0 as i32,width:(x1-x0) as i32,height:(y1-y0) as i32}).unwrap();
    
    let mut color_img = Mat::default();
    if tpl_img.dims() == 2{
        let code = ColorConversionCodes::COLOR_GRAY2BGR as i32;
        cvt_color(&tpl_img, &mut color_img, code, 0).expect("convert img to color error");
    }  
    
    let canves = verify_align_camera(H_21,&img,&color_img);
    println!("verify {}",verify);
    let params:Vector<i32> = Vector::new();
    let mut canves1 =  Mat::default();
    cvt_color(&canves.get(0).unwrap(),&mut canves1,COLOR_BGR2RGB,0).unwrap();
    let mut canves2 =  Mat::default();
    cvt_color(&canves.get(1).unwrap(),&mut canves2,COLOR_BGR2RGB,0).unwrap();

    imwrite(&format!("./{}/{}_init.jpg",verify,&ctx.name), &canves1, &params).unwrap();
    imwrite(&format!("./{}/{}_warp.jpg",verify,&ctx.name), &canves2, &params).unwrap();

    let end_time = time::now();
    println!("resize align time over {:?}",end_time-start_time);
    

}
fn run_stage_align_camera(ctx:&mut ctx){
    println!("Running align_camera");
    let start_time = time::now();

    let img = filter_channel.rx.recv().unwrap();
    let C = &ctx.C;

    let save_dir =ctx.save_aligned_images.clone();
    let save_path = format!("{}/{}",save_dir,ctx.img_name);
    println!("save_path is {}",save_path);
    println!("save_dir is {}",save_dir);
    let mut path = PathBuf::new();
    path.push(".");
    if find_file(&path, &save_dir){
        path.push(&save_dir);
        if find_file(&path, &ctx.img_name.clone()){
            let mut trans_img =imread(&save_path,opencv::imgcodecs::IMREAD_COLOR).unwrap();
            let p_fn = run_stage_gmm as *const ();
            let run_stage_gmm:fn(&ctx,&mut Mat) = unsafe{std::mem::transmute(p_fn)};
            run_next_stage_1(ctx,&mut trans_img,run_stage_gmm);
            //align_img.set_value_img(trans_img.clone());

            return;
        }
    }
    let mut s = 1.0;
    if ctx.zoom == 1{
        s = s / C.get("base").unwrap().get("align_scale").unwrap().as_float().unwrap();
    }

    let mut tform = tform_channel.rx.recv().unwrap();
    println!("tform is {:#?}",tform);

    let mut H_ =&mut tform.H_21; 
    println!("H_ is {:#?}",H_);
    scale_H(H_, s);
    println!("scale H_ is {:#?}",H_);
    let H = H_.to_vec();

    let mut tpl_h = tform.tpl_h;

    let mut tpl_w = tform.tpl_w;
    println!("tpl_w is {}",tpl_w);
    tpl_h = (tpl_h as f64 * s) as i32;
    tpl_w = (tpl_w as f64 * s) as i32;
    let size = Size::new(tpl_w, tpl_h);

    
    let mut trans_img = transform_img(&img,H,size,Option::None,Option::None,Option::None);
    
    if save_path.is_empty() == false{
        println!("align_camera save...");
        if find_file(&path, &save_dir)==false{
            fs::create_dir_all(save_dir).unwrap();
        }
        imsave(&save_path, &trans_img, 100);
    }

    let p_fn = run_stage_gmm as *const ();
    let run_stage_gmm:fn(&ctx,&mut Mat) = unsafe{std::mem::transmute(p_fn)};
    run_next_stage_1(ctx,&mut trans_img,run_stage_gmm);

    //let mut crcb_img = Mat::default();
    //let params:Vector<i32> = Vector::new();
    //cvt_color(&trans_img, &mut crcb_img, COLOR_BGR2YCrCb,0).unwrap();
    //imwrite("crcb.jpg", &crcb_img, &params).unwrap();

    //保存ycrcb图像
    
    let end_time = time::now();
    println!("filter time over {:?}",end_time-start_time);    
}
fn run_stage_gmm(ctx:& ctx,img:&mut Mat){
    println!("runing stage gmm");
    let C = &ctx.C;

    let save_dir =ctx.save_seg_gmm.clone();
    let save_path = format!("{}/{}",save_dir,ctx.img_name);
    println!("save_path is {}",save_path);
    // println!("save_dir is {}",save_dir);
    let mut path = PathBuf::new();
    path.push(".");
    if find_file(&path, &save_dir){
        path.push(&save_dir);
        if find_file(&path, &ctx.img_name.clone()){
            let mut segmap_mat =imread(&save_path,opencv::imgcodecs::IMREAD_GRAYSCALE).unwrap();//grey CV_8UC1
            let p_fn = run_stage_seg_ood as *const ();
            let run_stage_seg_ood:fn(&ctx,&mut Mat,&mut Mat) = unsafe{std::mem::transmute(p_fn)};
            run_next_stage_2(ctx,&mut segmap_mat,run_stage_seg_ood, img);
            return;
        }
    }

    let cfg = ctx.C.get("target").unwrap().get("gmm_seg").unwrap();
    let feature = cfg.get("feature").unwrap().as_array().unwrap();
    //println!("feature is {:?}",feature[0]);
    let img_array = get_gmm_img(img,feature);

    let sample_nr = utils::get_zoomed_len(cfg.get("sample_nr").unwrap(),ctx.zoom);
    let random_seed = utils::get_zoomed_len(cfg.get("random_seed").unwrap(),ctx.zoom);
    let ys_init = {
        let tmp = (10.0/(ctx.zoom as f64)).ceil();
        if tmp > ctx.zoom as f64{
            tmp
        }
        else{
            ctx.zoom.into()
        }
    };
    let classes = C.get("classes").unwrap().as_table().unwrap();
    let blank_mask:ArrayBase<OwnedRepr<bool>, Dim<[usize; 2]>> = img_array.slice(s![..,..,0]).to_owned().mapv(|x|x < 5.0);

    // let mut segmap = process_gmm_seg(&img_array, classes, blank_mask,sample_nr as i32,1.0/4.0,1,
    //     random_seed as i32,ys_init as usize);
    let mut segmap = process_gmm_seg_2(img,&img_array, classes, blank_mask,sample_nr as i32,1.0/4.0,1,
        random_seed as i32,ys_init as usize);

    println!("process_gmm over");
    let bw = cfg.get("blank_border_width").unwrap().as_integer().unwrap() as i32;
    if bw!=0 {
        segmap.slice_mut(s![..bw,..]).fill(0);
        segmap.slice_mut(s![-bw..,..]).fill(0);
        segmap.slice_mut(s![..,..bw]).fill(0);
        segmap.slice_mut(s![..,-bw..]).fill(0);
    }
    let mut segmap_mat = base::Array2Mat(&segmap);//segmap_mat CV_8UC1

    println!("segmap_mat shape is {} {}",segmap_mat.rows(),segmap_mat.cols());
    if save_path.is_empty() == false{
        if find_file(&path, &save_dir)==false{
            fs::create_dir_all(save_dir).unwrap();
        }
        imsave(&save_path, &segmap_mat,100);
    }

    let p_fn = run_stage_seg_ood as *const ();
    let run_stage_seg_ood:fn(&ctx,&mut Mat,&mut Mat) = unsafe{std::mem::transmute(p_fn)};
    run_next_stage_2(ctx,&mut segmap_mat,run_stage_seg_ood, img);

    let p_fn = run_stage_detect_deviations as *const ();
    let run_stage_detect_deviations:fn(&ctx,&mut Mat,&mut Mat) = unsafe{std::mem::transmute(p_fn)};
    run_next_stage_2(ctx,&mut segmap_mat,run_stage_detect_deviations, img);

    let verify = &ctx.verify_seg_gmm;
    if verify.is_empty() ==false{
        if find_file(&path, verify) == false{
            fs::create_dir_all(verify).unwrap();
        }
    }
    let C_classes = C.get("classes").unwrap().as_table().unwrap();

    let mut canvas = draw::draw_segmap(&segmap_mat,C_classes);
    let rows = canvas.shape()[0] as i32;
    let cols = canvas.shape()[1] as i32;
    let channels = 3;
    let data = canvas.as_mut_ptr();
    let step = (cols*channels)  as usize * std::mem::size_of::<u8>();
    let mat = unsafe{Mat::new_rows_cols_with_data(rows, cols, CV_8UC3,data as *mut std::ffi::c_void,step).unwrap()}; 

    imsave(&format!("./{}/{}.jpg",verify,&ctx.name), &mat, 100);

}

fn run_stage_seg_ood(ctx:&ctx,segmap:&mut Mat,img:&mut Mat){
    println!("Running sseg_ood");

    let name = &ctx.name;
    let C = &ctx.C;

    let save_dir =ctx.save_seg_ood.clone();
    let save_path = format!("{}/{}",save_dir,ctx.img_name);
    //println!("save_path is {}",save_path);
    let mut path = PathBuf::new();
    path.push(".");
    
    // if find_file(&path, &save_dir){
    //     path.push(&save_dir);
    //     if find_file(&path, &ctx.img_name.clone()){
    //         let mut segmap_mat =imread(&save_path,opencv::imgcodecs::IMREAD_GRAYSCALE).unwrap();//grey CV_8UC1
    //         let p_fn = run_stage_detect_foreigns as *const ();
    //         let run_stage_detect_foreigns:fn(&ctx,&mut Mat,&mut Mat) = unsafe{std::mem::transmute(p_fn)};
    //         run_next_stage_2(ctx,&mut segmap_mat,run_stage_detect_foreigns, img);
    //         return;
    //     }
    // }

    let tpl_dir = format!("{}/target_{}x/segmaps/{}.png",ctx.tpl_dir,ctx.zoom,ctx.cam_id);
    let tpl_segmap = imread(tpl_dir.as_str(),opencv::imgcodecs::IMREAD_GRAYSCALE).unwrap();

    let copper_cfg = ctx.C.get("target").unwrap().get("ood_seg").unwrap().get("copper").unwrap();
     
    let mask_value = C.get("classes").unwrap().get("copper").unwrap().get("label").unwrap().as_integer().unwrap() as f64;
    let mut copper_mask = Mat::default();
    compare(segmap,&mask_value,&mut copper_mask,CMP_EQ).unwrap();
    
    let (copper_ood_mask_, copper_mask_) = process_ood_seg(img,&copper_mask,copper_cfg,None,ctx.zoom);
    let wl_copper_label = C.get("classes").unwrap().get("wl_copper").unwrap().get("label").unwrap().as_integer().unwrap();
    let mut segmap_array = Mat2Array_2(segmap.clone());

    for ((i,j),value) in copper_ood_mask_.indexed_iter(){
        if *value {
            segmap_array[(i,j)] |= wl_copper_label as u8;
        }
        else{
            segmap_array[(i,j)] = segmap_array[(i,j)];
        }
    }
    //println!("segmap copper array is {:?}",segmap_array);
    let bg_cfg = ctx.C.get("target").unwrap().get("ood_seg").unwrap().get("bg").unwrap();
    //println!("cfg is {:#?}",bg_cfg);

     
    let mask_value = C.get("classes").unwrap().get("bg").unwrap().get("label").unwrap().as_integer().unwrap() as f64;
    let mut bg_mask = Mat::default();
    compare(segmap,&mask_value,&mut bg_mask,CMP_EQ).unwrap();
    let shadow_value = C.get("classes").unwrap().get("bg_shadow").unwrap().get("label").unwrap().as_integer().unwrap() as f64;
    let mut bg_shadow_mask = Mat::default();
    compare(&tpl_segmap,&shadow_value,&mut bg_shadow_mask,CMP_EQ).unwrap();

    let (bg_ood_mask_, bg_mask_) = process_ood_seg(img,&bg_mask,bg_cfg,Some(&bg_shadow_mask),ctx.zoom);
    let wl_bg_label = C.get("classes").unwrap().get("wl_bg").unwrap().get("label").unwrap().as_integer().unwrap();

    for ((i,j),value) in bg_ood_mask_.indexed_iter(){
        if *value {
            segmap_array[(i,j)] |= wl_bg_label as u8;
        }
        else{
            segmap_array[(i,j)] = segmap_array[(i,j)];
        }
    }
    
    let mut segmap_mat = base::Array2Mat(&segmap_array);//segmap_mat CV_8UC1
    if save_path.is_empty() == false{
        if find_file(&path, &save_dir)==false{
            fs::create_dir_all(save_dir).unwrap();
        }
    }
    imsave(&save_path, &segmap_mat,100);
    let p_fn = run_stage_detect_foreigns as *const ();
    let run_stage_detect_foreigns:fn(&ctx,&mut Mat,&mut Mat) = unsafe{std::mem::transmute(p_fn)};
    run_next_stage_2(ctx,&mut segmap_mat,run_stage_detect_foreigns, img);


    let verify = &ctx.verify_seg_ood;
    if verify.is_empty() ==false{
        if find_file(&path, verify) == false{

            println!("create directory...");
            fs::create_dir_all(verify).unwrap();
        }
    }
    let mut segmaps_hash = HashMap::new();
    let binding_1 = copper_mask_.mapv(|x| if x {255} else {0});
    let binding_2 = bg_mask_.mapv(|x| if x {255} else {0});
    segmaps_hash.insert("copper", &binding_1);
    segmaps_hash.insert("bg", &binding_2);

    let img_ptr = img.data_mut() as *mut u8;
    let row = img.rows() as usize;
    let col = img.cols() as usize;
    let deep = 3 as usize;
    let img_array =unsafe{ ArrayViewMut::from_shape_ptr((row,col,deep), img_ptr)}.to_owned();

//verify mat
    // let mut canvas = draw::verify_ood_seg(img_array, segmaps_hash);
    // let rows = canvas.shape()[0] as i32;
    // let cols = canvas.shape()[1] as i32;
    // let channels = 3;
    // let data = canvas.as_mut_ptr();
    // let step = (cols*channels)  as usize * std::mem::size_of::<u8>();
    // let verify_Mat = unsafe{Mat::new_rows_cols_with_data(rows, cols, CV_8UC3,data as *mut std::ffi::c_void,step).unwrap()}; 

    // println!("verify_Mat: {:#?}", verify_Mat);

    // let params:Vector<i32> = Vector::new();
    // imwrite(&format!("./{}/{}.jpg",verify,&ctx.name), &verify_Mat, &params).unwrap();

    // use ndarray::Array2;
    // use ndarray_npy::read_npy;

    // let mut arr1: Array2<u8> = read_npy("/home/mxz/python/pcb_test/data_bench_infer/segmap.npy").unwrap();
    // let rows = arr1.shape()[0] as i32;
    // let cols = arr1.shape()[1] as i32;
    // let channels = 3;
    // let data = arr1.as_mut_ptr();
    // let step = (cols*channels)  as usize * std::mem::size_of::<u8>();
    // let seg_mat = unsafe{Mat::new_rows_cols_with_data(rows, cols, CV_8UC3,data as *mut std::ffi::c_void,step).unwrap()}; 
    
    let C_classes = C.get("classes").unwrap().as_table().unwrap();
    //println!("C_clases is {:#?}",C_classes);

    let mut canvas = draw::draw_segmap(&segmap_mat,C_classes);
    let rows = canvas.shape()[0] as i32;
    let cols = canvas.shape()[1] as i32;
    let channels = 3;
    let data = canvas.as_mut_ptr();
    let step = (cols*channels)  as usize * std::mem::size_of::<u8>();
    let mat = unsafe{Mat::new_rows_cols_with_data(rows, cols, CV_8UC3,data as *mut std::ffi::c_void,step).unwrap()}; 

    imsave(&format!("./{}/{}.jpg",verify,&ctx.name),&mat,100);


}


fn run_stage_detect_foreigns(ctx:& ctx,segmap:&mut Mat,img:&mut Mat){
    println!("Running detect_foreigns");
    let start_time = time::now();
    let name = &ctx.name;
    let C = &ctx.C;

    let save_dir =ctx.save_foreigns.clone();
    let save_path = format!("{}/{}",save_dir,ctx.img_name);

    let mut path = PathBuf::new();
    path.push(".");

   let imgh = segmap.rows();
   let imgw = segmap.cols();

   let no_valid_mask = false;
   let mut mask:Option<Mat> = None;
   if no_valid_mask == false{
    let tile_vaild_mask_f = format!("{}/target_{}x/valid_masks/{}.png",ctx.tpl_dir,ctx.zoom,ctx.cam_id);
    let tpl_valid_mask = imread(&tile_vaild_mask_f,opencv::imgcodecs::IMREAD_GRAYSCALE).unwrap();
    mask = Some(tpl_valid_mask);
   }
   else{
    println!("no valid mask")
   }
   let cfg = C.get("foreign").unwrap();
   let classes = C.get("classes").unwrap();
 

   let (objs,groups) = foreign::detect_foreigns(segmap,img,classes,cfg,mask,ctx.zoom);

   let mut canvas = Mat::ones(imgh,imgw,CV_8UC4).unwrap().to_mat().unwrap();
   draw::draw_defects("foreigns", name.as_str(), img, &mut canvas, 
    objs, groups, "draw_box_forign", C.get("foreign").unwrap());

   let verify = &ctx.verify_foreigns;
   if verify.is_empty() ==false{
        if find_file(&path, verify) == false{

            println!("create foreign directory...");
            fs::create_dir_all(verify).unwrap();
        }
   }
   imsave(&format!("./{}/{}.png",verify,&ctx.name),&canvas,100);
   let end_time = time::now();
   println!("foreign time over {:?}",end_time-start_time);

}
fn run_stage_detect_deviations(ctx:& ctx,segmap:&mut Mat,img:&mut Mat){//需要gmm阶段segmap，可以与foreign并行
    println!("Running detect_deviations");
    let start_time = time::now();
    let name = &ctx.name;
    let C = &ctx.C;

    let mut path = PathBuf::new();
    path.push(".");
    
    let tpl_distmap_dir = format!("{}/target_{}x/distmaps/{}.jpg",ctx.tpl_dir,ctx.zoom,ctx.cam_id);
    let tpl_distmap = imread(tpl_distmap_dir.as_str(),opencv::imgcodecs::IMREAD_GRAYSCALE).unwrap();

    let tpl_segmap_dir = format!("{}/target_{}x/segmaps/{}.png",ctx.tpl_dir,ctx.zoom,ctx.cam_id);
    let tpl_segmap = imread(tpl_segmap_dir.as_str(),opencv::imgcodecs::IMREAD_GRAYSCALE).unwrap();

    let tpl_img_dir = format!("{}/target_{}x/images/{}.jpg",ctx.tpl_dir,ctx.zoom,ctx.cam_id);
    let tpl_img = imread(tpl_img_dir.as_str(),opencv::imgcodecs::IMREAD_GRAYSCALE).unwrap();


    println!("tpl_distmap is {:#?}",tpl_distmap);
    println!("tpl_segmap is {:#?}",tpl_segmap);

    let imh = img.rows();
    let imw = img.cols();

    let no_valid_mask = false;
    let mut mask:Option<Mat> = None;

    if no_valid_mask == false{
        let tile_vaild_mask_f = format!("{}/target_{}x/valid_masks/{}.png",ctx.tpl_dir,ctx.zoom,ctx.cam_id);
        let tpl_valid_mask = imread(&tile_vaild_mask_f,opencv::imgcodecs::IMREAD_GRAYSCALE).unwrap();
        mask = Some(tpl_valid_mask);
       }
       else{
        println!("no valid mask")
       }

    let (objs,groups) = detect_deviations(&name,&segmap,&tpl_segmap,&tpl_img,&tpl_distmap,img,C,&mask,ctx.zoom);

    let mut canvas = Mat::ones(imh,imw,CV_8UC4).unwrap().to_mat().unwrap();
    draw::draw_defects("deviation", name.as_str(), img, &mut canvas, 
        objs, groups, "draw_box_deviation", C.get("deviation").unwrap());

    let verify = &ctx.verify_deviations;
    if verify.is_empty() ==false{
            if find_file(&path, verify) == false{

                println!("create deviation directory...");
                fs::create_dir_all(verify).unwrap();
            }
    }
    imsave(&format!("./{}/{}.png",verify,&ctx.name),&canvas,100);
    let end_time = time::now();
    println!("deviation time over {:?}",end_time-start_time);
}


fn main() {

    let args:Vec<String> = env::args().collect();
    let zoom_str = &args[1];
    let mut stage = &"detect_foreigns_deviations".to_string();
    if args.len()> 2{
        stage = &args[2];
    }
    let zoom:i32 =  get_zoom(&zoom_str);
    println!("zoom: {:?}", zoom);

    let mut file = std::fs::File::open("Config.toml").unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let cfg = contents.parse::<Table>().unwrap();

    let cam_map =cfg.get("target").unwrap().get("cam_mapping").unwrap();
    //list.txt 中未注释的图片信息 包括编号 面料等
    let cls_info=read_list("List.txt", cam_map, false);
    //println!("{:#?}",cls_info);

    // 启动监听进程
    let mut p = Command::new("./target/debug/lisening")
        .stdout(Stdio::piped()) // 将子进程的标准输出重定向到管道
        .spawn()
        .expect("start listening process error");
    
    let mut p_stdout = BufReader::new(p.stdout.as_mut().expect("bufReader error"));
    let mut line = String::new();
    
    loop {
        //let (tx,rx) = unbounded();
        
        line.clear();   
        p_stdout.read_line(&mut line).unwrap();
        println!("read from listing process:{}", line);
        let listing_data:Vec<String> = line.trim().split(" ").map(|x|x.to_string()).collect();
        println!("listing data: {:?}", listing_data);
        
        let start_time = time::now();

        listing_data.par_iter().for_each(|img|{
            println!("msg is {:?}",img);
            let mut ctx = pre_img(&img, &cls_info,&cfg,&zoom);
            let mut start_img = get_start_img(&img, &zoom);
            thread_pool.install(||{run_stage(&ctx, &mut start_img)});
            
        });
        let end_time = time::now();
        println!("main time over {:?}",end_time-start_time);
        println!("that`s ok");
        
    }
    // 等待监听进程结束
    p.wait().unwrap();
}

fn pre_img(img:&String,cls_info: &HashMap<String, HashMap<String, String>>,cfg:&Map<String, Value>,zoom:&i32)->ctx{
    println!("img is {:?}",img);
    let img_name = base_name(img);
    println!("img_name is {:?}",img_name);
    let name =file_name_(&img_name);
    println!("name is {:?}",name);
    let info = cls_info.get(&name).unwrap();
    //println!("{:#?}",info);

    

    let mut tpl_dir = "templates/".to_string();
    let board = info.get("board").unwrap();
    tpl_dir.push_str(board);

    let ctx = ctx{
        C : cfg.clone(),
        name,
        img_name: img_name,
        tpl_dir: tpl_dir,
        board: board.to_string(),
        cam_id: info.get("cam_id").unwrap().to_string(),

        save_crop: String::from("save_crop"),
        save_filter:String::from("save_filter") ,
        save_resize_align: String::from("save_resize_align"),
        save_transform: String::from("save_transform"),
        save_aligned_images: String::from("save_aligned_images"),
        save_seg_gmm: String::from("save_seg_gmm"),
        save_seg_ood: String::from("save_seg_ood"),
        save_foreigns: String::from("save_foreigns"),
        save_deviations: String::from("save_deviations"),

        verify_transform: String::from("verify_transform"),
        verify_seg_gmm: String::from("verify_seg_gmm"),
        verify_seg_ood: String::from("verify_seg_ood"),
        verify_foreigns: String::from("verify_foreigns"),
        verify_deviations: String::from("verify_deviations"),

        zoom: *zoom,
    };
    ctx
}
fn get_start_img(img:&String,zoom:&i32)->Mat{
    let img_start_time = time::now();
    let start_img = imread(img,opencv::imgcodecs::IMREAD_COLOR).unwrap();

    let img_end_time = time::now();
    println!("img time over {:?}",img_end_time-img_start_time);
    //println!("start_img is {:#?}",start_img);
    let _scalar=Scalar::new(0.0,0.0,0.0,0.0);
    let mut resize_image = Mat::default();
    if *zoom > 1{
        resize(&start_img,&mut resize_image,
            Size_ { width: start_img.cols()/2, height: start_img.rows()/2 },
            0.0,0.0,3).unwrap();

        return resize_image;
    } 
    start_img

}
fn read_list(file_name: &str,cam_campping:&Value,return_dict :bool)->HashMap<String, HashMap<String, String>>{
    let mut data:Vec<String> = Vec::new();
    let mut map = HashMap::new();
    if let Ok(lines) = read_lines(file_name) {
        for line in lines {
            if line.as_ref().unwrap().to_string().starts_with("#") == false {
                data=line.unwrap().trim().split(' ').map(|x|x.to_string()).collect();

                let mut face = ' ';
                let mut cam_id =' ';
                match cam_campping.get(data[2].clone()).unwrap(){
                    Value::String(a) =>{
                    face = a.as_bytes()[0] as char;
                    cam_id = a.as_bytes()[1] as char;
                        },
                    _ => todo!(),
                }
                let mut tmp_map = HashMap::new();
                tmp_map.insert(String::from("name"),file_name_(&data[0]));
                tmp_map.insert(String::from("board"),data[1].clone());
                tmp_map.insert(String::from("face"),face.to_string());
                tmp_map.insert(String::from("cam_id"),cam_id.to_string());
                map.insert(file_name_(&data[0]), tmp_map);
            }
        }
    }
    map
}
fn file_name_(name:&str)->String{
    let data:Vec<String> = name.trim().split('.').map(|x|x.to_string()).collect();
    data[0].clone()
}
fn base_name(name:&String)->String{
    let data:Vec<String> = name.trim().split('/').map(|x|x.to_string()).collect();
    data[data.len() -1].clone()
}
fn get_zoom(zoom_str:&str)->i32{
    let data:Vec<String> = zoom_str.trim().split('=').map(|x|x.to_string()).collect();
    data[data.len() -1].clone().parse().unwrap()
}
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

