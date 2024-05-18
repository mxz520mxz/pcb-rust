use std::collections::HashMap;
use std::sync::MutexGuard;
use crate::base;
use base::imsave;
use lazy_static::lazy_static;
use rayon::ThreadPool;
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::sync::Mutex;
use std::sync::{Arc,RwLock};
use std::thread;
use linfa_nn::distance::L2Dist;
use opencv::types:: VectorOfPoint;
use opencv::{core::*, imgproc::*, types::VectorOfMat};
use opencv::prelude::*;
use ndarray::*;
use toml::{Table, Value};
use base::{disk};
use rand::thread_rng;
use rand::seq::index::sample as rand_sample;      
use linfa_nn::*;

use crate::utils;    


#[derive(Debug)]
pub struct obj<'a>{
    pub tp:&'a str,
    pub area:i32,
    pub centroid:(f64,f64),
    pub bbox:(i32,i32,i32,i32),
    pub mask:Mat,
    pub id:usize,
    pub level:&'a str,
    pub located:&'a str,
    pub group:i32,
}
impl<'a> obj<'a> {
    pub fn update_id(&mut self, id:usize){
        self.id = id;
    }
    pub fn update_level(&mut self, level:&'a str){
        self.level = level;
    }
    pub fn update_located(&mut self, located:&'a str){
        self.located = located;
    }
    pub fn update_group(&mut self, group:i32){
        self.group = group;
    }
}
#[derive(Debug)]
struct out<'a>{
    detector_config:Value,
    segmap: Mat,
    object:Vec<obj<'a>>
}

#[derive(Debug)]
pub struct Children{
    pub area:i32,
    pub bbox:(i32,i32,i32,i32),
    pub level:String,
    pub child:Vec<usize>,
}

const N_WORKERS: usize = 20;
#[macro_use]
//extern crate lazy_static;
lazy_static!{
    static ref thread_pool:ThreadPool = rayon::ThreadPoolBuilder::new().
        num_threads(N_WORKERS).build().unwrap();
}

fn expand_rprop_mask(coordinate:(i32, i32, i32, i32),gap:i32,shape:(i32,i32),in_mask:Mat)
    ->(Mat,(i32, i32, i32, i32)){

    let (imh,imw) = shape;
    let (yy0,xx0,yy1,xx1) = coordinate;

    let x0 = (xx0 - gap).max(0);
    let y0 = (yy0-gap).max(0);
    let x1 = (xx1+gap).min(imw);
    let y1 = (yy1+gap).min(imh);

    let off_x = xx0 - x0;
    let off_y = yy0 - y0;

    let mut mask = Mat::zeros(y1-y0, x1-x0, CV_8UC1).unwrap().to_mat().unwrap();
    for i in off_y..off_y+in_mask.rows(){
        for j in off_x..off_x+in_mask.cols(){
            *mask.at_2d_mut::<u8>(i, j).unwrap() = *in_mask.at_2d::<u8>(i-off_y,j-off_x).unwrap();
        }
    }

    return (mask,(x0,y0,x1,y1))
}

fn is_obj_overlap_mask(coordinate:(i32, i32, i32, i32),mask:&Mat,rp_image:&Mat)->bool{
    let (x,y,width,height) = coordinate;
    let m = Mat::roi(&mask,Rect_{x:x,y:y,width:width,height:height}).unwrap();
    let mut ret = Mat::default();
    bitwise_and(&m, rp_image, &mut ret, &no_array()).unwrap();
    let sum = count_non_zero(&ret).unwrap();
    return sum > 0;
}

fn segment_object_from_seeds(img:&Mat,seeds:Vector::<Point_<i32>>,img_mask:&Mat,
    bg_init:&mut Mat,valid_mask:Option<&Mat>,method:&str,floodfill_tol:f64,id:i32)->Mat{

        let mut tol_fill = Mat::zeros(img.rows(), img.cols(), CV_8UC1).unwrap().to_mat().unwrap();
        let lo_tol = Scalar::all(floodfill_tol);
        let up_tol = Scalar::all(floodfill_tol);

        for point in seeds.iter(){
            let mut canvas = img.clone();
            let mut canvas_mask = Mat::ones(img.rows()+2, img.cols()+2, CV_8UC1).unwrap().to_mat().unwrap();
            for i in 0..img_mask.rows(){
                for j in 0..img_mask.cols(){
                    *canvas_mask.at_2d_mut::<u8>(i+2, j+2).unwrap() = *img_mask.at_2d::<u8>(i, j).unwrap();
                }
            }
            bitwise_not(&canvas_mask.clone(), &mut canvas_mask, &no_array()).unwrap();

            flood_fill(&mut canvas, point,
            Scalar::all(1.0), &mut Rect::default(),
            lo_tol, up_tol, 8).unwrap();
            let params:Vector<i32> = Vector::new();
    
            let mut tmp = Mat::default();
            compare(&canvas, &1.0, &mut tmp, CMP_EQ).unwrap();
            bitwise_or(&tmp, &tol_fill.clone(), &mut tol_fill, &no_array()).unwrap();
        }

        let mut bg_init_not = Mat::default();
        bitwise_not(&bg_init.clone(), bg_init, &no_array()).unwrap();
        bitwise_and(&tol_fill.clone(), bg_init, &mut tol_fill, &no_array()).unwrap();

        if let Some(mat) = valid_mask{
            bitwise_and(&tol_fill.clone(), mat, &mut tol_fill, &no_array()).unwrap()
        }
        tol_fill
    }

fn detect_holes(roi:&Mat,wl_mask:&Mat,imdic:&HashMap<&str, Mutex<Mat>>,cfg:&Value,zoom:i32) -> (Mat,Mat){
    let I = &imdic["I"].lock().unwrap();
    let imh = I.rows();
    let imw = I.cols();
   
    let mut labels = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();

    let mut not_roi = Mat::default();
    bitwise_not(&roi, &mut not_roi, &no_array()).unwrap();
   
    let nccomps = connected_components_with_stats (
        &not_roi, 
        &mut labels,     
        &mut stats, 
        &mut centroids,
        4,CV_32S
        ).unwrap();
    let max_area = utils::get_zoomed_area(cfg.get("max_area").unwrap(), zoom) as i32;
    //去除最小孔洞
    let mut clean = not_roi.clone();
    for i in 0..roi.rows(){
        for j in 0..roi.cols(){
            let label = labels.at_2d::<i32>(i, j).unwrap();
            if *label > 0{
                let area = *stats.at_2d::<i32>(*label,CC_STAT_AREA).unwrap();
                if  area < max_area { 
                    *clean.at_2d_mut::<u8>(i,j).unwrap() = 0;
                }
            }
        }
    }

    let mut diff = Mat::default();
    bitwise_xor(&not_roi, &clean, &mut diff, &no_array()).unwrap();
    let nccomps = connected_components_with_stats (
        &diff, 
        &mut labels,    
        &mut stats, 
        &mut centroids,
        4,CV_32S
        ).unwrap();

    let radius = utils::get_zoomed_len(cfg.get("surr_radius").unwrap(), zoom) as i32;
    let crop_border = utils::get_zoomed_len(cfg.get("crop_border").unwrap(), zoom) as i32;
    
    let mut tgt_mask = Mat::zeros(imh, imw, CV_8UC1).unwrap().to_mat().unwrap();
    for i in 0..nccomps{
        let left = stats.at_2d::<i32>(i, CC_STAT_LEFT).unwrap();   
        let top = stats.at_2d::<i32>(i, CC_STAT_TOP).unwrap();     
        let width= stats.at_2d::<i32>(i, CC_STAT_WIDTH).unwrap();  
        let height = stats.at_2d::<i32>(i, CC_STAT_HEIGHT).unwrap(); 
        let in_mask = Mat::roi(&diff,Rect_{x:*left,y:*top,width:*width,height:*height}).unwrap();

        if is_obj_overlap_mask((*left,*top,*width,*height), 
            &tgt_mask, &in_mask) {
                continue;
            }
        if *top < crop_border || *height > imw -crop_border {
            println!("continue ... left {}, top {}, height {}, width {}", left, top, height,width);
            continue;
        }
        
        let (rp_mask,(x0,y0,x1,y1)) = expand_rprop_mask(
            (*top,*left,top+height,left+width), radius, (imh,imw), in_mask);  

        let sub_img = Mat::roi(&I,Rect_{x:x0,y:y0,width:x1-x0,height:y1-y0}).unwrap();

        let mut sub_valid = Mat::roi(&wl_mask.clone(),Rect_{x:x0,y:y0,width:x1-x0,height:y1-y0}).unwrap();
        
        let mut non_zero_points = Vector::<Point_<i32>>::new();
        find_non_zero(&rp_mask, &mut non_zero_points).unwrap();
        //find_topk_points()
        let meth = cfg.get("method").unwrap().as_str().unwrap();
        let floodfill_tol= utils::get_zoomed_len(cfg.get("floodfill_tol").unwrap(), zoom);

        let fill_mask = segment_object_from_seeds(&sub_img,non_zero_points,&rp_mask,&mut sub_valid,
            None,meth,floodfill_tol as f64,i);
            
        if count_non_zero(&fill_mask).unwrap() > 0{
            for i in y0..y1{
                for j in x0..x1{
                    let piexl = *fill_mask.at_2d::<u8>(i - y0, j - x0).unwrap();
                    if piexl > 0{
                        *clean.at_2d_mut::<u8>(i, j).unwrap() = 0;
                        *tgt_mask.at_2d_mut::<u8>(i, j).unwrap() = piexl;
                    }
                }
            }
        }
        else{
            continue;
        }
        
    }

    bitwise_not(&clean.clone(), &mut clean, &no_array()).unwrap();
    (clean,tgt_mask)
}

fn filter_holes(nccomps:i32,stats:&Mat,diff:&Mat,obj_mask:&mut Mat)
    ->(Vec<i32>,Vec<i32>){
    let imh = obj_mask.rows();
    let imw = obj_mask.cols();
    let kernel = disk(1);
    let mut cand_rprops:Vec<i32> = vec![];
    let mut out:Vec<i32> = vec![];
    for i in 1..nccomps{
        let left = stats.at_2d::<i32>(i, CC_STAT_LEFT).unwrap();   
        let top = stats.at_2d::<i32>(i, CC_STAT_TOP).unwrap();     
        let width= stats.at_2d::<i32>(i, CC_STAT_WIDTH).unwrap();  
        let height = stats.at_2d::<i32>(i, CC_STAT_HEIGHT).unwrap(); 
        let in_mask = Mat::roi(&diff,Rect_{x:*left,y:*top,width:*width,height:*height}).unwrap();

    
        let (rp_mask,(x0,y0,x1,y1)) = expand_rprop_mask(
            (*top,*left,top+height,left+width), 1, (imh,imw), in_mask);  

        let mut ext_rp_mask = Mat::default();
        dilate(&rp_mask, &mut ext_rp_mask, &kernel,Point_{x:-1,y:-1}, 1, BORDER_CONSTANT, morphology_default_border_value().unwrap()).expect("erode failed");
        let mut count_mat = Mat::default();
        let roi_mask = Mat::roi(&obj_mask, Rect_{x:x0,y:y0,width:x1-x0,height:y1-y0}).unwrap();
        bitwise_and(&ext_rp_mask, &roi_mask, &mut count_mat, &no_array()).unwrap();
        
        if count_non_zero(&count_mat).unwrap() > 0{
            out.push(i);
        }
        else{
            cand_rprops.push(i);
        }
        
    }
    (cand_rprops,out)
    
}

fn filter_small_area(rest:Vec<i32>,stats:&Mat,diff:&Mat,foreign_map:&Mat,max_area:i32)
    ->(Vec<i32>,Vec<i32>){
    let mut out = Vec::new();
    let mut cand_rprops = Vec::new();
    for i in rest.iter(){
        let left = stats.at_2d::<i32>(*i, CC_STAT_LEFT).unwrap();   
        let top = stats.at_2d::<i32>(*i, CC_STAT_TOP).unwrap();     
        let width= stats.at_2d::<i32>(*i, CC_STAT_WIDTH).unwrap();  
        let height = stats.at_2d::<i32>(*i, CC_STAT_HEIGHT).unwrap(); 
        let in_mask = Mat::roi(&diff,Rect_{x:*left,y:*top,width:*width,height:*height}).unwrap();

        if is_obj_overlap_mask((*left,*top,*width,*height), 
            foreign_map, &in_mask) {
                continue;
            }
        let area = *stats.at_2d::<i32>(*i,CC_STAT_AREA).unwrap();
        if area <= max_area {
            out.push(*i);
        }
        else{
            cand_rprops.push(*i);
        }
    }
    (cand_rprops,out)
}

fn find_topk_points(img:&Mat,mask:&Mat,topk:i32,reverse:bool)->Vector<Point_<i32>>{
    let mut non_zero_points = Vector::<Point_<i32>>::new();
    find_non_zero(mask, &mut non_zero_points).unwrap();
    let mut intensities = Vector::<u8>::new();

    for value in non_zero_points.iter() {
        intensities.push(*img.at_2d::<u8>(value.y, value.x).unwrap());
    }

    let mut min_val:f64 = Default::default();
    let mut max_val:f64 = Default::default();
    let mut min_loc= Point::default();
    let mut max_loc= Point::default();
    
    min_max_loc(&intensities,Some(&mut min_val),Some(&mut max_val),
        Some(&mut min_loc),Some(&mut max_loc),&no_array()).unwrap();
    let mut ret = Vector::default();
    if reverse{
        ret.push(non_zero_points.get(max_loc.x as usize).unwrap());
        ret
    }
    else{
        ret.push(non_zero_points.get(min_loc.x as usize).unwrap());
        ret
        
    }

}
fn filter_similar_color(rest:Vec<i32>,stats:&Mat,diff:&Mat,foreign_map:&mut Mat,imdic:&HashMap<&str, Mutex<Mat>>,
    roi:&Mat,wl_mask:&Mat,cfg:&Value,zoom:i32)->(Vec<i32>,Vec<i32>){
    let I = &imdic["I"].lock().unwrap();
    let imh = I.rows();
    let imw = I.cols();
    let mut out = Vec::new();
    let mut cand_rprops = Vec::new();
    let radius = utils::get_zoomed_len(cfg.get("surr_radius").unwrap(), zoom) as i32;
    let kernel = disk(3);

    for i in rest.iter(){
        //println!("start rprops is {}",i);
        let left = stats.at_2d::<i32>(*i, CC_STAT_LEFT).unwrap();   
        let top = stats.at_2d::<i32>(*i, CC_STAT_TOP).unwrap();     
        let width= stats.at_2d::<i32>(*i, CC_STAT_WIDTH).unwrap();  
        let height = stats.at_2d::<i32>(*i, CC_STAT_HEIGHT).unwrap(); 
        let in_mask = Mat::roi(&diff,Rect_{x:*left,y:*top,width:*width,height:*height}).unwrap();

        if is_obj_overlap_mask((*left,*top,*width,*height), 
            foreign_map, &in_mask) {
                continue;
            }
        let (rp_mask,(x0,y0,x1,y1)) = expand_rprop_mask(
            (*top,*left,top+height,left+width), radius, (imh,imw), in_mask);
        
        let sub_img = Mat::roi(&I,Rect_{x:x0,y:y0,width:x1-x0,height:y1-y0}).unwrap();
        let valid_mask = Mat::roi(roi,Rect_{x:x0,y:y0,width:x1-x0,height:y1-y0}).unwrap();
        let bg_init = Mat::roi(&wl_mask.clone(),Rect_{x:x0,y:y0,width:x1-x0,height:y1-y0}).unwrap();

        let mut min_val:f64 = Default::default();
        let mut max_val:f64 = Default::default();
        let mut min_loc= Point::default();
        let mut max_loc= Point::default();
        
        min_max_loc(&sub_img,Some(&mut min_val),Some(&mut max_val),
            Some(&mut min_loc),Some(&mut max_loc),&no_array()).unwrap();
        let max_intensity_range = utils::get_zoomed_len(cfg.get("max_intensity_range").unwrap(), zoom) as f64;
        if max_val - min_val > max_intensity_range {
            cand_rprops.push(*i);
            continue;
        }
        
        let mut bg_erode = Mat::default();
        erode(&bg_init, &mut bg_erode, &kernel, Point_{x:-1,y:-1}, 1, BORDER_CONSTANT, morphology_default_border_value().unwrap()).expect("erode failed");
        
        let seed_top_k =1;
        let seed_mode = cfg.get("seed_mode").unwrap().as_str().unwrap();
        let seeds = find_topk_points(&sub_img, &rp_mask, seed_top_k, seed_mode=="light");

        let meth = cfg.get("method").unwrap().as_str().unwrap();
        let floodfill_tol= utils::get_zoomed_len(cfg.get("floodfill_tol").unwrap(), zoom);

        let fill_mask = segment_object_from_seeds(&sub_img,seeds,&rp_mask,&mut bg_erode.clone(),
            Some(&valid_mask),meth,floodfill_tol as f64,*i);
        if count_non_zero(&fill_mask).unwrap() > 0{
            for i in y0..y1{
                for j in x0..x1{
                    let piexl = *fill_mask.at_2d::<u8>(i - y0, j - x0).unwrap();
                    if piexl > 0{
                        *foreign_map.at_2d_mut::<u8>(i, j).unwrap() = piexl;
                    }
                }
            }
        }
        else{
            out.push(*i);
            continue;
        }

        let mut bg_not = Mat::default();
        bitwise_not(&bg_erode, &mut bg_not, &no_array()).unwrap();
        let mut expand_area = Mat::default();
        bitwise_and(&bg_not,&valid_mask, &mut expand_area, &no_array()).unwrap();
        let mut tmp_mask = Mat::default();
        bitwise_or(&fill_mask,&rp_mask,&mut tmp_mask,&no_array()).unwrap();
        let mut rp_mask_not = Mat::default();
        bitwise_not(&rp_mask,&mut rp_mask_not, &no_array()).unwrap();
        let mut expand_mask = Mat::default();
        bitwise_and(&tmp_mask, &rp_mask_not, &mut expand_mask, &no_array()).unwrap();

        let fill_max_factor = cfg.get("fill_max_factor").unwrap().as_float().unwrap();
        let expand_mask_sum = count_non_zero(&expand_mask).unwrap() as f64;
        let expand_area_sum = count_non_zero(&expand_area).unwrap() as f64;

        if expand_mask_sum > fill_max_factor*expand_area_sum{
            out.push(*i);
        }
        else{
            cand_rprops.push(*i);
        }
    }

    (cand_rprops,out)
}
fn get_spilit_max_min(img:&Mat)->[f64;6]{
    let mut img_channel1 = Mat::default();
    let mut img_channel2 = Mat::default();
    let mut img_channel3 = Mat::default();

    let mut spilt_mat = VectorOfMat::new();
    spilt_mat.insert(0, img_channel1).unwrap();
    spilt_mat.insert(1, img_channel2).unwrap();
    spilt_mat.insert(2, img_channel3).unwrap();
    split(&img, &mut spilt_mat).unwrap();

    let mut min_val:f64 = Default::default();
    let mut max_val:f64 = Default::default();
    let mut min_loc= Point::default();
    let mut max_loc= Point::default();
    
    min_max_loc(&spilt_mat.get(0).unwrap(),Some(&mut min_val),Some(&mut max_val),
        Some(&mut min_loc),Some(&mut max_loc),&no_array()).unwrap();
    let min_val_0 = min_val.clone();
    let max_val_0 = max_val.clone();

    min_max_loc(&spilt_mat.get(1).unwrap(),Some(&mut min_val),Some(&mut max_val),
        Some(&mut min_loc),Some(&mut max_loc),&no_array()).unwrap();
    let min_val_1 = min_val.clone();
    let max_val_1 = max_val.clone();

    min_max_loc(&spilt_mat.get(2).unwrap(),Some(&mut min_val),Some(&mut max_val),
        Some(&mut min_loc),Some(&mut max_loc),&no_array()).unwrap();
    let min_val_2 = min_val.clone();
    let max_val_2 = max_val.clone();

    [min_val_0,max_val_0,min_val_1,max_val_1,min_val_2,max_val_2]
}
fn calc_dist(a:[f64;2],b:[f64;2])->f64{

    let x0 = a[0];
    let y0 = a[1];
    let x1 = b[0];
    let y1 = b[1];

    ((x1 - x0).powf(2.0) + (y1 - y0).powf(2.0)).powf(0.5)
}
fn filter_different_color(rest:Vec<i32>,stats:&Mat,diff:&Mat,foreign_map:&mut Mat,imdic:&HashMap<&str, Mutex<Mat>>,
    roi:&Mat,wl_mask:&Mat,cfg:&Value,zoom:i32)->(Vec<i32>,Vec<i32>){

        let lab = &imdic["lab"].lock().unwrap();
        let imh = lab.rows();
        let imw = lab.cols();
        let mut out = Vec::new();
        let mut cand_rprops = Vec::new();
        let radius = utils::get_zoomed_len(cfg.get("surr_radius").unwrap(), zoom) as i32;
        let surr_kernel = disk(radius);

        for i in rest.iter() {
            //println!("different start rprops is {}",i);
            let left = stats.at_2d::<i32>(*i, CC_STAT_LEFT).unwrap();   
            let top = stats.at_2d::<i32>(*i, CC_STAT_TOP).unwrap();     
            let width= stats.at_2d::<i32>(*i, CC_STAT_WIDTH).unwrap();  
            let height = stats.at_2d::<i32>(*i, CC_STAT_HEIGHT).unwrap(); 
            let in_mask = Mat::roi(&diff,Rect_{x:*left,y:*top,width:*width,height:*height}).unwrap();

            if is_obj_overlap_mask((*left,*top,*width,*height), 
                foreign_map, &in_mask) {
                    continue;
                }
            let (rp_mask,(x0,y0,x1,y1)) = expand_rprop_mask(
                (*top,*left,top+height,left+width), radius, (imh,imw), in_mask);
            
            let sub_img = Mat::roi(lab,Rect_{x:x0,y:y0,width:x1-x0,height:y1-y0}).unwrap();
            let intensity = get_spilit_max_min(&sub_img);
            let rp_min = intensity[0];
            let rp_max = intensity[1];
            let rp_rb_min = [intensity[2],intensity[4]];
            let rp_rb_max = [intensity[3],intensity[5]];
            
            let min_intensity_var = utils::get_zoomed_len(cfg.get("min_intensity_var").unwrap(), zoom) as f64;
            let min_rb_var = utils::get_zoomed_len(cfg.get("min_rb_var").unwrap(), zoom) as f64;
            if rp_max - rp_min > min_intensity_var && calc_dist(rp_rb_min, rp_rb_max) > min_rb_var{
                for i in y0..y1{
                    for j in x0..x1{
                        let piexl = *rp_mask.at_2d::<u8>(i - y0, j - x0).unwrap();
                        if piexl > 0{
                            *foreign_map.at_2d_mut::<u8>(i, j).unwrap() = piexl;
                        }
                    }
                }
                out.push(*i);
                continue;
            }
            let mut surr_mask = Mat::default();
            dilate(&rp_mask, &mut surr_mask, &surr_kernel,Point_{x:-1,y:-1}, 1, BORDER_CONSTANT, morphology_default_border_value().unwrap()).expect("erode failed");
            bitwise_xor(&surr_mask.clone(), &rp_mask, &mut surr_mask, &no_array()).unwrap();

            // culc percentile
            let mut count = 0;
            let mut surr_pix_mean =[0.0,0.0,0.0];
            for i in 0..surr_mask.rows(){
                for j in 0..surr_mask.cols(){
                    if *surr_mask.at_2d::<u8>(i,j).unwrap() > 0{
                        let a = sub_img.at_2d::<Vec3b>(i,j).unwrap();
                        let a0 = a.get(0).unwrap();
                        let a1 = a.get(1).unwrap();
                        let a2 = a.get(2).unwrap();

                        surr_pix_mean[0] += *a0 as f64;
                        surr_pix_mean[1] += *a1 as f64;
                        surr_pix_mean[2] += *a2 as f64;
                        count += 1;
                    }
                }
            }
            surr_pix_mean[0] /= count as f64;
            surr_pix_mean[1] /= count as f64;
            surr_pix_mean[2] /= count as f64;
            
            let seed_mode = cfg.get("seed_mode").unwrap().as_str().unwrap();
            if seed_mode == "dark_gray"{
                if surr_pix_mean[0] - rp_min > min_intensity_var && calc_dist([surr_pix_mean[1],surr_pix_mean[2]], rp_rb_min) > min_rb_var{
                    for i in y0..y1{
                        for j in x0..x1{
                            let piexl = *rp_mask.at_2d::<u8>(i - y0, j - x0).unwrap();
                            if piexl > 0{
                                *foreign_map.at_2d_mut::<u8>(i, j).unwrap() = piexl;
                            }
                        }
                    }
                    out.push(*i);
                    continue;
                }
            }
            else if seed_mode == "dark"{
                if surr_pix_mean[0] - rp_min > min_intensity_var {
                    for i in y0..y1{
                        for j in x0..x1{
                            let piexl = *rp_mask.at_2d::<u8>(i - y0, j - x0).unwrap();
                            if piexl > 0{
                                *foreign_map.at_2d_mut::<u8>(i, j).unwrap() = piexl;
                            }
                        }
                    }
                    out.push(*i);
                    continue;
                }
            }
            else if seed_mode == "light"{
                if rp_max - surr_pix_mean[0] > min_intensity_var {
                    for i in y0..y1{
                        for j in x0..x1{
                            let piexl = *rp_mask.at_2d::<u8>(i - y0, j - x0).unwrap();
                            if piexl > 0{
                                *foreign_map.at_2d_mut::<u8>(i, j).unwrap() = piexl;
                            }
                        }
                    }
                    out.push(*i);
                    continue;
                }
            }
            cand_rprops.push(*i);
        }

        (cand_rprops,out)
}

fn update_output(tp:&'static str,out:Arc<Mutex<out>>,nccomps:Vec<i32>,
    stats:Arc<Mutex<Mat>>,centroids:Arc<Mutex<Mat>>,mask:Arc<Mutex<Mat>>){

    let mut out_mutex = out.lock().unwrap();
    let stats = stats.lock().unwrap();
    let centroids = centroids.lock().unwrap();
    let mask = mask.lock().unwrap();

    for i in nccomps.iter(){
        let area = *stats.at_2d::<i32>(*i,CC_STAT_AREA).unwrap();
        let centroid = (*centroids.at_2d::<f64>(*i,0).unwrap(),*centroids.at_2d::<f64>(*i,1).unwrap());
        let x = stats.at_2d::<i32>(*i, CC_STAT_LEFT).unwrap();   
        let y = stats.at_2d::<i32>(*i, CC_STAT_TOP).unwrap();     
        let width= stats.at_2d::<i32>(*i, CC_STAT_WIDTH).unwrap();  
        let height = stats.at_2d::<i32>(*i, CC_STAT_HEIGHT).unwrap(); 
        let bbox = (*x,*y,*width,*height);
        let in_mask = Mat::roi(&mask,Rect_{x:*x,y:*y,width:*width,height:*height}).unwrap();

        let obj = obj{
            tp:tp,
            area:area,
            centroid:centroid,
            bbox:bbox,
            mask:in_mask,
            id:0,
            level:" ",
            located:" ",
            group:-1,
        };
        out_mutex.object.push(obj);
    }
    drop(out_mutex);
}

fn detect_foreign_on_copper(continent:Arc<Mutex<Mat>>,land:Arc<Mutex<Mat>>,imdic:Arc<Mutex<HashMap<&str, Mutex<Mat>>>>,out:Arc<Mutex<out>>
    ,mask:Arc<Mutex<Option<Mat>>>,cfg:&Value,zoom:i32)->Mat{
        
        let land = &mut *land.lock().unwrap();
       
        let (continent_,init_inland_sea_) = detect_holes(&continent.lock().unwrap(),&land,&imdic.lock().unwrap(),cfg.get("inland_sea").unwrap(),zoom);
        let imh = continent_.rows();
        let imw = continent_.cols();
        let mut init_inland_sea = init_inland_sea_.clone();
        drop(init_inland_sea_);

        let mut continent = Mat::default();
        let shrink = utils::get_zoomed_len(cfg.get("copper_margin").unwrap(), zoom);
        let kernel=disk(shrink as i32);
        erode(&continent_, &mut continent, &kernel, Point_{x:-1,y:-1}, 1, BORDER_CONSTANT, morphology_default_border_value().unwrap()).expect("erode failed");
        let mut continent_tmp = Mat::default();
        erode(&continent, &mut continent_tmp, &disk(1), Point_{x:-1,y:-1}, 1, BORDER_CONSTANT, morphology_default_border_value().unwrap()).expect("erode failed");
        let mut coastline = Mat::default();
        bitwise_xor(&continent, &continent_tmp, &mut coastline, &no_array()).unwrap();
        drop(continent_tmp);
        //bitwise_not(&continent, &mut continent_tmp, &no_array()).unwrap();

        for i in 0..imh{
            for j in 0..imw{
                if *continent.at_2d::<u8>(i, j).unwrap() == 0{
                    *land.at_2d_mut::<u8>(i, j).unwrap() = 0;
                }
            }
        }
        
        let binding = mask.lock().unwrap();
        let mask = binding.as_ref();
        if let Some(mat) = mask{
            bitwise_and(&init_inland_sea.clone(), mat, &mut init_inland_sea, &no_array()).unwrap();
            bitwise_and(&continent.clone(), mat, &mut continent, &no_array()).unwrap();
            bitwise_and(&coastline.clone(), mat, &mut coastline, &no_array()).unwrap();
            bitwise_and(&land.clone(), mat, land, &no_array()).unwrap();
        }
        drop(binding);

        let mut diff = Mat::default();
        bitwise_xor(&continent, land, &mut diff, &no_array()).unwrap(); 
      
        let mut labels = Mat::default();
        let mut stats = Mat::default();
        let mut centroids = Mat::default();
        let nccomps = connected_components_with_stats (
            &diff, 
            &mut labels,    
            &mut stats, 
            &mut centroids,
            4,CV_32S
            ).unwrap();

        let (rest,obj_nccomps) = filter_holes(nccomps,&stats,&diff,&mut init_inland_sea);
      
        let mutex_stats = Arc::new(Mutex::new(stats));
        let mutex_centroids = Arc::new(Mutex::new(centroids));
        let mutex_diff = Arc::new(Mutex::new(diff));
        
        let (tx_mat,rx_mat) = unbounded();
        let (_,(rest,obj_nccomps)) = thread_pool.join(||{
            let out_lock = out.clone();
            tx_mat.send(out_lock.lock().unwrap().segmap.clone()).unwrap();
            update_output("inland_sea", out_lock, obj_nccomps, 
                mutex_stats.clone(), mutex_centroids.clone(),
                mutex_diff.clone())
        },||{
            let max_area = utils::get_zoomed_area(cfg.get("small_pond").unwrap().get("max_area").unwrap(), zoom);
            let segmap = rx_mat.recv().unwrap();
            let (rest,obj_nccomps) = filter_small_area(rest, 
                &mutex_stats.lock().unwrap(), &mutex_diff.lock().unwrap(), 
                &segmap, max_area as i32);
            (rest,obj_nccomps)
        });

        let continent = Mutex::new(continent);
        let land = Mutex::new(land);
 
        let (_,(rest,obj_nccomps)) = rayon::join(||{
            let out_lock = out.clone();
            tx_mat.send(out_lock.lock().unwrap().segmap.clone()).unwrap();
            update_output("small_pond", out_lock, obj_nccomps, 
                mutex_stats.clone(), mutex_centroids.clone(),
                mutex_diff.clone())
        },||{
            let mut segmap = rx_mat.recv().unwrap();
            let (rest,obj_nccomps) = filter_similar_color(rest,&mutex_stats.lock().unwrap(),
             &mutex_diff.lock().unwrap(),&mut segmap, 
             &imdic.lock().unwrap(),&continent.lock().unwrap(),&land.lock().unwrap(),cfg.get("shallow_water").unwrap(),zoom);
            (rest,obj_nccomps)
        });
        
        let (_,(_,obj_nccomps)) = rayon::join(||{
            let out_lock = out.clone();
            tx_mat.send(out_lock.lock().unwrap().segmap.clone()).unwrap();
            update_output("shallow_water", out_lock, obj_nccomps, 
                mutex_stats.clone(), mutex_centroids.clone(),
                mutex_diff.clone())
        },||{
            let mut segmap = rx_mat.recv().unwrap();
            let (rest,obj_nccomps) = filter_different_color(rest,&mutex_stats.lock().unwrap(), &mutex_diff.lock().unwrap(), 
            &mut segmap, &imdic.lock().unwrap(),&continent.lock().unwrap(),
            &land.lock().unwrap(),cfg.get("deep_water").unwrap(),zoom);
            (rest,obj_nccomps)
        });
    
        update_output("deep_water", out.clone(), obj_nccomps,mutex_stats.clone(), mutex_centroids.clone(),
          mutex_diff.clone());
        let ret_continent = continent.lock().unwrap().clone();
        ret_continent

        
}

fn detect_foreign_on_bg(_ocean:Arc<Mutex<Mat>>,sea:Arc<Mutex<Mat>>,imdic:Arc<Mutex<HashMap<&str, Mutex<Mat>>>>,out:Arc<Mutex<out>>
    ,mask:Arc<Mutex<Option<Mat>>>,cfg:&Value,zoom:i32)->Mat{
        let _ocean = &*_ocean.lock().unwrap();
        let sea = &mut *sea.lock().unwrap();
        
        let imh = _ocean.rows();
        let imw = _ocean.cols();
        let mut ocean = Mat::default();
        let bg_margin = utils::get_zoomed_len(cfg.get("bg_margin").unwrap(), zoom) as i32;
        
        erode(_ocean, &mut ocean, &disk(bg_margin), Point_{x:-1,y:-1}, 1, BORDER_CONSTANT, morphology_default_border_value().unwrap()).expect("erode failed");
        let mut coastline = Mat::default();
        let mut ocean_tmp = Mat::default();
        erode(&ocean, &mut ocean_tmp, &disk(1), Point_{x:-1,y:-1}, 1, BORDER_CONSTANT, morphology_default_border_value().unwrap()).expect("erode failed");
        bitwise_xor(&ocean, &ocean_tmp, &mut coastline, &no_array()).unwrap();
        drop(_ocean);drop(ocean_tmp);

        for i in 0..imh{
            for j in 0..imw{
                if *ocean.at_2d::<u8>(i, j).unwrap() == 0{
                    *sea.at_2d_mut::<u8>(i, j).unwrap() = 0;
                }
            }
        }
       
        let (mut ocean,mut init_insea_land) = detect_holes(&ocean,&sea,&imdic.lock().unwrap(),cfg.get("insea_land").unwrap(),zoom);
        
        let binding = mask.lock().unwrap();
        let mask = binding.as_ref();
        if let Some(mat) = mask{
            bitwise_and(&init_insea_land.clone(), mat, &mut init_insea_land, &no_array()).unwrap();
            bitwise_and(&ocean.clone(), mat, &mut ocean, &no_array()).unwrap();
            bitwise_and(&coastline.clone(), mat, &mut coastline, &no_array()).unwrap();
            bitwise_and(&sea.clone(), mat, sea, &no_array()).unwrap();
        }
        drop(binding);
        println!("on bg ....................");
        let mut diff = Mat::default();
        bitwise_xor(&ocean, sea, &mut diff, &no_array()).unwrap();
        //imsave("A_copper_bg.jpg",&diff,100);
       
        let mut labels = Mat::default();
        let mut stats = Mat::default();
        let mut centroids = Mat::default();
        let nccomps = connected_components_with_stats (
            &diff, 
            &mut labels,    
            &mut stats, 
            &mut centroids,
            4,CV_32S
            ).unwrap();

            

        let (rest,obj_nccomps) = filter_holes(nccomps,&stats,&diff,&mut init_insea_land);
        
        let mutex_stats = Arc::new(Mutex::new(stats));
        let mutex_centroids = Arc::new(Mutex::new(centroids));
        let mutex_diff = Arc::new(Mutex::new(diff));
        update_output("insea_land", out.clone(), obj_nccomps,
            mutex_stats.clone(), mutex_centroids.clone(), mutex_diff.clone());
        
        let max_area = utils::get_zoomed_area(cfg.get("small_reef").unwrap().get("max_area").unwrap(), zoom);
        let (rest,obj_nccomps) = filter_small_area(rest, &mutex_stats.lock().unwrap(), &mutex_diff.lock().unwrap(), &out.lock().unwrap().segmap, max_area as i32);

        update_output("small_reef", out.clone(), obj_nccomps,
            mutex_stats.clone(), mutex_centroids.clone(), mutex_diff.clone());
            
        let (rest,obj_nccomps) = filter_similar_color(rest,&mutex_stats.lock().unwrap(), &mutex_diff.lock().unwrap(), 
            &mut out.lock().unwrap().segmap, &imdic.lock().unwrap(),&ocean,&sea,cfg.get("shallow_sand").unwrap(),zoom);
      
        update_output("shallow_sand", out.clone(), obj_nccomps, 
            mutex_stats.clone(), mutex_centroids.clone(), mutex_diff.clone());

        let (rest,obj_nccomps) = filter_different_color(rest,&mutex_stats.lock().unwrap(), &mutex_diff.lock().unwrap(), 
            &mut out.lock().unwrap().segmap, &imdic.lock().unwrap(),&ocean,&sea,cfg.get("high_sand").unwrap(),zoom);
       
        update_output("high_sand", out.clone(), obj_nccomps, mutex_stats.clone(), mutex_centroids.clone(), mutex_diff.clone());
        
        ocean

        
}

pub fn detect_foreigns<'a>(segmap:&'a Mat,img:&'a Mat,classes:&'a Value,cfg:&'a Value,mask:Option<Mat>,zoom:i32) -> (Vec<obj<'a>>,HashMap<i32, Children>){
    
    let imdic = get_foreign_img(img, true, false, true);       
    
    let foreign_map = Mat::zeros(segmap.rows(),segmap.cols(),CV_8UC1).unwrap().to_mat().unwrap();
    let mut out = Arc::new(Mutex::new(out{detector_config:cfg.clone(),segmap:foreign_map,object:Vec::new()}));

    let copper_label = classes.get("copper").unwrap().get("label").unwrap().as_integer().unwrap() as f64;
    let wl_copper_label = classes.get("wl_copper").unwrap().get("label").unwrap().as_integer().unwrap() as f64;

    let mut bitand_mat = Mat::default();
    bitwise_and(segmap, &copper_label, &mut bitand_mat, &no_array()).unwrap();
    let mut continent = Mat::default();
    compare(&bitand_mat, &0.0, &mut continent, CMP_GT).unwrap();

    bitwise_and(segmap, &wl_copper_label, &mut bitand_mat, &no_array()).unwrap();
    let mut land = Mat::default();
    compare(&bitand_mat, &0.0, &mut land, CMP_GT).unwrap();

    let bg_label = classes.get("bg").unwrap().get("label").unwrap().as_integer().unwrap() as f64;
    let wl_bg_label = classes.get("wl_bg").unwrap().get("label").unwrap().as_integer().unwrap() as f64;
    
    let mut ocean = Mat::default();
    bitwise_and(&segmap, &bg_label, &mut ocean, &no_array()).unwrap();
    compare(&ocean.clone(), &0.0, &mut ocean, CMP_GT).unwrap();
    
    let mut sea = Mat::default();
    bitwise_and(&segmap, &wl_bg_label, &mut sea, &no_array()).unwrap();
    compare(&sea.clone(), &0.0, &mut sea, CMP_GT).unwrap();

    let imdic = Arc::new(Mutex::new(imdic));
    let mask = Arc::new(Mutex::new(mask.clone()));

    let (continent,ocean) = thread_pool.join(||{
        let continent = Arc::new(Mutex::new(continent));
        let land = Arc::new(Mutex::new(land));
        let mask = mask.clone();
        let imdic = imdic.clone();
       
        let x =detect_foreign_on_copper(
            continent,
            land,
            imdic,
            out.clone(),
            mask,
            cfg,
            zoom
        );
        x
     
    },||{
        let ocean = Arc::new(Mutex::new(ocean));
        let sea = Arc::new(Mutex::new(sea));
        let mask = mask.clone();
        let imdic = imdic.clone();
        
        let y =detect_foreign_on_bg(
            ocean,
            sea,
            imdic,
            out.clone(),
            mask,
            cfg,
            zoom
        );
        y
    });

    for i in 0..ocean.rows(){
        for j in 0..ocean.cols(){
            if *continent.at_2d::<u8>(i, j).unwrap() > 0{
                *out.lock().unwrap().segmap.at_2d_mut::<u8>(i, j).unwrap() |= copper_label as u8;
            }
            if *ocean.at_2d::<u8>(i, j).unwrap() > 0{
                *out.lock().unwrap().segmap.at_2d_mut::<u8>(i, j).unwrap() |= bg_label as u8;
            }
        } 
    }

   
    let mut guard = out.lock().unwrap();

    let objs = &mut guard.object;
    
    let mut cluster_objs = Vec::<&mut obj>::new();
    for (idx,o) in objs.iter_mut().enumerate() {
        let type_ = o.tp;
        let c = cfg.get(type_).unwrap();
        if type_.starts_with("small_"){
            continue;
        }
        
        let level = c.get("level").unwrap().as_str().unwrap();
        let located = c.get("located").unwrap().as_str().unwrap();
        o.update_id(idx);
        o.update_level(level);
        o.update_located(located);

        cluster_objs.push(o);
    }

    if cluster_objs.len() > 0 {
        let grp_labels = cluster_defects(&cluster_objs, (segmap.rows(),segmap.cols()),cfg.get("cluster").unwrap());
        for (i,obj) in cluster_objs.iter_mut().enumerate() {
            obj.update_group(grp_labels[i]);
        }
    }

    let groups = build_group(&objs);

    let objs = std::mem::replace(&mut guard.object, Vec::new());
    drop(guard);
    (objs, groups)

    
}

pub fn build_group(objs:&Vec<obj>) -> HashMap<i32, Children>{
    let mut groups = HashMap::new();
    let mut children = Children{area:0,bbox:(0,0,0,0),level:String::from(" "),child:vec![]};
    groups.insert(-1,children);

    for obj in objs.iter(){
        if obj.group == -1{
            groups.get_mut(&-1).unwrap().child.push(obj.id);
            continue;
        }

        let gid = obj.group;
        let mut children = Children{area:0,bbox:(0,0,0,0),level:String::from(" "),child:vec![]};

        if !groups.contains_key(&gid){
            groups.insert(gid.clone(), children);
        }
        groups.get_mut(&gid).unwrap().child.push(obj.id);
    }

    for (gid,v) in groups.iter_mut(){
        if *gid == -1{
            continue;
        }
        let mut children = vec![];
        for oid in v.child.iter(){
            children.push(&objs[*oid]);
        }
        let (mut x0,mut y0,mut w0,mut h0) = (i32::MAX,i32::MAX,i32::MIN,i32::MIN);
        let mut area = 0;
        let mut level_stats = HashMap::new();
        for i in children.iter(){
            let (x,y,w,h) = i.bbox;
            if x < x0 {x0 = x};
            if y < y0 {y0 = y};
            if w > w0 {w0 = w};
            if h > h0 {h0 = h};

            area += i.area;

            level_stats.insert(i.level.clone(), 1);
        } 
        v.bbox = (x0,y0,w0,h0);
        v.area = area;
        
        if level_stats.contains_key("black"){
            v.level = "black".to_string();
        }
        else if level_stats.contains_key("gray"){
            v.level = "gray".to_string();
        }
        else if level_stats.contains_key("white"){
            v.level = "white".to_string();
        }
    } 
    groups
}

pub fn cluster_defects(objs:&Vec<&mut obj>,shape:(i32,i32),cluster_cfg:&Value)->Vec<i32>{
    let (row,col) = shape;
    let cluster_dist = cluster_cfg.get("cluster_dist").unwrap().as_integer().unwrap();
    let nn_k = cluster_cfg.get("nn_k").unwrap().as_integer().unwrap();
    let max_edge_points = cluster_cfg.get("max_edge_points").unwrap().as_integer().unwrap() as usize;

    let nn_k = (objs.len()).min(nn_k as usize);
    let mut edge_list:Vector<Vector<Point>> = Vector::new();
    let mut centers:Array2<f64> = Array::zeros((0, 2));
    for o in objs.iter() {
        centers.push_row(ArrayView::from(&[o.centroid.0,o.centroid.1])).unwrap();

        let (off_x,off_y,_,_) = o.bbox;
        let m = &o.mask;
        let mut contours:Vector<Vector<Point>> = Vector::new();
        find_contours(m, &mut contours, 
            RETR_LIST, CHAIN_APPROX_NONE,Point::default()).unwrap();
        let mut rest_point = VectorOfPoint::new();
        for edge_points in contours.iter() {
            if max_edge_points > 0{
                let mut rng = thread_rng();
                let edge_len = edge_points.len();
                let length = max_edge_points.min(edge_len);
                let index = rand_sample(&mut rng, edge_len, length);
         
                for i in index.iter(){
                    rest_point.push(Point_{x:edge_points.get(i).unwrap().x+off_x, y:edge_points.get(i).unwrap().y+off_y});
                }
            }
        }   
        edge_list.push(rest_point);  
    }
 
    let tree = BallTree::new();

    let NerarestNeighbor = tree.from_batch(&centers, L2Dist).unwrap();
    let (rows,_) = centers.dim();
    let mut nn_idxs:Vec<Vec<usize>> = Vec::new();
    for i in 0..rows{
        let query = centers.row(i);
        let nearest = NerarestNeighbor.k_nearest(query.view(), nn_k).unwrap();
        let mut nearest_index = Vec::new();
        let mut count=0;
        for (_,index) in nearest.iter(){
            if count > 0{nearest_index.push(*index);}
            count+=1;
        }
        nn_idxs.push(nearest_index);
    }
    
    let dismap = calc_edge_distmap(edge_list, nn_idxs);

    let cluster_label = dbscan(&dismap,cluster_dist as f64,1);
    println!("clusters is {:#?} ",cluster_label);
    cluster_label
    
}

fn calc_edge_dist(pts0:&Vector<Point_<i32>>,pts1:Vector<Point_<i32>>)->f64{
    let mut min_d = f64::INFINITY;
    let mut d = 0.0;
    for point_0 in pts0.iter() {
        for point_1 in pts1.iter() {
            d = (point_0.x as f64 - point_1.x as f64).powf(2.0) + (point_0.y as f64 - point_1.y as f64).powf(2.0);
            if d < min_d{
                min_d = d;
            }
        }
    }
    min_d.powf(0.5)
}

fn calc_edge_distmap(edge_list:Vector<Vector<Point>>,nn_idxs:Vec<Vec<usize>>)
    ->ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>{
    let N = nn_idxs.len();

    let mut out = Array2::<f64>::ones((N,N))*10000.0;

    for (i,edge_i) in edge_list.iter().enumerate(){

        for j in &nn_idxs[i]{
            let edge_j = edge_list.get(*j).unwrap();
            out[(i,*j)] = calc_edge_dist(&edge_i,edge_j);
        }
    }
    out
}

fn find_core_objects(similarity_matrix: &Array2<f64>, epsilon: f64, min_points: usize) -> Vec<usize> {
    let n = similarity_matrix.nrows();
    let mut core_objects = vec![];
    for i in 0..n {
        let mut neighbors = vec![];
        for j in 0..n {
            if similarity_matrix[(i, j)] < epsilon {
                neighbors.push(j);
            }
        }
        if neighbors.len() >= min_points {
            core_objects.push(i);
        }
    }
    core_objects
}

fn dbscan(similarity_matrix: &Array2<f64>, epsilon: f64, min_points: usize) -> Vec<i32> {
    let core_objects = find_core_objects(&similarity_matrix, epsilon, min_points);

    let n = similarity_matrix.nrows();
    let mut clusters = vec![];
    let mut visited = vec![false; n];
    for core_object in core_objects {
        if visited[core_object] {
            continue;
        }
        visited[core_object] = true;

        let mut cluster = vec![core_object];
        let mut i = 0;
        while i < cluster.len() {
            let object = cluster[i];
            for j in 0..n {
                if similarity_matrix[(object, j)] < epsilon {
                    if !visited[j] {
                        visited[j] = true;
                        cluster.push(j);
                    }
                }
            }
            i += 1;
        }

        clusters.push(cluster);
    }
    let mut label = Vec::with_capacity(n);
    label.extend(std::iter::repeat(-1).take(n));
    for (id,i) in clusters.iter().enumerate() {
        for j in 0..n{
            if i.contains(&j){
                label[j] = id as i32;
            }
        }
    }
    label
}

fn get_foreign_img(img:&Mat,mut with_lab:bool,mut with_grad:bool,mut with_gray:bool)->HashMap<&'static str, Mutex<Mat>>{
    let mut imdic = HashMap::new();
    
    imdic.insert("img",Mutex::new(img.clone()));
    if with_grad{
        with_gray = true;
    }
    if with_lab && !imdic.contains_key("lab"){
        let mut lab_img = Mat::default();
        cvt_color(&img, &mut lab_img, ColorConversionCodes::COLOR_BGR2Lab as i32, 0).expect("convert img to color error");
        imdic.insert("lab", Mutex::new(lab_img));
    }

    if with_gray && !imdic.contains_key("I"){
        if imdic.contains_key("lab"){
            let mut img_channel1 = Mat::default();
            let mut img_channel2 = Mat::default();
            let mut img_channel3 = Mat::default();
        
            let mut spilt_mat = VectorOfMat::new();
            spilt_mat.insert(0, img_channel1).unwrap();
            spilt_mat.insert(1, img_channel2).unwrap();
            spilt_mat.insert(2, img_channel3).unwrap();
            split(&img, &mut spilt_mat).unwrap();

            imdic.insert("I", Mutex::new(spilt_mat.get(0).unwrap()));
        }
        else{
            let mut grey_img = Mat::default();
            cvt_color(&img, &mut grey_img, ColorConversionCodes::COLOR_BGR2GRAY as i32, 0).expect("convert img to color error");
            imdic.insert("I", Mutex::new(grey_img));
        }
    }
    imdic
}
