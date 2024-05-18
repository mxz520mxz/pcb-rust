use crate::base;
use crate::foreign;
use base::Array2Mat_3;
use ndarray::Array;
use ndarray::Array3;
use ndarray::ArrayBase;
use ndarray::Dim;
use ndarray::IxDyn;
use ndarray::OwnedRepr;
use opencv::core::*;
use opencv::imgproc::*;
use ndarray::Array2;
use opencv::types::VectorOfMat;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use toml::{Table, Value};
use toml::map::Map;

pub fn verify_align_camera(H:Vec<Vec<f32>>, src_img:&Mat,tpl_img:&Mat) -> VectorOfMat{

    let tpl_h = tpl_img.rows();
    let tpl_w = tpl_img.cols();
    let size = Size::new(tpl_w, tpl_h);
    let eye:Vec<Vec<f32>> = Array2::eye(3).rows().into_iter().
        map(|row|row.to_vec()).collect();
    let src_warped = transform_img(src_img, eye,size, Option::None,Option::None,Option::None);
   
    let canvas1 = draw_match_image(tpl_img,&src_warped);
   
    let warped_img = transform_img(src_img, H, size,Option::None,Option::None,Option::None);
    let canvas2 = draw_match_image(tpl_img,&warped_img);

    let mut canves = VectorOfMat::new();
    canves.insert(0, canvas1).unwrap();
    canves.insert(1, canvas2).unwrap();
    
    canves

}
pub fn transform_img(img:&Mat,T:Vec<Vec<f32>>,size:Size,mut order:Option<&str>,mut border_mode:Option<i32>,mut border_value_:Option<f64>)->Mat{
    if border_mode == Option::<i32>::None{
        border_mode = Some(BORDER_CONSTANT);
    }
    if border_value_ == Option::<f64>::None{
        border_value_ = Some(0.0);
    }
    let _value = border_value_.unwrap();
    let border_value = Scalar::all(_value);
    //let max_l = std::cmp::max(img.rows(),img.cols());
    let order = get_order(order);
    let mut wrap_img = Mat::default(); 
    let m = Mat::from_slice_2d(T.as_slice()).unwrap();
    warp_perspective(img,&mut wrap_img,&m,size,order,border_mode.unwrap(),border_value).unwrap();

    
    wrap_img
}   
fn get_order(order: Option<&str>) -> i32 {
    let order = match order {
        Some(method) => match method.to_uppercase().as_str() {
            "NEAREST" => INTER_NEAREST,
            "LINEAR" => INTER_LINEAR,
            "CUBIC" => INTER_CUBIC,
            "LANCZOS4" => INTER_LANCZOS4,
            _ => INTER_LINEAR,
        },
        None => INTER_LINEAR,
    };
    order
}
fn draw_match_image(img1:&Mat, img2:&Mat)->Mat{
    let mut mul_img1=channel_split_mul_scalar(img1,[1.0,0.5,0.0]);
    let mut mul_img2=channel_split_mul_scalar(img2,[0.0,0.5,1.0]);
    
    let mut mix_together_img = Mat::default();
    let scalar=Scalar::new(0.0,0.0,0.0,0.0);
   
    add(&mul_img1, &mul_img2, &mut mix_together_img, &no_array(), -1).expect("add error");
    mix_together_img
}

fn channel_split_mul_scalar(img:&Mat,_scalar:[f64;3]) ->Mat{
   
    let mut img_channel1 = Mat::default();
    let mut img_channel2 = Mat::default();
    let mut img_channel3 = Mat::default();
   
    let mut spilit = VectorOfMat::new();
    spilit.insert(0, img_channel1).unwrap();
    spilit.insert(1, img_channel2).unwrap();
    spilit.insert(2, img_channel3).unwrap();
    split(img, &mut spilit).unwrap();
 

    let a = spilit.get(0).expect("get a error").mul(&_scalar[0], 1.0).unwrap().to_mat().expect("error");
    let b = spilit.get(1).expect("get b error").mul(&_scalar[1], 1.0).unwrap().to_mat().expect("get error");
    let c = spilit.get(2).expect("get c error").mul(&_scalar[2], 1.0).unwrap().to_mat().unwrap();
   
    let mut merge_img =Mat::default();
    merge(&spilit, &mut merge_img).expect("merge error");
    merge_img
}


pub fn draw_segmap(img:&Mat,C:&Map<String, Value>) -> ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>>{
    let imgh = img.rows();
    let imgw = img.cols();

    let canvans = Mat::zeros(imgh, imgw, CV_8UC3).unwrap().to_mat().unwrap();
    let mut canvans_array = base::Mat2Array_3(canvans);
    let mut canvans_array_view = canvans_array.view_mut();
    
        
    for (cls_name,v) in C.iter(){
        let value = v.get("label").unwrap().as_integer().unwrap() as f64;
        let color:Vec<u8> = v.get("color").unwrap()
            .as_array().unwrap().iter().map(|x|x.as_integer().unwrap() as u8).collect();
       
        let mut bitand_mat = Mat::default();
        bitwise_and(img, &value, &mut bitand_mat, &no_array()).unwrap();
        let mut compare_mat = Mat::default();
        compare(&bitand_mat, &0.0, &mut compare_mat, CMP_GT).unwrap();

        let array_compare = base::Mat2Array_2(compare_mat);
        let array_compare_view = array_compare.view();
        
        for ((i,j),value) in array_compare_view.indexed_iter() {
             if *value > 0 {
                canvans_array_view[[i,j,0]] = color[2];
                canvans_array_view[[i,j,1]] = color[1];
                canvans_array_view[[i,j,2]] = color[0];
             }
        }
    }
    let canvans_ = canvans_array_view.to_owned();
    canvans_

}

pub fn verify_ood_seg(img: Array3<u8>,segmaps: HashMap<&str, &ArrayBase<OwnedRepr<u8>, Dim<[usize; 2]>>>) 
    -> ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>>{
    let copper_mask = segmaps["copper"];
    let bg_mask = segmaps["bg"];

    let a = img.mapv(|x| x / 2);
    let b = Array::from_shape_fn(
        (copper_mask.shape()[0],copper_mask.shape()[1],3), |(i,j,k)|{
            let v = copper_mask[(i,j)];
            if k == 0 {
                v*80
            }
            else {
                v*0
            }
        });

    let c = Array::from_shape_fn(
        (bg_mask.shape()[0],bg_mask.shape()[1],3), |(i,j,k)|{
            let v = copper_mask[(i,j)];
            if k == 1 {
                v*80
            }
            else {
                v*0
            }
        });

    println!("a is {:?}",a);
    println!("a shape is {:?}",a.shape());
    println!("b is {:?}",b);
    println!("b shape is {:?}",b.shape());
    println!("c is {:?}",c);
    println!("c shape is {:?}",c.shape());
    
    let mut canvas = a+b+c; 
    println!("canvas is {:?}",canvas);
    println!("canvas shape is {:?}",canvas.shape());

    canvas
   

}

use foreign::Children;
use foreign::obj;
use base::find_file;

fn draw_box_forign(level:&str,area:i32)->([i32;3],i32){
    if level == "black"{
        let color = [255,0,0];
        let thickness = 10;
        (color,thickness)
    }
    else if level == "gray"{
        let color = [255,255,0];
        let thickness = 5;
        (color,thickness)
    }
    else if area > 10{
        let color = [0,255,0];
        let thickness = 5;
        (color,thickness)
    }
    else{
        let color = [213,233,0];
        let thickness = 0;
        (color,thickness)
    }
}

fn draw_box_deviation(tp:&str)->([i32;3],i32){
    if tp == "concave"{
        let color = [255,0,0];
        let thickness = 10;
        (color,thickness)
    }
    else if tp == "convex"{
        let color = [255,255,0];
        let thickness = 5;
        (color,thickness)
    }
    else{
        let color = [213,233,0];
        let thickness = 0;
        (color,thickness)
    }
}

pub fn draw_defects(detect_type:&str,name:&str,img:&Mat,canvas:&mut Mat,defects_objs:Vec<obj>,
    defects_groups:HashMap<i32, Children>,box_fn:&str,cfg:&Value){

        for obj in defects_objs.iter(){

            if obj.level == " " && obj.located == " " && obj.id == 0{
                continue;
            }
            let color = cfg.get(obj.tp).unwrap().get("color").unwrap().as_array().unwrap();
            
            let (x0,y0,ww,hh) = obj.bbox;
            let x1 = x0 + ww;
            let y1 = y0 + hh;
            let m = &obj.mask;

            for row in y0..y1{
                for col in x0..x1{
                    let value = canvas.at_2d_mut::<Vec4b>(row, col).unwrap();
                    if *m.at_2d::<u8>(row-y0, col-x0).unwrap() > 0{
                        *value.get_mut(0).unwrap() = color[0].as_integer().unwrap() as u8;
                        *value.get_mut(1).unwrap() = color[1].as_integer().unwrap() as u8;
                        *value.get_mut(2).unwrap() = color[2].as_integer().unwrap() as u8;
                        *value.get_mut(3).unwrap() = 255;
                    }
                }
            }
        }

        let r = 64;
        let h = canvas.rows();
        let w = canvas.cols();

        if detect_type == "deviations"{
            let convex_dir = String::from("./deviations_patches/convex");
            let concave_dir = String::from("./deviations_patches/concave");
            let deviations_group_dir = String::from("./deviations_patches/group");

            fs::create_dir_all(convex_dir).unwrap();
            fs::create_dir_all(concave_dir).unwrap();
            fs::create_dir_all(deviations_group_dir).unwrap();            
        }

        if detect_type == "foreign"{
            let black_dir = "./foreigns_patches/black";
            let gray_dir = "./foreigns_patches/gray";
            let white_dir = "./foreigns_patches/white";
            let foreigns_group_dir = "./foreigns_patches/group";

            fs::create_dir_all(black_dir).unwrap();
            fs::create_dir_all(gray_dir).unwrap();
            fs::create_dir_all(white_dir).unwrap();  
            fs::create_dir_all(foreigns_group_dir).unwrap();  
        }

        for (gid,v) in defects_groups.iter(){
            if *gid < 0 {
    
                for oid in &v.child{
                    let o = &defects_objs[*oid];
                    if box_fn == "draw_box_forign"{
                        let (color,tickness) = draw_box_forign(o.level,o.area);
                        let (x0,y0,ww,hh) = o.bbox;
                        let c = Scalar::new(color[0] as f64,color[1] as f64,color[2] as f64,255.0);
                        put_text(canvas, format!("S{}",o.id).as_str(),
                             Point_{x:x0-r,y:y0-r-10}, FONT_HERSHEY_SIMPLEX, 1.0, c, 
                             2,LINE_AA, false).unwrap();
                        rectangle(canvas,Rect_ { x: x0-r, y: y0-r, width: ww+r+r, height: hh+r+r },c,
                        tickness,LINE_8,0).unwrap();
                    }
                    else{
                        let (color,tickness) = draw_box_deviation(o.tp);
                        let (x0,y0,ww,hh) = o.bbox;
                        
                        // if ww+r+x0 > w || hh+r+y0 > h{
                        //     continue;
                        // }
                        let c = Scalar::new(color[0] as f64,color[1] as f64,color[2] as f64,255.0);
                        put_text(canvas, format!("S{}",o.id).as_str(),
                             Point_{x:x0-r,y:y0-r-10}, FONT_HERSHEY_SIMPLEX, 1.0, c, 
                             2,LINE_AA, false).unwrap();
                        rectangle(canvas,Rect_ { x: x0-r, y: y0-r, width: ww+r+r, height: hh+r+r },c,
                        tickness,LINE_8,0).unwrap();
                    }
                }
            }
            else{
                if box_fn == "draw_box_forign"{
                    let (color,tickness) = draw_box_forign(&v.level,v.area);
                    let (x0,y0,ww,hh) = v.bbox; 
                    let c = Scalar::new(color[0] as f64,color[1] as f64,color[2] as f64,255.0);
                    put_text(canvas, format!("G{}",gid).as_str(),
                            Point_{x:x0-r,y:y0-r-10}, FONT_HERSHEY_SIMPLEX, 1.0, c, 
                            2,LINE_AA, false).unwrap();
                    rectangle(canvas,Rect_ { x: x0-r, y: y0-r, width: ww+r+r, height: hh+r+r },c,
                    tickness,LINE_8,0).unwrap();
                }
                else{
                    let (color,tickness) = draw_box_deviation(&v.level);
                    let (x0,y0,ww,hh) = v.bbox; 
                    let c = Scalar::new(color[0] as f64,color[1] as f64,color[2] as f64,255.0);
                    put_text(canvas, format!("G{}",gid).as_str(),
                            Point_{x:x0-r,y:y0-r-10}, FONT_HERSHEY_SIMPLEX, 1.0, c, 
                            2,LINE_AA, false).unwrap();
                    rectangle(canvas,Rect_ { x: x0-r, y: y0-r, width: ww+r+r, height: hh+r+r },c,
                    tickness,LINE_8,0).unwrap();
                }
            }
        }

    
}