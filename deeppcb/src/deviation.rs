use std::collections::HashMap;
use crate::align_edge;
use crate::base;
use crate::draw;
use base::imsave;
use draw::transform_img;
use foreign::{obj, cluster_defects, build_group, Children};
use toml::map::Map;
use opencv::types:: VectorOfPoint;
use opencv::{core::*, imgproc::*};
use toml::{Table, Value};
use base::{disk};
use tch::Tensor;

use crate::{utils, foreign};


fn unique_contours(c:Vector<Point_<i32>>)->Vector<Point_<i32>>{
    let mut prev = None;
    let mut out = VectorOfPoint::new();

    for i in c.iter(){
        if Some(i) != prev{
            prev = Some(i);
            out.push(i);
        }
    }
    out
}

fn filter_contours(conts:Vector<Vector<Point>>, crop_mask:&Mat)->Vector<Vector<Point>>{
    let mut out:Vector<Vector<Point>> = Vector::new();
    for edge_points in conts.iter(){
        // let c = unique_contours(edge_points);
        let mut rest_point = VectorOfPoint::new();
        for point in edge_points.iter() {
            let value = crop_mask.at_2d::<u8>(point.y, point.x).unwrap();
            if *value > 0{
                rest_point.push(point);
            }
        }
        if !rest_point.is_empty(){
            out.push(rest_point);
        }
    }
    out
}

fn check_is_closed(c:&Vec<Point_<i32>>)->bool{
    let n = c.len();
    let a = c[n-1];
    let b = c[0];
    a == b
}

fn compute_nn_dists(c: &Vec<Point2i>) -> Vec<f64> {
    c.windows(2)
        .map(|pair| {
            let diff_x = (pair[1].x - pair[0].x) as f64;
            let diff_y = (pair[1].y - pair[0].y) as f64;
            (diff_x * diff_x + diff_y * diff_y).sqrt()
        })
        .collect()
}

fn split_parts(mut c:Vec<Point_<i32>>,mut breakpoints:Vec<usize>)->Vec<Vec<Point_<i32>>>{
    if breakpoints.len() == 0{
        return vec![c];
    }
    if breakpoints.len() == 1{
        let bp = breakpoints[0];
        let split_c = c.split_off(bp+1);
        return vec![c,split_c];
    }
    let mut out = vec![];
    breakpoints.reverse();
    for bp in breakpoints.iter() {
        let split_c = c.split_off(bp+1);
        out.push(split_c);
    }
    out.push(c);
    out
}
fn split_conts(conts:Vector<Vector<Point>>)->Vec<Vec<Point_<i32>>>{
    let mut out = vec![];
    for c in conts.iter(){
        let n = c.len();
        let c = c.to_vec();
        let is_closed = check_is_closed(&c);
        let mut nn_dist = compute_nn_dists(&c);
        
        if !is_closed {
            let mut last_d = compute_nn_dists(&vec![c[n-1],c[0]]);
            nn_dist.append(&mut last_d);
        }
        // println!("c {:?}", c);
        // println!("nn_dist {:?}", nn_dist);
        let mut breakpoints = vec![];
        for (i,v) in nn_dist.iter().enumerate() {
            if *v > 1.5{
                breakpoints.push(i);
            }
        }
        println!("breakpoints {}", breakpoints.len());
        let mut parts = split_parts(c, breakpoints);
        out.append(&mut parts);
    }
    out
}


fn ectract_contours(img:&Mat,valid_mask:&Mat,inside_mask:&Option<Mat>)->Vec<Vec<Point_<i32>>>{
    let crop_mask = valid_mask;
    let mut contours:Vector<Vector<Point>> = Vector::new();
    find_contours(img, &mut contours, 
        RETR_LIST, CHAIN_APPROX_NONE ,Point::default()).unwrap();
    
    println!("detect conts {:?}",contours.len());
    let conts = filter_contours(contours, &crop_mask);

    println!("filter conts {:?}",conts.len());
    let conts = split_conts(conts);
    println!("split conts {:?}",conts.len());

    if let Some(mat) = inside_mask{
        let conts = conts.into_iter().filter(
            |x|x.iter().any(|point|
                *mat.at_2d::<u8>(point.y, point.x).unwrap() > 0
                )).collect::<Vec<_>>();      
        return conts;  
    }
    conts
}
fn flood_fill_segment(cont:&Vec<Point_<i32>>,dists:&Vec<u8>,seeds:Vec<usize>,connect_len:i32)->Vec<Vec<usize>>{
    let n = cont.len();
    let mut visited:Vec<usize> = Vec::new();
    let mut out:Vec<Vec<usize>> = Vec::new();
    for seed in seeds.clone(){
        if visited.contains(&seed){
            continue;
        }
        visited.push(seed.clone());
        let mut patience = 0;

        let mut l:Vec<usize> = Vec::new();
        
        let mut prev_dist = dists[seed] as f64;
        for j in (0..seed).rev(){
            if visited.contains(&j){
                continue;
            }
            visited.push(j.clone());

            let d = dists[j] as f64;
            if seeds.contains(&j){
                prev_dist = d;
                l.push(j);
                continue;
            }
           
            if d < prev_dist{
                prev_dist = d;
                patience = 0;
                l.push(j);
                continue;
            }

            patience += 1;
            if patience <= connect_len{
                l.push(j);
                continue;
            }
            else{
                break;
            }
        }
        l.push(seed.clone());
        let mut prev_dist = dists[seed] as f64;
        for j in seed+1 .. n{
            if visited.contains(&j){
                continue;
            }
            visited.push(j.clone());

            let d = dists[j] as f64;
            if seeds.contains(&j){
                prev_dist = d;
                l.push(j);
                continue;
            }

            if d < prev_dist{
                prev_dist = d;
                patience = 0;
                l.push(j);
                continue;
            }

            patience += 1;
            if patience <= connect_len{
                l.push(j);
                continue;
            }
            else{
                break;
            }
        }
        out.push(l); 
    } 
    out
}
fn search_id(c:&Vec<Point_<i32>>,v:Point_<i32>)->i32{
    for (k,point) in c.iter().enumerate(){
        if *point == v{
            return k as i32;
        }
    }
    -1
}
fn filter_conts_by_dist(conts:&Vec<Vec<Point_<i32>>>,tpl_distmap:&Mat,
    far_dist_th:&i32,far_ratio:&f64,near_dist_th:&u8,connect_len:&i32,inside_mask:&Option<Mat>)
    ->(Vec<Vec<Point_<i32>>>,Vec<Vec<Point_<i32>>>){
        let mut outside_mask = Mat::default();
        let mut has_inside_mask = false;
        if let Some(mat) = inside_mask{
            bitwise_not(&mat, &mut outside_mask, &no_array()).unwrap();
            has_inside_mask = true;
        }
        let mut cand_conts:Vec<Vec<Point_<i32>>> = Vec::new();
        let mut tgt_conts:Vec<Vec<Point_<i32>>> = Vec::new(); 

        for (k,c) in conts.iter().enumerate(){
            let mut dists:Vec<u8> = Vec::new();
            if c.len() == 0{
                continue;
            }
            for point in c.iter(){
                let d = *tpl_distmap.at_2d::<u8>(point.y, point.x).unwrap();
                dists.push(d);
            } 
            println!("dists len {}",dists.len());
            let dist_max = dists.iter().max().unwrap();
            if *dist_max <= *near_dist_th{
                println!("{} dist_max is too small",k);
                continue;
            }
            let mut sel:Vec<bool> = dists.iter().map(|x| *x > *far_dist_th as u8).collect();
            for (idx,point) in c.iter().enumerate() {
                if sel[idx]{
                    let p = Point_{x:point.x,y:point.y+1};
                    let id = search_id(c, p);
                    if id > -1{
                        if !sel[id as usize] {
                            sel[idx] = false;
                        }
                    }

                    let p = Point_{x:point.x,y:point.y-1};
                    let id = search_id(c, p);
                    if id > -1{
                        if !sel[id as usize] {
                            sel[idx] = false;
                        }
                    }
                }
                else{
                    let p = Point_{x:point.x+1,y:point.y};
                    let id = search_id(c, p);
                    if id > -1{
                        sel[id as usize] = false;
                    }

                    let p = Point_{x:point.x,y:point.y-1};
                    let id = search_id(c, p);
                    if id > -1{
                        sel[id as usize] = false;
                    }
                }
            }
            let count_true = sel.iter().filter(|&x| *x).count();
        
            if count_true == 0{
                
                cand_conts.push(c.to_vec());
                println!("{} is cand",k);
                continue;
            }

            if count_true as f64 >= far_ratio * c.len() as f64 {
                tgt_conts.push(c.to_vec());
                
                println!("{} is target",k);
                continue;
            }
           
       
            let mut seeds = vec![];
            for (i,v) in sel.iter().enumerate() {
                if *v{
                    seeds.push(i);

                }
            }

            let black_segments = flood_fill_segment(&c,&dists,seeds,*connect_len);
           
            println!("{} black_segments: {:?}",k,black_segments);

            let mut rest_sel = vec![1;c.len()];
            for psel in black_segments{
                let mut p = Vec::new();
                for i in psel.iter(){
                    p.push(c[*i]);
                    rest_sel[*i] = 0;
                }
                if has_inside_mask && outside_all(&p,&outside_mask){
                    continue;
                }
                println!("{} black_segments is target",k);
                tgt_conts.push(p);
            }
            let mut rest_c = Vec::new();
            let mut rest_dists = Vec::new();
            for i in rest_sel.iter(){
                if *i == 1{
                    rest_c.push(c[*i]);
                    rest_dists.push(dists[*i]);
                }
            }
            if rest_c.len() == 0{
                continue;
            }
            let rest_dist_max = rest_dists.iter().max().unwrap();
            if *rest_dist_max <= *near_dist_th{
                println!("{} rest_dist_max is too small",k);
                continue;
            }
            cand_conts.push(rest_c);
        }
        
        (cand_conts,tgt_conts)
}
fn outside_all(p:&Vec<Point_<i32>>,outside:&Mat)->bool{
    println!("test outside !");
    let mut count = 0;
    for point in p.iter(){
        let v = outside.at_2d::<u8>(point.y,point.x).unwrap();
        if *v == 1{
            count += 1;
        }
    }
    if count == p.len(){
        true
    }
    else
    {
        false
    }
}
fn align_contour(crop_distmap:&Mat,crop_img:&Mat,edge_points:Option<Mat>,lr:f64) -> (f32,Vec<Vec<f32>>){
    
    let ret = align_edge::align_edge(crop_img.clone(),&crop_distmap,None,Mat::default(),edge_points,
    String::from("projective"),false,String::from("adam"),lr,1,1,300,
    10,0.9,10,200,0.001,Option::None,
    String::from(" "),"cpu".to_string(),true);
    // let millis = time::Duration::from_millis(10000);
    // thread::sleep(millis);
    let loss = ret.err;
  
    let mut H_21 = ret.H_20;
    
    (loss,H_21)
    
}

fn spilit_conts(conts:&Vec<Vec<Point_<i32>>>,num:usize)->Vec<Vec<Point_<i32>>>{
    let mut spilit_conts = Vec::new();
    for c in conts.iter(){
        let len = c.len();
        let counts = len/num ;
        if len < num{
            spilit_conts.push(c.clone());
        }
        else{
            for i in 0..counts{
                let (left,right) = c.split_at((i+1)*num);
                spilit_conts.push(left.to_vec());
                spilit_conts.push(right.to_vec());
            }
        }
        
    }
    spilit_conts
}
fn align_contours(conts:&Vec<Vec<Point_<i32>>>,tpl_distmap:&Mat,img:&Mat,color_img:&Mat,copper:&Mat,inside_mask:&Option<Mat>,margin:i32)->Vec<Vec<Point_<i32>>>{
    let h = tpl_distmap.rows();
    let w = tpl_distmap.cols();

    // println!("tpl_distmap: {:#?}",tpl_distmap);
    // println!("img: {:#?}",img);

    let mut align_contours = Vec::new();
    let mut mat = Mat::zeros(h, w, CV_8UC4).unwrap().to_mat().unwrap();
    let mut count = 0;
    //let spilit_conts = spilit_conts(conts,500);
    for c in conts.iter() {

        let mut x0 = i32::MAX;
        let mut y0 = i32::MAX;
        let mut x1 = i32::MIN;
        let mut y1 = i32::MIN;
        for point in c.iter(){
            if x0 > point.x {
                x0 = point.x;
            }
            if y0 > point.y {
                y0 = point.y;
            }
            if x1 < point.x {
                x1 = point.x;
            }
            if y1 < point.y {
                y1 = point.y
            }
        }

        x0 = (x0 - margin).max(0);
        y0 = (y0 - margin).max(0);
        x1 = (x1 + margin).min(w-1);
        y1 = (y1 + margin).min(h-1);
        
        let crop_distmap = Mat::roi(&tpl_distmap,Rect_{x:x0,y:y0,width:x1-x0,height:y1-y0}).unwrap();
        let mut moving = Mat::zeros(y1-y0, x1-x0, CV_8UC1).unwrap().to_mat().unwrap();
        let mut edge_point = Mat::zeros(c.len() as i32, 1, CV_32FC2).unwrap().to_mat().unwrap();
        for (k,point) in c.iter().enumerate() {
            // println!("point_x {}",point.x);
            // println!("point_y {}",point.y);
            let value:VecN<f32, 2> = opencv::core::VecN([(point.y -y0 ) as f32,(point.x -x0 )as f32]);
            *edge_point.at_2d_mut::<Vec2f>(k as i32, 0).unwrap() = value;
            *moving.at_2d_mut::<u8>(point.y -y0, point.x -x0).unwrap() = 255;
        }
        let (loss,H21)= align_contour(&crop_distmap,&moving,None,0.0003);
        println!("{} loss is {}",count,loss);
    
        let warped_img = transform_img(&moving, H21.clone(), Size_ { width: moving.cols(), height: moving.rows() },Option::None,Option::None,None);
        
        let mut edge_points = Vector::<Point_<i32>>::new();
        let mut align_edge_points = Vec::new();
        find_non_zero(&warped_img, &mut edge_points).unwrap();
        for point in edge_points.iter() {
            let x = point.x + x0;
            let y: i32 = point.y + y0;
            align_edge_points.push(Point_{x: x, y: y});
            let value = mat.at_2d_mut::<Vec4b>(y, x).unwrap();
            *value.get_mut(0).unwrap() = 255;
            *value.get_mut(1).unwrap() = 0;
            *value.get_mut(2).unwrap() = 0;
            *value.get_mut(3).unwrap() = 255;
        }
        align_contours.push(align_edge_points);
        
        count+=1;
        
    }
    imsave("align.png",&mat,100);
    align_contours
}


fn transform_contour(H:Tensor,c:&Vec<Point_<i32>>,shape:(i32,i32))->Vec<Point_<i32>>{
    let (h,w) = shape;
   
    let mut wrap_contours:Vector<Point_<i32>> = Vector::new();
    let mut c_mat = Mat::zeros(c.len() as i32, 1, CV_32FC2).unwrap().to_mat().unwrap();
    for (k,point) in c.iter().enumerate() {
        // println!("point_x {}",point.x);
        // println!("point_y {}",point.y);
        let value:VecN<f32, 2> = opencv::core::VecN([(point.y ) as f32,(point.x )as f32]);
        *c_mat.at_2d_mut::<Vec2f>(k as i32, 0).unwrap() = value;
    }
    let ptr = H.data_ptr();
    let step = 3 * std::mem::size_of::<f32>();
    let m: Mat = unsafe{Mat::new_rows_cols_with_data(3, 3, CV_32FC1,ptr,step).unwrap()}; 
    
    //println!("m 0,0 is {:#?} ",m.at_row::<f32>(0));
    let mut warp_mat = Mat::default();
    perspective_transform(&c_mat,&mut warp_mat,&m).unwrap();
    //warp_perspective(&c_mat,&mut warp_mat,&m,Size_ { width: 1, height: c.len() as i32 },INTER_NEAREST,BORDER_CONSTANT,Scalar::all(0.0)).unwrap();

    // println!("warp_mat 0,0 is {:#?} ",warp_mat.at_row::<Vec2f>(1));
    // println!("warp_mat 0 is {:#?} ",warp_mat.at_row::<Vec2f>(1));
    for id in 0..warp_mat.rows(){
        let value = *warp_mat.at_2d::<Vec2f>(id, 0).unwrap();
        wrap_contours.push(Point_{x:*value.get(1).unwrap() as i32 , y:*value.get(0).unwrap() as i32 });
    }
    let mut wrap_contours_:Vec<Point_<i32>> = wrap_contours.iter()
        .map(|v|Point_{x:v.x,y:v.y}).collect();
    //println!("wrap_contours_ {:#?}",wrap_contours_);
    for point in wrap_contours_.iter_mut(){
        if point.x < 0 {
            point.x = 0;
        }
        if point.y < 0 {
            point.y = 0;
        }
        if point.x > w-1{
            point.x = w-1;
        }
        if point.y > h-1{
            point.y = h-1;
        }
    }
    wrap_contours_
}
fn objs_update(outs:&mut Vec<obj>,objs:Vec<Vec<Point_<i32>>>, tpl_copper:&Mat, margin:&i32, shape:(i32,i32),id:&mut usize){
    let (h,w) = shape;
    for (k,c) in objs.iter().enumerate(){
        let mut x0 = i32::MAX;
        let mut y0 = i32::MAX;
        let mut x1 = i32::MIN;
        let mut y1 = i32::MIN;
        let mut tp = "convex";
        let mut centroid_x = 0.0;
        let mut centroid_y = 0.0;
        for point in c.iter(){
            if x0 > point.x {
                x0 = point.x;
            }
            if y0 > point.y {
                y0 = point.y;
            }
            if x1 < point.x {
                x1 = point.x;
            }
            if y1 < point.y {
                y1 = point.y
            }
            centroid_x+=point.y as f64;
            centroid_y+=point.x as f64;
            if *tpl_copper.at_2d::<u8>(point.y, point.x).unwrap() == 1{
                tp = "concave";
            }
        }

        x0 = (x0 - margin).max(0);
        y0 = (y0 - margin).max(0);
        x1 = (x1 + margin).min(w);
        y1 = (y1 + margin).min(h);
        let bbox = (x0,y0,x1-x0,y1-y0);
        let mut mask =Mat::zeros(y1-y0, x1-x0, CV_8UC1).unwrap().to_mat().unwrap();
        for point in c.iter(){
            *mask.at_2d_mut::<u8>(point.y-y0, point.x-x0).unwrap() = 1;
        }
        let o = foreign::obj{
            tp:tp,
            area:c.len() as i32,
            centroid:(centroid_x/c.len() as f64,centroid_y/c.len() as f64),
            bbox:bbox,
            mask:mask,
            id:*id,
            level:" ",
            located:" ",
            group:-1
        };
        *id+=1;
        outs.push(o);
    }
}
pub fn detect_deviations<'a>(name:&str,segmap:&Mat,tpl_segmap:&'a Mat,tpl_image:&'a Mat,tpl_distmap:&Mat,img:&Mat,C:&'a Map<String, Value>,mask:&Option<Mat>,zoom:i32)->(Vec<obj<'a>>,HashMap<i32, Children>){
    let cfg = C.get("deviation").unwrap();
    let h = segmap.rows();
    let w = segmap.cols();

    let blank_label = C.get("classes").unwrap().get("blank").unwrap().get("label").unwrap().as_integer().unwrap() as f64;
    let mut valid_mask = Mat::default();
    compare(&segmap, &blank_label, &mut valid_mask, CMP_NE).unwrap();
    if let Some(mat) = mask{
        bitwise_and(&valid_mask.clone(), &mat, &mut valid_mask, &no_array()).unwrap();
    }

    erode(&valid_mask.clone(), &mut valid_mask, &disk(3), Point_{x:-1,y:-1}, 1, BORDER_CONSTANT, morphology_default_border_value().unwrap()).expect("erode failed");
    let mut inside_mask = Mat::default();
    erode(&valid_mask, &mut inside_mask, &disk(cfg.get("border_gap").unwrap().as_integer().unwrap() as i32), Point_{x:-1,y:-1}, 1, BORDER_CONSTANT, morphology_default_border_value().unwrap()).expect("erode failed");

    let copper_label = C.get("classes").unwrap().get("copper").unwrap().get("label").unwrap().as_integer().unwrap() as f64;
    let mut copper = Mat::default();
    compare(&segmap, &copper_label, &mut copper, CMP_EQ).unwrap();
    let mut tpl_copper = Mat::default();
    compare(&tpl_segmap, &copper_label, &mut tpl_copper, CMP_EQ).unwrap();

    let inside_mask = Some(inside_mask);
    let conts = ectract_contours(&copper, &valid_mask, &inside_mask);
    
    println!("ectract_contours {}", conts.len());

    let far_dist_th = utils::get_zoomed_len(cfg.get("coarse_far_dist_th").unwrap(), zoom) as i32;
    let far_ratio = utils::get_zoomed_len_f(cfg.get("coarse_far_ratio").unwrap(), zoom);
    let near_dist_th = utils::get_zoomed_len(cfg.get("coarse_near_dist_th").unwrap(), zoom) as u8;
    let connect_len = utils::get_zoomed_len(cfg.get("connect_len").unwrap(), zoom) as i32;

    let (conts,mut objs) = filter_conts_by_dist(&conts,&tpl_distmap,
        &far_dist_th,&far_ratio,&near_dist_th,&connect_len,&inside_mask);
    
    let margin = utils::get_zoomed_len(cfg.get("align_contour_margin").unwrap(), zoom) as i32;
    let mut outs:Vec<obj> = Vec::new();
    let mut id = 0  as usize;
    let mut objs_mask = Mat::zeros(h, w, CV_8UC1).unwrap().to_mat().unwrap();
   
    objs_update(&mut outs, objs, &tpl_copper, &margin, (h,w),&mut id);

   
    let mut color_img = Mat::default();
    if tpl_image.dims() == 2{
        let code = ColorConversionCodes::COLOR_GRAY2BGR as i32;
        cvt_color(&tpl_image, &mut color_img, code, 0).expect("convert img to color error");
    }  
    
    let strict_dist_th = utils::get_zoomed_len(cfg.get("strict_dist_th").unwrap(), zoom) as i32;
    let strict_ratio = utils::get_zoomed_len_f(cfg.get("strict_ratio").unwrap(), zoom);
    
    let aligned_conts = align_contours(&conts,&tpl_distmap,&img,&color_img,&copper,&inside_mask,margin);
    println!("align lens {}",aligned_conts.len());
    let (conts,mut objs) = filter_conts_by_dist(&aligned_conts,&tpl_distmap,
            &strict_dist_th,&strict_ratio,&(strict_dist_th as u8),&connect_len,&inside_mask);
   
    println!("aligned over");
    for c in conts.iter(){
        for point in c.iter() {
            *objs_mask.at_2d_mut::<u8>(point.y, point.x).unwrap() = 255;
        }
    }
    imsave("A_objs_mask.jpg",&objs_mask,100);
    objs_update(&mut outs, objs, &tpl_copper, &margin, (h,w),&mut id);  

    for i in outs.iter(){
        let mut mask = i.mask.clone();
        for i in 0..mask.rows(){
            for j in 0..mask.cols(){
                if *mask.at_2d::<u8>(i, j).unwrap() == 1{
                    *mask.at_2d_mut::<u8>(i, j).unwrap() = 255;
                }
            }
        }
    } 
   
    let mut cluster_objs = Vec::<&mut obj>::new();
    for (idx,o) in outs.iter_mut().enumerate() {
        let type_ = o.tp;
        cluster_objs.push(o);
    }

    if cluster_objs.len() > 0 {
        let grp_labels = cluster_defects(&cluster_objs, (segmap.rows(),segmap.cols()),cfg.get("cluster").unwrap());
        for (i,obj) in cluster_objs.iter_mut().enumerate() {
            obj.update_group(grp_labels[i]);
        }
    }

 
    let groups = build_group(&outs);
    (outs, groups)

}


