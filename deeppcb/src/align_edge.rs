use std::collections::HashMap;
use std::time::Instant;
use ndarray::ArrayViewMut;
use tch::*;
use crate::base;

#[derive(Debug)]
pub struct py_params{
    pub tform_tp:String,
    pub sim_no_scale:bool,
    pub lr:f64,
    pub max_iters:i32,
    pub max_patience:i32,
    pub max_dist:i32,
    pub H_scale:[i32;2],
}
impl py_params{
    pub fn clone(&self)-> Self{
        let r = py_params{
            tform_tp:self.tform_tp.clone(),
            sim_no_scale:self.sim_no_scale.clone(),
            max_dist:self.max_dist.clone(),
            lr:self.lr.clone(),
            max_patience:self.max_patience.clone(),
            max_iters:self.max_iters.clone(),
            H_scale:self.H_scale.clone(),
        };
        r
    }
}

impl From<py_params> for HashMap<String,String>{
    fn from(value: py_params) -> Self {
        let mut map = HashMap::new();
        map.insert(String::from("tform_tp"),value.tform_tp);
        map.insert(String::from("sim_no_scale"),value.sim_no_scale.to_string());
        map.insert(String::from("lr"),value.lr.to_string());
        map.insert(String::from("max_iters"),value.max_iters.to_string());
        map.insert(String::from("max_patience"),value.max_patience.to_string());
        map.insert(String::from("max_dist"),value.max_dist.to_string());
        map.insert(String::from("H_scale"),format!("[{},{}]",value.H_scale[0],value.H_scale[1]));
        map
    }
}

#[derive(Debug)]
pub struct align_params{
    pub tform_tp :String,
    pub msg_prefix : String,
    pub optim : String,
    pub lr : f64,
    pub max_patience : i32,
    pub max_iters : i32,
}

#[derive(Debug)]
pub struct ret {
    pub H_21:Vec<Vec<f32>>,
    pub H_20:Vec<Vec<f32>>,
    pub err:f32,
    pub tpl_h:i32,
    pub tpl_w:i32,
    pub params:py_params,
}
impl ret {
    pub fn new()->Self{
        let r = ret{
            H_21:Vec::new(),
            H_20:Vec::new(),
            err:0.0,
            tpl_h:0,
            tpl_w:0,
            params:py_params { tform_tp: String::new(), sim_no_scale: true, lr: 0.0, max_iters: 0, max_patience: 0, max_dist: 0, H_scale: [0,0] },
        };
        r
    }
    pub fn clone(&self) -> Self{
        let r = ret{
            H_21:self.H_21.clone(),
            H_20:self.H_20.clone(),
            err:self.err.clone(),
            tpl_h:self.tpl_h.clone(),
            tpl_w:self.tpl_w.clone(),
            params:self.params.clone(),
        };
        r
    }
}


use opencv::{core::*, imgcodecs::imwrite, imgproc::canny};
use tch::nn::Adam;
use tch::nn::Sgd;
use tch::nn::Optimizer;
use tch::nn::OptimizerConfig;
use torch_sys::C_tensor;
use tch::Device;
use tch::nn;

struct Homography{
    parameters : Tensor,
}
impl Homography{
    fn new(vs:&nn::Path) -> Self {
        let tmp_params = Tensor::eye(3,(tch::Kind::Float, Device::Cpu));
        let parameters = vs.f_var_copy("model_parameters", &tmp_params).unwrap();
        Self{parameters}
        
    }
    fn reset_model(&mut self){
        self.parameters.init(nn::Init::Const(1.0));
    }
    fn forward(&mut self) -> Tensor{
        let a =self.parameters.get(2).get(2);
        let new_tensor =self.parameters.f_div(&a).unwrap();
        new_tensor.unsqueeze(0);
        new_tensor
    }
    fn forward_inverse(&self) -> Tensor{
        self.parameters.inverse()
    }
}

fn learning_rate_schedule(epoch: i32, initial_lr: f64, decay_factor: f64, num_epochs_slow: i32,
     num_epochs_fast: i32) -> f64 {
    if epoch < num_epochs_slow {
        initial_lr
    } else if epoch < num_epochs_slow + num_epochs_fast {
        let slow_lr = initial_lr;
        let fast_lr = initial_lr * decay_factor;
        let progress = (epoch - num_epochs_slow) as f64 / num_epochs_fast as f64;
        slow_lr + progress * (fast_lr - slow_lr)
    } else {
        let lr = initial_lr * decay_factor.powi(num_epochs_fast);
        if lr == 0.0{
            initial_lr
        }
        else{
            lr
        }
    }
}


pub fn align_edge(moving:Mat,fixed:&Mat,init_bbox:Option<[i64;4]>,moving_mask:Mat,edge_points:Option<Mat>,
    tform_tp:String,sim_no_scale:bool,optim:String,mut lr: f64,H_sx:i32,H_sy:i32,
    max_iters:i32,max_patience:i32,lr_gamma:f64, lr_sched_step:i32, max_dist:i32, abs_tol:f32,
    err_th:Option<f64>,msg_prefix:String,dev:String,verbose:bool) -> ret{

    let params = py_params{
        tform_tp:tform_tp,
        sim_no_scale:sim_no_scale,
        lr:lr,
        max_iters:max_iters,
        max_patience:max_patience,
        max_dist:max_dist,
        H_scale:[H_sx,H_sy],
    };

    if verbose{
        //println!("align params:{:#?}",params);
    }

    // let params:Vector<i32> = Vector::new();
    // imwrite("moving.jpg", &moving, &params).unwrap();

    let fixed_h = fixed.rows() as f32;
    let fixed_w = fixed.cols() as f32;
    // println!("fixed h:{}",fixed_h);
    // println!("fixed w:{}",fixed_w);

    let v = vec![[fixed_w/2.0,0.0,fixed_w/2.0], [0.0,fixed_h/2.0,fixed_h/2.0],[0.0,0.0,1.0]];
    let v:Vec<f32> = v
        .iter()
        .flat_map(|array| array.iter())
        .cloned()
        .collect();
    let data = unsafe{
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<f32>())
    };
    let mut K = Tensor::of_data_size(data, &[3,3], tch::Kind::Float);
    //println!("K is ");
    //K.print();
    // println!("invk is ");
    let mut invK = K.inverse();
    //invK.print();

    let mut moving_edge = Mat::default();
    let threshold1 =100.0;
    let threshold2 =200.0;
    let aperture_size = 3;
    let mut edge_points_ = Mat::default();

    if let Some(mat) = edge_points{
        edge_points_ = mat.clone();
    }
    else{
        canny(&moving,&mut moving_edge,threshold1,threshold2,aperture_size,false).expect("get edges error");
        //base::imsave(format!("edge_img_{}_{}.jpg",init_bbox[0],init_bbox[1]).as_str(),&moving_edge,100);
        
        //println!("edge_points is {:?}",edge_points);
        find_non_zero(&moving_edge, &mut edge_points_).unwrap();
    }
    

    let mut xs0 = base::Mat2Tensor(&edge_points_,2);
    //println!("xs0 first {:#?}",xs0.get(0));

    // println!("{:#?}",edge_points_);
    // println!("edge_points :{:#?}", edge_points_.at_2d::<Vec2i>(0,0));
    // println!("edge_points :{:#?}", edge_points_.at_2d::<Vec2i>(1,0));
    // println!("edge_points :{:#?}", edge_points_.at_2d::<Vec2i>(2,0));

    // println!("tensor shape {:#?}",xs0.size());
    // println!("{:#?}",xs0.get(0));
    // println!("{:#?}",xs0.get(1));
    // println!("{:#?}",xs0.get(2));
    //没有处理moving_mask
    if moving_mask.empty() == false {
        panic!("please implement moving_mask");
    }
    let moving_shape = [moving.rows(),moving.cols()];
   
    let init_box = base::get_bbox(&init_bbox, &moving_shape);
    //println!("init_box is {:#?}",init_box);

    let mut H_01 = base::get_bbox_H(&moving_shape,&init_box,&params.tform_tp);
    let mut H_10 = H_01.inverse();
    // H_01.print();
    // H_10.print();
    let xs0_len=xs0.size()[0];
    xs0 = Tensor::hstack(&[xs0,Tensor::ones(&[xs0_len, 1], (tch::Kind::Float, Device::Cpu))]);
    // println!("xs0 0 is {:#?}",xs0.get(0));
    // println!("xs0 1 is {:#?}",xs0.get(1));
    // println!("xs0 shape is {:#?}",xs0.size());
    // println!("H_10 shape is {:#?}",H_10.size());

    let mut xs1 = xs0.matmul(&H_10.t_()); 
    //H_10.t_();//tch 转置会作用到本身
    
    xs1 = xs1.matmul(&invK.t_());
    invK.t_();
    let mut t_xs1 = xs1.t_();
    t_xs1 = t_xs1.unsqueeze(0);
    // println!("t_xs1 0 is {:#?}",t_xs1.get(0).get(0));
    // println!("t_xs1 1 is {:#?}",t_xs1.get(0).get(1));
    // println!("t_xs1 2 is {:#?}",t_xs1.get(0).get(2));
    // println!("t_xs1 size is {:#?}",t_xs1.size());

    let mut t_fixed_distmap = base::Mat2Tensor(&fixed, 0);
    // println!("t_fixed_distmap 0 is {:#?}",t_fixed_distmap.get(0).get(0));
    // println!("t_fixed_distmap 1 is {:#?}",t_fixed_distmap.get(0).get(1));
    // println!("t_fixed_distmap 2 is {:#?}",t_fixed_distmap.get(0).get(2));
    t_fixed_distmap = t_fixed_distmap.unsqueeze(0);

    t_fixed_distmap = t_fixed_distmap.unsqueeze(0);
    
    //println!("t_fixed_distmap size is {:#?}",t_fixed_distmap.size());

    // let model = Homography::new();
    // if params.tform_tp == "similarity" {
    //     println!("similarity..........");
    // } else if params.tform_tp == "affine" {
    //     println!("AffineModel.........");
    // } else if params.tform_tp == "projective" {
    //     println!("projective.........");
    //     let model = Homography::new();
    // } else {
    //     panic!("Invalid tform_tp value");
    //}

    let vs = nn::VarStore::new(Device::Cpu);
    let path = vs.root();
    let mut model = Homography::new(&path);
    let mut optimizer = match optim.as_str() {
        "adam" => {
            //println!("model.parameters(): {:?}", model.parameters);
            Adam::default().build(&vs, lr).unwrap()
        },
        "sgd" =>{
            //println!("model.parameters(): {:?}", model.parameters);
            Sgd::default().build(&vs, lr).unwrap()
        }
        _ => {
            panic!("Invalid optimizer type!");
        }
    };

    let mut best_loss = f32::INFINITY;
    let mut best_model = model.forward().detach();
    let mut best_iter = 0;
    //let mut scheduler = Tensor::exponential()
    let align_start = Instant::now();

    for cur_iter in 0..max_iters {
        optimizer.zero_grad();

        let t_m = &model.parameters.unsqueeze(0);
        // println!("t_m is :");
        // t_m.print();
        let proj_points = t_m.bmm(&t_xs1);
        // println!("proj_points 0 is {:?}", proj_points.get(0).get(0).get(0));
        // println!("proj_points 1 is {:?}", proj_points.get(0).get(0).get(1));
        // println!("proj_points 2 is {:?}", proj_points.get(0).get(0).get(2));
        // println!("proj_points size is {:?}", proj_points.size());

        let proj_us = &proj_points.select(1, 0) / &proj_points.select(1, 2);
        let proj_vs = &proj_points.select(1, 1) / &proj_points.select(1, 2);
        // println!("proj_us 0 is {:#?}",proj_us.get(0).get(0));
        // println!("proj_us 1 is {:#?}",proj_us.get(0).get(1));
        // println!("proj_us size is {:#?}",proj_us.size());
        // println!("proj_vs 0 is {:#?}",proj_vs.get(0).get(0));
        // println!("proj_vs 1 is {:#?}",proj_vs.get(0).get(1));
        // println!("proj_vs size is {:#?}",proj_vs.size());

        let grid = Tensor::stack(&[proj_us, proj_vs], -1).unsqueeze(1);
        // println!("grid size is {:#?}",grid.size());
        // println!("grid 0 is {:#?}",grid.get(0).get(0).get(0).get(0));
        let sampled_values = Tensor::grid_sampler(&t_fixed_distmap,&grid,0,0,true,).squeeze();
        // println!("sampled_values size is {:#?}",sampled_values.size());
        // println!("sampled_values 0 is {:#?}",sampled_values.get(0));
        
        let loss = sampled_values.mean(Kind::Float);
        //println!("loss is {:#?}",loss);

        let cur_loss_p = loss.data_ptr() as *mut f32;
        let cur_loss = unsafe{*cur_loss_p};
        
        if cur_loss == 0.0{
            break;
        }
        //println!("cur_loss is {:#?}",cur_loss);
        if cur_loss.is_nan(){
            println!("cur_loss is nan");
            break;
        }
        if cur_loss < best_loss - abs_tol {
            best_loss = cur_loss;
            best_iter = cur_iter;
            // println!("best_loss is {:#?}",best_loss);
            best_model = t_m.detach();
            if err_th.is_some() && best_loss < err_th.unwrap() as f32{
                break;
            }            
        } else if cur_iter - best_iter > max_patience {
            break;
        }

        loss.backward();
        optimizer.step();

        
        if cur_iter > 0 && cur_iter % lr_sched_step == 0 {
           //lr = lr * lr_gamma.powf(cur_iter as f64);
           lr = learning_rate_schedule(cur_iter,lr,lr_gamma,50,130);
           //lr = lr;
        }

        if cur_iter % 2 == 0 && verbose {
            // println!(
            //     "{}iter: {}/{}, loss: {:.6}/{:.6}, lr: {:.6}", msg_prefix, cur_iter, best_iter, cur_loss, best_loss, lr
            // );
        }

    }
    //println!("best loss {}",best_loss);
    let m = best_model.squeeze();
    // m.print();
    // K.print();
    // invK.print();
    let H_21 = K.matmul(&m).matmul(&invK);
    let H_21_point = H_21.data_ptr() as *mut f32;
    let H_21_array =  unsafe{ArrayViewMut::from_shape_ptr((H_21.size()[0] as usize,H_21.size()[1] as usize), H_21_point).to_owned()};
    //println!("H_21_array is {:#?}",H_21_array);

    let H_20 = H_21.matmul(&H_10);
    let H_20_point = H_20.data_ptr() as *mut f32;
    let H_20_array =  unsafe{ArrayViewMut::from_shape_ptr((H_20.size()[0] as usize,H_20.size()[1] as usize), H_20_point).to_owned()};
    //println!("H_20_array is {:#?}",H_20_array);

    //H_20.print();
    let align_end = Instant::now();
    let align_time = align_end - align_start;
    //println!("torch takes time: {:?}", align_time);
    let a:Vec<Vec<f32>> = H_21_array.rows().into_iter().
    map(|row|row.to_vec()).collect();
    //println!("align H_21 is {:?}",a); 
    let ret = ret{
        H_21:H_21_array.rows().into_iter().
            map(|row|row.to_vec()).collect(),
        H_20:H_20_array.rows().into_iter().
            map(|row|row.to_vec()).collect(),
        err:best_loss,
        tpl_h:fixed.rows() as i32,
        tpl_w:fixed.cols() as i32,
        params:params,
    };
    ret
}

