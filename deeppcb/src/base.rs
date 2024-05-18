use std::any::Any;
use std::iter::empty;

use ndarray::Array2;
use ndarray::Array3;
use ndarray::ArrayBase;
use ndarray::Axis;
use ndarray::Dim;
use ndarray::OwnedRepr;
use opencv::imgcodecs::IMWRITE_JPEG_QUALITY;
use opencv::imgcodecs::imwrite;
use opencv::prelude::*;
use opencv::core::*;
use opencv::imgproc::*;
use opencv::prelude::*;
use ndarray::ArrayViewMut;
use tch::*;

pub fn get_gray_image(img:&Mat) -> Mat{
    let mut grey_img = Mat::default();
    if img.channels() == 3{
        println!("converting grey image............");
        let code = ColorConversionCodes::COLOR_BGR2GRAY as i32;
        cvt_color(img, &mut grey_img, code, 0).expect("convert img to gray error");
        grey_img
    }
    else{
        println!("failed to convert gray image to gray because it is not RGB");
        grey_img
    }
}


pub fn get_bbox(_box:&Option<[i64; 4]>,shape:&[i32])->[[i64;2];4]{
    let h = shape[0] as i64;
    let w = shape[1] as i64;
    
    if let Some(box_) = _box {
        let mut x0 = 0;
        let mut y0 = 0;
        let mut x1 = 0;
        let mut y1 = 0;
        //box可能是个迭代器?
        if box_.len() == 2{
            x0 = box_[0];
            y0 =box_[1];
            x1 = x0 + w;
            y1 = y0 + h;
        }
        else{
            x0 = box_[0];
            y0 = box_[1];
            x1 = box_[2];
            y1 = box_[3];
        }
    
        if x0==0&&y0==0&&x1==0&&y1==0{
            panic!("get box error ,the x0,x1,y0,y1 all zero!");
        }
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]        
    }
    else{
        return [[0,0],[w,0],[w,h],[0,h]]
    }
    
}

pub fn get_bbox_clockwise(bbox:&[[i64;2];4]) ->f32{
    let [x0, y0] = bbox[0];
    let [x1, y1] = bbox[1];
    let [x2, y2] = bbox[2];
    let verify_matrix = vec![[x0 as f32, y0 as f32, 1.0],
        [x1 as f32, y1 as f32, 1.0],[x2 as f32, y2 as f32, 1.0]];
    let v:Vec<f32> =verify_matrix.iter()
        .flat_map(|array|array.iter()).cloned().collect();
    let data = unsafe{
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<f32>())
    };
    let t = Tensor::of_data_size(data, &[3,3], tch::Kind::Float);
    
    let tt = t.f_det().unwrap();
    let point = tt.data_ptr() as *mut f32;

    unsafe{
        *point
    }
}

pub fn Mat2Tensor(mat:&Mat,cols:i32)->Tensor{
    let mut _mat = Mat::default();
    let h = mat.size().unwrap().height;
    let w = mat.size().unwrap().width;
    let channel = mat.channels();
    let rtype = format!("CV_32FC{}",channel);
    match rtype.as_str() {
        "CV_32FC1" =>mat.convert_to(&mut _mat, CV_32FC1, 1.0, 0.0).unwrap(),
        "CV_32FC2" =>mat.convert_to(&mut _mat, CV_32FC2, 1.0, 0.0).unwrap(),
        "CV_32FC3" =>mat.convert_to(&mut _mat, CV_32FC3, 1.0, 0.0).unwrap(),
        _=> panic!(""),
    };

 
    let data = _mat.data_bytes_mut().unwrap(); 
    let mut tensor = Tensor::new();
    if cols != 0 {
        tensor = tch::Tensor::of_data_size(data, &[h as i64, cols as i64], tch::Kind::Float);
    }
    else{
        tensor = tch::Tensor::of_data_size(data, &[h as i64, w as i64], tch::Kind::Float);
    }
    tensor
}

pub fn get_bbox_H(shape:&[i32;2],bbox1:&[[i64;2];4],tform_tp:&String) -> Tensor{
    let mh =shape[0] as f32;
    let mw =shape[1] as f32;
    let cw = get_bbox_clockwise(bbox1);
    //println!("cw is {:#?}",cw);
    let mut pts0 = [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]];
    if cw > 0.0{
        pts0 = [[0.0,0.0],[mw, 0.0], [mw, mh], [0.0, mh]];
    }
    else{
        pts0 = [[0.0,0.0],[0.0, mh], [mw, mh], [mw, 0.0]];
    }
    
    let mut H_01 = Mat::default();

    let mut Src_Point:Vector<Point2f> = Vector::new();
    Src_Point.push(Point2f::new(bbox1[0][0] as f32, bbox1[0][1] as f32));
    Src_Point.push(Point2f::new(bbox1[1][0] as f32, bbox1[1][1] as f32));
    Src_Point.push(Point2f::new(bbox1[2][0] as f32, bbox1[2][1] as f32));

    let mut Dist_Point:Vector<Point2f> = Vector::new();
    Dist_Point.push(Point2f::new(pts0[0][0], pts0[0][1]));
    Dist_Point.push(Point2f::new(pts0[1][0], pts0[1][1]));
    Dist_Point.push(Point2f::new(pts0[2][0], pts0[2][1]));

    
    let mut H_01_tensor = Tensor::new();
    if tform_tp == &String::from("affine") || tform_tp == &String::from("similarity"){
       H_01 = get_affine_transform(&Src_Point,&Dist_Point).unwrap();
       let H_01_t = Mat2Tensor(&H_01,0);
       let tmp = Tensor::of_slice::<f32>(&[0.0,0.0,1.0]);
       H_01_tensor =Tensor::vstack(&[H_01_t , tmp]);

    }
    else{
        assert_eq!(tform_tp, &String::from("projective"));
        Src_Point.push(Point2f::new(bbox1[3][0] as f32, bbox1[3][1] as f32));
        Dist_Point.push(Point2f::new(pts0[3][0], pts0[3][1]));
       
        H_01 = get_perspective_transform(&Src_Point,&Dist_Point,DECOMP_LU).unwrap();
       
        H_01_tensor = Mat2Tensor(&H_01,0);
    }
    H_01_tensor
}

pub fn scale_H(H:&mut Vec<Vec<f32>>,s_:f64) {
    //H[:2,2]
    let s = s_ as f32;
    for i in 0..2{
        let mut tmp = &mut H[i];
        tmp[2] *= s;
    }

    let mut tmp = &mut H[2];
    tmp[0] /= s;
    tmp[1] /= s;
}

pub fn rescale(img:&Mat,scale:f64,order:i32) -> Mat{
    let mut resize_image = Mat::default();
    let h = img.rows() as f64 * scale;
    let w = img.cols() as f64 * scale;

    resize(img,&mut resize_image,
        Size_ { width: w as i32, height: h as i32 },
        0.0,0.0,order).unwrap();

    resize_image
}
//3dim BGR to RGB
// pub fn imgsave(filename:&String,img:&Mat){
//     let params:Vector<i32> = Vector::new();
//     if find_file(&path, &save_dir)==false{
//         fs::create_dir_all(save_dir).unwrap();
//     }
//     imwrite(&save_path, &resize_image, &params).unwrap();
// }
use std::path::{Path, PathBuf};
pub fn find_file(path:&PathBuf,fid_name:&String)->bool{
    
    let dir = path.as_path().read_dir().unwrap();
    
    for x in dir {
        //println!("{:?}",x);
       if let Ok(path) = x {
           // 文件是否存在
           if path.file_name().eq(fid_name.as_str()) {
                println!("存在文件!");
                return true;
           }
        }
    }
    false
}

pub fn Mat2Array_3(mut img:Mat) ->ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>> {
    let img_ptr = img.data_mut() as *mut u8;
    let row = img.rows() as usize;
    let col = img.cols() as usize;
    let deep = 3 as usize;
    let array =unsafe{ ArrayViewMut::from_shape_ptr((row,col,deep), img_ptr)}.to_owned();
    array
}
pub fn Mat2Array_2(mut img:Mat) ->ArrayBase<OwnedRepr<u8>, Dim<[usize; 2]>> {
    let img_ptr = img.data_mut() as *mut u8;
    let row = img.rows() as usize;
    let col = img.cols() as usize;
    let array =unsafe{ ArrayViewMut::from_shape_ptr((row,col), img_ptr)}.to_owned();
    array
}
pub fn Array2Mat(array:&Array2<u8>) -> Mat{
    let rows = array.shape()[0] as i32;
    let cols = array.shape()[1] as i32;
    let data = array.as_ptr() as *mut std::ffi::c_void;
    let step = cols as usize * std::mem::size_of::<u8>();
    let mat = unsafe{Mat::new_rows_cols_with_data(rows, cols, CV_8UC1,data,step).unwrap()}; 
    mat
}

pub fn Array2Mat_3(mut array:Array3<u8>) ->Mat{
    println!("Array2Mat.. array is {:?}", array);
    
    let rows = array.shape()[0] as i32;
    let cols = array.shape()[1] as i32;
    let channels = 3;
    let data = array.as_mut_ptr();
    println!("mat 00 is {:#?}",unsafe{*data});
    let step = (cols*channels)  as usize * std::mem::size_of::<u8>();
    let mat = unsafe{Mat::new_rows_cols_with_data(rows, cols, CV_8UC3,data as *mut std::ffi::c_void,step).unwrap()}; 
    mat
}

pub fn disk(radius: i32) -> Mat {
    let size = 2 * radius + 1;
    let selem = get_structuring_element(
        MORPH_CROSS,
        (size, size).into(),
        Point_{x:-1,y:-1},     
    )
    .unwrap();
    selem
    
}

pub fn empirical_covariance(X: &Array2<f64>) -> Array2<f64> {
    let n_samples = X.nrows();
    let X_centered = X - &X.mean_axis(Axis(0)).unwrap();
    let cov = X_centered.t().dot(&X_centered) / (n_samples as f64);
    cov
}


pub fn imsave(filename:&str,Mat:&Mat,quality:i32){
    let mut params:Vector<i32> = Vector::new();
    params.push(IMWRITE_JPEG_QUALITY);  
    params.push(quality); 
    imwrite(filename, Mat, &params).unwrap();
}