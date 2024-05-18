use std::fmt::format;

use toml::{Table, Value};
use toml::map::Map;
use std::fmt;
pub fn get_zoomed_len(l: &Value,zoom:i32)->i64{
    //println!("l is {:#?}",l);
    if l.is_integer(){
        let tmp = (l.as_integer().unwrap() as f64/(zoom as f64)).ceil();
        tmp as i64
    }
    else{
        let inter = l.get(fmt::format(format_args!("{}x",zoom))).unwrap().as_integer().unwrap();
        //println!("inter is {:#?}",inter);
        inter
    }
    
}
pub fn get_zoomed_len_f(l: &Value,zoom:i32)->f64{
    //println!("l is {:#?}",l);
    if l.is_float(){
        let tmp = (l.as_float().unwrap()/(zoom as f64));
        tmp 
    }
    else{
        let inter = l.get(fmt::format(format_args!("{}x",zoom))).unwrap().as_float().unwrap();
        //println!("inter is {:#?}",inter);
        inter
    }
    
}
pub fn get_zoomed_area(l: &Value,zoom:i32)->i64{
    //println!("l is {:#?}",l);
    let min_val =1.0;
    if l.is_integer(){
        let tmp = ((l.as_integer().unwrap()/(zoom.pow(2) as i64)) as f64).ceil();
        if tmp > min_val{
            tmp as i64
        }
        else{
            min_val as i64
        }
    }
    else{
        let inter = l.get(fmt::format(format_args!("{}x",zoom))).unwrap().as_integer().unwrap();
        //println!("inter is {:#?}",inter);
        inter
    }
    
}
pub fn update_zoomed_len(cfg:&Value, keys:[&str; 5], zoom:i32) ->Vec<i64>{
    let mut ret_cfg:Vec<i64> = Vec::new();
    
    for i in keys{
        let k:Vec<&str> = i.split(".").collect();
        let update_index = k[k.len()-1];
        let update_value = cfg.get(update_index).unwrap();
        let value = get_zoomed_len(update_value, zoom);
        ret_cfg.push(value);
    }
    ret_cfg
}