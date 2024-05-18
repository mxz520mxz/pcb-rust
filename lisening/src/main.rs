use std::io;
fn main() {    
    let mut index  = 0;
    let stdin = io::stdin();	// 打开标准输入
    for i in 0..10 {
        let mut s = String::new();
        stdin.read_line(&mut s).unwrap();	// 	读取标准输入中的一行
        println!("{}", s.trim());	

        index += 1;
    }    
}