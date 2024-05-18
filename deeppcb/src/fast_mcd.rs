use ndarray::iter::Indices;
use ndarray::{Array2, Array1, Axis, s, ArrayBase, OwnedRepr, Dim, Array, ViewRepr};
use rand::prelude::*;
use linfa_linalg::svd::SVD;
use ndarray::{ArrayView1};
use tch::Tensor;
use std::cmp::Ordering;
use std::cmp::PartialEq;

/// check two values are close in terms of the absolute tolerance
pub fn aclose(test: f64, truth: f64, atol:f64) {
    let dev = (test - truth).abs();
    if dev > atol {
        eprintln!("==== Assetion Failed ====");
        eprintln!("Expected = {}", truth);
        eprintln!("Actual   = {}", test);
        panic!("Too large deviation in absolute tolerance: {}", dev);
    }
}
pub fn pinvh(a: &Array2<f64>, tol: f64) -> Array2<f64> {
    let (u, s, vt) = a.svd(true, true).unwrap();

    let cols = s.dim();
    let mut smat = Array::zeros((cols, cols));
    let tol_smallest = tol * s[0];
    let s_inv = s.mapv(|x| if x > tol_smallest { 1.0 / x } else { 0.0 });
    for i in 0..s_inv.len() {
        smat[[i, i]] = s_inv[i];
    }
    let u = u.unwrap();
    let ut = u.t();
    let vt = vt.unwrap();
    let v = vt.t();

    let vsmat = v.dot(&smat);
    let pinv = vsmat.dot(&ut);
    pinv
}

// Define a struct to hold an index and a value
struct IndexedValue<'a> {
    index: usize,
    value: &'a f64,
}

// Implement the `Ord` trait for `IndexedValue`
impl<'a> Ord for IndexedValue<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.partial_cmp(other.value).unwrap_or(Ordering::Equal)
    }
}

// Implement the `PartialOrd` trait for `IndexedValue`
impl<'a> PartialOrd for IndexedValue<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Implement the `Eq` trait for `IndexedValue`
impl<'a> Eq for IndexedValue<'a> {}

// Implement the `PartialEq` trait for `IndexedValue`
impl<'a> PartialEq for IndexedValue<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

// Define the `argsort` function
pub fn argsort(a: &ArrayView1<f64>) -> Array1<usize> {
    // Create a vector of IndexedValues
    let mut indexed_values: Vec<IndexedValue> = a.iter().enumerate().map(|(i, v)| IndexedValue { index: i, value: v }).collect();
    // Sort the vector by value
    indexed_values.sort();

    // Extract the sorted indices from the vector
    indexed_values.iter().map(|iv| iv.index).collect()
}

pub fn fast_mcd(
    x: Array2<f64>,
    support_fraction: Option<f64>,
    cov_computation_method: fn(&Array2<f64>) -> Array2<f64>,
    random_state: Option<&mut dyn RngCore>,
) ->Result<(Array1<f64>, Array2<f64>, Array1<bool>, Array1<f64>),&'static str> {

    let mut rng = match random_state {
        Some(r) => rand::rngs::StdRng::from_seed(r.gen()),
        None => rand::rngs::StdRng::from_entropy(),
    };

    let (n_samples, n_features) = x.dim();

    let n_support = match support_fraction{
        Some(s) => (s*n_samples as f64) as usize,
        None => ((n_samples + n_features + 1) as f64 / 2.0).ceil() as usize,
    };
    println!("n_samples: {}", n_samples);
    println!("n_features: {}", n_features);
    println!("n_support: {}", n_support);
    //数据量过大，分组分别进行fast_mcd
    if n_samples > 500 && n_features > 1 {
        // 1. Find candidate supports on subsets
        // a. split the set in subsets of size ~ 300
        let n_subsets = n_samples / 300;
        let n_samples_subsets = n_samples / n_subsets;

        println!("n_subsets: {}",n_subsets);
        println!("n_samples_subsets: {}",n_samples_subsets);
        let mut samples_shuffle: Vec<_> = (0..n_samples).collect();
        samples_shuffle.shuffle(&mut rng);
        let h_subset = ((n_samples_subsets as f64) * ((n_support as f64) / (n_samples as f64))).ceil() as usize;
        println!("h_subset: {}",h_subset);
        // b. perform a total of 500 trials
        let n_trials_tot = 500;
        // c. select 10 best (location, covariance) for each subset
        let n_best_sub = 10;
        let n_trials = std::cmp::max(10, n_trials_tot / n_subsets);
        let n_best_tot = n_subsets * n_best_sub;
        // let mut all_best_locations = Array2::<f64>::zeros((n_best_tot, n_features));
        // let mut all_best_covariances = Array3::<f64>::zeros((n_best_tot, n_features, n_features));

        let mut all_best_locations = Vec::new();
        let mut all_best_covariances = Vec::new();

        // If the above is too big. 
        // Let's try with something much small(and less optimal)
        // n_best_tot = 10
        // all_best_covariances = np.zeros((n_best_tot, n_features, n_features))
        // n_best_sub = 2

        for i in 0..n_subsets{
            let low_bound = i * n_samples_subsets;
            let high_bound = low_bound + n_samples_subsets;
            println!("low_bound is {:?}", low_bound);
            println!("high_bound is {:?}", high_bound);
            let current_subset_ : Array2<f64> = Default::default();
            let mut current_subset = current_subset_.to_shape((0,n_features)).unwrap().to_owned();

            for i in low_bound..high_bound {
                let _value = samples_shuffle[i];
                let tmp = x.slice(s![_value,..]).to_shape((1,n_features)).unwrap().to_owned();
                current_subset.append(Axis(0), tmp.view()).unwrap();
                
            }
            println!("current_subset is {:?}", current_subset);

            let (best_locations_sub, best_covariances_sub, _, _) = select_candidates(
                &current_subset,
                h_subset,
                n_trials,
                None,
                n_best_sub,
                2,
                cov_computation_method,
                Some(&mut rng)
            ).unwrap();
            //let subset_slice = s![i * n_best_sub..(i + 1) * n_best_sub, ..];
            for i in best_locations_sub {
                all_best_locations.push(i);
            }
            for i in best_covariances_sub {
                all_best_covariances.push(i);
            }
            
            //all_best_covariances.slice_mut(subset_slice).assign(&best_covariances_sub);
        }

        println!("all_best_locations shape {}",all_best_locations.len());
        println!("all_best_covariances shape {}",all_best_covariances.len());
        // 2. Pool the candidate supports into a merged set
        // (possibly the full dataset)
        println!("get all best locations and covariations");
        let n_samples_merged = std::cmp::min(1500,n_samples);
        let h_merged = ((n_support as f64 / n_samples as f64)*n_samples_merged as f64).ceil() as usize;
        let mut n_best_merged:usize;
        if n_samples >1500{
            n_best_merged = 10;
        }
        else{
            n_best_merged = 1;
        }
        // find the best couples (location, covariance) on the merged set
        let mut indic_selection = (0..n_samples).collect::<Vec<_>>();
        indic_selection.shuffle(&mut rng);

        let selection_info : Array2<f64> = Default::default();
        let mut selection = selection_info.to_shape((0,n_features)).unwrap().to_owned();
        for i in 0..n_samples_merged {
            let _value = indic_selection[i];
            let tmp = x.slice(s![_value,..]).to_shape((1,n_features)).unwrap().to_owned();
            selection.append(Axis(0), tmp.view()).unwrap();
            
        }

        println!("selection is {:?}", selection);
        println!("all_best_locations[0].shape()[0] is {:?}", all_best_locations[0].shape()[0]);
        // let selection_info = s![indic_selection[0]..indic_selection[n_samples_merged],..];
        // let selection = x.slice(selection_info).to_owned();
        let (locations_merged, covariances_merged, supports_merged, d) = select_candidates(
            &selection, 
            h_merged, 
            all_best_locations.len(),
            Some((&all_best_locations,&all_best_covariances)),
            n_best_merged, 
            30, 
            cov_computation_method, 
            Some(&mut rng)
        ).unwrap();

        println!("locations_merged is {:?}", locations_merged);
        println!("covariances_merged is {:?}", covariances_merged);
        // 3. Finally get the overall best (locations, covariance) couple
        if n_samples < 1500{
            //directly get the best couple (location, covariance)
            let location = locations_merged[0].clone();
            let covariance = covariances_merged[1].clone();
            let mut dist = Array1::from_elem(n_samples, 0.0);
            let mut support = Array1::from_elem(n_samples, false);
            for i in 0..n_samples_merged{
                let index = indic_selection[i];
                support[index] = supports_merged[0][index];
                dist[index] = d[0][index];
            }
            // support.slice_mut(s![indic_selection[0]..indic_selection[n_samples_merged]]).assign(&supports_merged[0]);
            // dist.slice_mut(s![indic_selection[0]..indic_selection[n_samples_merged]]).assign(&d[0]);
            
            return Ok((location, covariance, support, dist));
        }
        else{
            println!("locations_merged shape {}",locations_merged.len());
            // select the best couple on the full dataset
            let (locations_full, covariances_full, supports_full, d) =  select_candidates(
                &x, 
                n_support, 
                locations_merged.len(),
                Some((&locations_merged,&covariances_merged)),
                1, 
                30, 
                cov_computation_method, 
                Some(&mut rng)
            ).unwrap();
            let location = locations_full[0].clone();
            let covariance = covariances_full[0].clone();
            let dist = d[0].clone();
            let support = supports_merged[0].clone();
            
            return Ok((location, covariance, support, dist));
        }
    }
    else{
        //don't implement n_smaples < 500
        return Err(" data samples too small");
    }
    
}

fn select_candidates(
    x: &Array2<f64>,
    n_support: usize,
    n_trials: usize,
    estimates_list: Option<(&Vec<Array1<f64>>, &Vec<Array2<f64>>)>,
    select: usize,
    n_iter: usize,
    cov_computation_method: fn(&Array2<f64>) -> Array2<f64>,
    random_state: Option<&mut dyn RngCore>,
) -> Result<(Vec<Array1<f64>>, Vec<Array2<f64>>, Vec<Array1<bool>>,Vec<Array1<f64>>), &'static str>{

    let mut rng = match random_state {
        Some(r) => rand::rngs::StdRng::from_seed(r.gen()),
        None => rand::rngs::StdRng::from_entropy(),
    };
    let (n_samples, n_features) = x.dim();
    println!("candidates n_samples is {:?}", n_samples);
    println!("candidates n_features is {:?}", n_features);
    println!("candidates n_support is {:?}", n_support);
    println!("candidates n_trials is {:?}", n_trials);
    println!("(n_samples + n_features + 1) / 2  is {:?}", (n_samples + n_features + 1) / 2 );
    if n_support <= (n_samples + n_features + 1) / 2 || n_support >= n_samples {
        return Err("Invalid n_support parameter");
    }
    
    //n_trials is a number
    let mut run_from_estimates:bool;
    if let None = estimates_list{
        run_from_estimates = false;
    }
    else{
        run_from_estimates = true;
    }

    let mut all_locs_sub:Vec<_> = Vec::new();
    let mut all_covs_sub:Vec<_> = Vec::new();
    let mut all_dets_sub:Vec<_> = Vec::new();
    let mut all_supports_sub:Vec<_> = Vec::new();
    let mut all_ds_sub:Vec<_> = Vec::new();

    if run_from_estimates == false {
        // perform n_trials computations from random initial supports
        for _ in 0..n_trials {
            let (locs_sub,covs_sub,dets_sub,supports_sub,ds_sub) =
                _c_step(
                    x,
                    n_support,
                    &mut rng,
                    n_iter.try_into().unwrap(),
                    None,
                    cov_computation_method,
                );
            all_locs_sub.push(locs_sub);
            all_covs_sub.push(covs_sub);
            all_dets_sub.push(dets_sub);
            all_supports_sub.push(supports_sub);
            all_ds_sub.push(ds_sub);
        }
    } else {
        // perform computations from every given initial estimates
        for i in 0..n_trials {
            let (estimates_list_0,estimates_list_1) = estimates_list.unwrap();
            let initial_estimates = Some((&estimates_list_0[i],&estimates_list_1[i]));
            let (locs_sub,covs_sub,dets_sub,supports_sub,ds_sub) =
                _c_step(
                    x,
                    n_support,
                    &mut rng,
                    n_iter.try_into().unwrap(),
                    initial_estimates,
                    cov_computation_method,
                );
            all_locs_sub.push(locs_sub);
            all_covs_sub.push(covs_sub);
            all_dets_sub.push(dets_sub);
            all_supports_sub.push(supports_sub);
            all_ds_sub.push(ds_sub);
        }
    }

    // find the `select` best results among the `n_trials` ones
    let mut index_best: Vec<usize> = (0..all_dets_sub.len()).collect();
    index_best.sort_unstable_by(|&a, &b| all_dets_sub[a].partial_cmp(&all_dets_sub[b]).unwrap());

    let mut best_locations = Vec::new();
    let mut best_covariances= Vec::new();
    let mut best_supports= Vec::new();
    let mut best_ds= Vec::new();
    if select == 0{
        return Err("select must be greater than 0");
    } 
    else{
        for i in 0..select{
            best_locations.push(all_locs_sub[index_best[i]].clone());
            best_covariances.push(all_covs_sub[index_best[i]].clone());
            best_supports.push(all_supports_sub[index_best[i]].clone());
            best_ds.push(all_ds_sub[index_best[i]].clone());
        }
    }

    Ok((best_locations, best_covariances, best_supports,best_ds))
}

fn fast_logdet(covariance: &Array2<f64>) -> f64 {
    let rows = covariance.nrows() as i64;
    let cols = covariance.ncols() as i64;
    let ptr = covariance.as_slice().unwrap();
    let tensor_covariance = Tensor::of_slice(ptr).reshape(&[rows, cols]);
    let (sign, natural_log) = tensor_covariance.slogdet();
    let point_na = natural_log.data_ptr() as *mut f64;
    let point_sign = sign.data_ptr() as *mut f64;
    let natural_log_value = unsafe{*point_na};
    let sign_value = unsafe{*point_sign};
    if sign_value > 0.0{
        natural_log_value
    }
    else{
        f64::INFINITY
    }
}

fn _c_step(
    x: &Array2<f64>,
    n_support: usize,
    rng: &mut dyn rand::RngCore,
    remaining_iterations: i32,
    initial_estimates: Option<(&Array1<f64>, &Array2<f64>)>,
    cov_computation_method: fn(&Array2<f64>) -> Array2<f64>,
)  -> (Array1<f64>, Array2<f64>, f64, Array1<bool>, Array1<f64>){
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];
    let mut dist = Array1::from_elem(n_samples, f64::INFINITY);
    let mut support = Array1::from_elem(n_samples, false);
    if initial_estimates.is_none() {
        // Compute initial robust estimates from a random subset
        let mut indices = (0..n_samples).collect::<Vec<_>>();
        indices.shuffle(rng);
        
        for i in indices {
            support[i] = true;
        }
    }
    else{
        // get initial robust estimates from the function parameters
        let (location,covariance) = initial_estimates.unwrap();
        // run a special iteration for that case (to get an initial support)
        let precision = pinvh(&covariance,1e-12);
        let x_centered = x - &location.broadcast((n_samples, n_features)).unwrap();
        let dist = (&x_centered.dot(&precision) * &x_centered).sum_axis(Axis(1));
        // compute new estimates
        let argsort_dist = argsort(&dist.view());
        for i in argsort_dist.slice(s![0..n_support]){
            support[*i] =true;
        }
    }
    let x_support_:Array2<f64> = Default::default();
    let cols = x.shape()[1];
    let mut x_support = x_support_.to_shape((0,cols)).unwrap().to_owned();
    for (i,value) in support.iter().enumerate(){
        if value == &true{
            x_support.push(Axis(0), x.slice(s![i,..])).unwrap();
        }
    }
   
    let mut location = x_support.mean_axis(Axis(0));
    //println!("x_support is {:?}",x_support);
    let mut covariance = cov_computation_method(&x_support);

    // Iterative procedure for Minimum Covariance Determinant computation
    let mut det = fast_logdet(&covariance);
    // If the data already has singular covariance, calculate the precision,
    // as the loop below will not be entered.
    let mut precision = pinvh(&covariance,1e-12);

    let mut remaining_iterations = remaining_iterations;

    let mut previous_location = location.to_owned();
    let mut previous_covariance = covariance.to_owned();
    let mut previous_det = f64::INFINITY;
    let mut previous_support = support.to_owned();
    while det < previous_det && remaining_iterations > 0 && !det.is_infinite() {
        // Save old estimates values
        previous_location = location.to_owned();
        previous_covariance = covariance.to_owned();
        previous_det = det;
        previous_support = support.to_owned();
        // Compute a new support from the full data set mahalanobis distances
        precision = pinvh(&covariance,1e-12);
        let x_centered = x - &location.unwrap().broadcast((n_samples, n_features)).unwrap();
        let dist = (&x_centered.dot(&precision) * &x_centered).sum_axis(Axis(1));
        // Compute new estimates
        support.fill(false);
        let indices_dist = argsort(&dist.view());
        for i in indices_dist{
            support[i] = true;
        }
        let X_support_:Array2<f64> = Default::default();
        let mut X_support = X_support_.to_shape((0,cols)).unwrap().to_owned();
        for (i,value) in support.iter().enumerate(){
            if value == &true{
                X_support.push(Axis(0), x.slice(s![i,..])).unwrap();
            }
        }
        location = x_support.mean_axis(Axis(0));
        covariance = cov_computation_method(&x_support);
        det = fast_logdet(&covariance);
        remaining_iterations-=1;
    }
    let previous_dist = dist.clone();
    let tmp_center = x-location.as_ref().unwrap();
    dist = (&(tmp_center).dot(&precision) * &(tmp_center)).sum_axis(Axis(1));
    //Check if best fit already found (det => 0, logdet => -inf)
    if det.is_infinite(){
        return (previous_location.unwrap(),previous_covariance,
        previous_det,previous_support,previous_dist);
    }
    if det > previous_det {
        return (location.unwrap(),covariance,det,support,dist);
    }
    if remaining_iterations == 0{
        return (location.unwrap(),covariance,det,support,dist);
    }
    aclose(det,previous_det,1.0e-8);
    return (location.unwrap(),covariance,det,support,dist);
    
}