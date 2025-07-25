use cuckoofilter_mmap::{CuckooFilter, FlushMode};
use std::env;
use std::fs;
use std::path::Path;
use rand::{thread_rng, Rng};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <capacity>", args[0]);
        std::process::exit(1);
    }

    let capacity_str = &args[1];
    let capacity: usize = capacity_str.parse()?;

    let path = Path::new("/tmp").join(format!("cuckoo_fpr_{}.db", capacity));
    let path_str = path.to_str().unwrap();

    // Ensure the file doesn't exist from a previous run
    let _ = fs::remove_file(&path);

    let mut filter = CuckooFilter::<String>::builder(capacity).build(&path)?;

    let num_elements_to_insert = capacity; // Insert up to capacity
    let num_elements_to_check = capacity; // Check same number of non-existent elements

    let mut rng = thread_rng();
    let mut inserted_items = Vec::with_capacity(num_elements_to_insert);
    let mut non_existent_items = Vec::with_capacity(num_elements_to_check);

    // Insert unique items
    for i in 0..num_elements_to_insert {
        let item = format!("inserted-item-{}", i);
        if filter.insert(&item)? {
            inserted_items.push(item);
        }
    }

    // Generate non-existent unique items
    for i in 0..num_elements_to_check {
        let item = format!("non-existent-item-{}-rand-{}", i, rng.gen::<u64>());
        non_existent_items.push(item);
    }

    let mut false_positives = 0;
    for item in &non_existent_items {
        if filter.contains(item) {
            false_positives += 1;
        }
    }

    let false_positive_rate = false_positives as f64 / num_elements_to_check as f64;

    println!("Capacity: {}, False Positives: {}, Total Checked: {}, FPR: {:.6}",
             capacity, false_positives, num_elements_to_check, false_positive_rate);

    // Clean up the file
    fs::remove_file(&path)?;

    Ok(())
}
