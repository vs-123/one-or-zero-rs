#![allow(warnings)]
extern crate image;

use image::error::ImageError;
use image::io::Reader as ImageReader;
use image::DynamicImage;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;

use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifierParameters;
use smartcore::tree::decision_tree_classifier::SplitCriterion;

use smartcore::neighbors::knn_classifier::KNNClassifier;

fn img_to_vec_f64(filepath: String) -> Result<Vec<f64>, ImageError> {
    let img = ImageReader::open(filepath)?.decode()?;

    Ok({
        match img {
            DynamicImage::ImageRgb8(e) => {
                e.into_vec().iter().map(|c| *c as f64).collect::<Vec<f64>>()
            }
            _ => Vec::new(),
        }
    })
}

fn main() {
    let x = vec![
        "training-data/zero1.png",
        "training-data/zero2.png",
        "training-data/zero3.png",
        "training-data/zero4.png",
        "training-data/zero5.png",
        "training-data/one1.png",
        "training-data/one2.png",
        "training-data/one3.png",
        "training-data/one4.png",
        "training-data/one5.png",
    ]
    .iter()
    .map(|f| img_to_vec_f64(f.to_string()).unwrap())
    .collect::<Vec<Vec<f64>>>();

    let y = vec![0., 0., 0., 0., 0., 1., 1., 1., 1., 1.];

    let x = DenseMatrix::from_2d_vec(&x);

    // let predictor = RandomForestClassifier::fit(&x, &y, Default::default()).unwrap();
    let predictor = RandomForestClassifier::fit(&x, &y, Default::default()).unwrap();

    let input = &DenseMatrix::from_2d_vec(&vec![img_to_vec_f64("test.png".to_string()).unwrap()]);

    println!("training done");
    println!(
        "The number on test.png is: {:?}",
        predictor.predict(input).unwrap()[0]
    );
}
