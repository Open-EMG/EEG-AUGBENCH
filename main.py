import argparse
# from define_compute import define_model
# from augmentation import apply_augmentation
from run_feature_extraction import feature_extraction
from run_classification import classification
from run_data_augmentation import data_augmentation

def main(args):
    print("=== Full Pipeline: Data Augmentation Benchmark for Epilepsy Detection ===")
    print(f"Augmentation method: {args.aug_method}")
    print(f"Augmentation ratio: {args.aug_ratio}")
    print(f"Classification method: {args.cls_method}")
    print(f"Sampling rate: {args.fs}")
    print("Applying data augmentation...")

    aug_method = args.aug_method
    aug_ratio = args.aug_ratio
    input_raw_data_root = args.input_dir    
    augmented_root = args.output_augmented_dir
    features_root = args.output_feature_dir
    result_root = args.output_result_dir
    cls_method = args.cls_method
    fs = args.fs
    # mode = args.mode
    if aug_method not in ['no']:
        data_augmentation(aug_method, aug_ratio, input_raw_data_root, augmented_root)
    feature_extraction(aug_method, aug_ratio, fs, input_raw_data_root, augmented_root, features_root)
    classification(aug_method, aug_ratio, cls_method, features_root, result_root)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a full pipeline for comparing data augmentation strategies in epilepsy detection tasks."
    )
    parser.add_argument("--aug_ratio", type=int, default=10,
                            help="Data augmentation ratio")
    
    parser.add_argument("--aug_method", type=str, default="no",
                        choices=['no', 'jitter', 'scaling', 'permutation', 'magwarp', 'timewarp', 'windowslice',  
                                 'windowwarp', 'rgw', 'rgws', 'scaling_multi', 'windowwarp_multi'],
                        help="Data augmentation method to apply")
    parser.add_argument("--fs", type=int, default=256,
                        help="Sampling rate of the input data")
    parser.add_argument("--cls_method", type=str, default="rfc",
                        choices=['knn', 'LRC', 'RFC', 'DTC', 'AdaBoost'],
                        help="Classification method to use")
    parser.add_argument("--input_dir", type=str, default="./data/chb-mit/",
                        help="Directory containing input data")
    parser.add_argument("--output_augmented_dir", type=str, default="./output/chb-mit/augmented/",
                        help="Directory to save augmented data")
    parser.add_argument("--output_feature_dir", type=str, default="./output/chb-mit/features/",
                        help="Directory to save extracted features")
    parser.add_argument("--output_result_dir", type=str, default="./output/chb-mit/results/",
                        help="Directory to save classfication results")
    args = parser.parse_args()
    main(args)
