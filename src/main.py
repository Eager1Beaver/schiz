from logger import logger
from omegaconf import OmegaConf
from utils.eda import SchizophreniaEDA
from utils.dataset_preparation import prepare_dataset
from utils.preprocessing import preprocess_images
from utils.augment_images import augment_images
from utils.feature_extractor import feature_extraction_pipeline
from utils.classifier import train_and_evaluate
from utils.plot_svm_metrics import SVMVisualizer


def main():
    logger.info("The program started...")

    # Load configuration from a config file
    config_data = OmegaConf.load('src/config/config.yaml')

    # Accessing parameters
    RAW_PT_DATA_DIR = config_data.base.raw_pt_data_dir
    RAW_NII_DATA_DIR = config_data.base.raw_nii_data_dir

    CLINICAL_DATA_DIR = config_data.eda.clinical_data_dir
    IS_PERFORM_EDA = config_data.eda.is_perform_eda
    OUTPUT_PATH_EDA = config_data.eda.output_path

    IS_PERFORM_DATASET_PREPARATION = config_data.dataset_preparation.is_perform_dataset_preparation
    TRAIN_SET_DIR = config_data.dataset_preparation.train_set_dir
    TEST_SET_DIR = config_data.dataset_preparation.test_set_dir
    TRAIN_SET_RATIO = config_data.dataset_preparation.train_set_ratio  # 0.80: 50 - train, 12 - test
    IS_NORMALIZE_WHEN_PREPARING = config_data.dataset_preparation.is_normalize_when_preparing
    NORMALIZATION_METHOD = config_data.dataset_preparation.normalization_method  # min-max

    IS_PERFORM_PREPROCESSING = config_data.preprocessing_params.is_perform_preprocessing
    PREPROCESSED_DATA_DIR = config_data.preprocessing_params.preprocessed_data_dir
    IS_NORMALIZE_WHEN_PREPROCESS = config_data.preprocessing_params.is_normalize_when_preprocess
    IS_BRAIN_EXTRACTION = config_data.preprocessing_params.is_brain_exaction
    IS_CROP = config_data.preprocessing_params.is_crop
    IS_SMOOTH = config_data.preprocessing_params.is_smooth
    IS_RE_NORMALIZE_AFTER_SMOOTH = config_data.preprocessing_params.is_re_normalize_after_smooth
    IS_PREPROCESS_TEST_SET = config_data.preprocessing_params.is_preprocess_test_set

    VOXEL_SIZE = tuple(config_data.preprocessing_params.voxel_size)
    MIN_VAL = config_data.preprocessing_params.min_val
    MAX_VAL = config_data.preprocessing_params.max_val
    MODALITY = config_data.preprocessing_params.modality
    VERBOSE = config_data.preprocessing_params.verbose
    SIGMA = config_data.preprocessing_params.sigma
    ORDER = config_data.preprocessing_params.order
    MODE = config_data.preprocessing_params.mode
    CVAL = config_data.preprocessing_params.cval
    TRUNCATE = config_data.preprocessing_params.truncate

    IS_PERFORM_AUGMENTATION = config_data.augmentation_params.is_perform_augmentation
    AUGMENTED_DATA_DIR = config_data.augmentation_params.augmented_data_dir
    HOW_MANY_AUGMENTATIONS = config_data.augmentation_params.how_many_augmentations
    IS_TRANSLATION = config_data.augmentation_params.is_translation
    TRANSLATION_SHIFT = config_data.augmentation_params.translation_shift
    IS_ROTATION = config_data.augmentation_params.is_rotation
    ROTATION_ANGLE = config_data.augmentation_params.rotation_angle
    IS_GAUSSIAN_NOISE = config_data.augmentation_params.is_gaussian_noise
    GAUSSIAN_NOISE_MEAN = config_data.augmentation_params.gaussian_noise_mean
    GAUSSIAN_NOISE_STD = config_data.augmentation_params.gaussian_noise_std

    IS_PERFORM_FEATURE_EXTRACTION = config_data.feature_extraction_params.is_perform_feature_extraction
    BASE_MODEL_NAME = config_data.resnet_params.base_model_name
    BASE_MODEL_WEIGHTS = config_data.resnet_params.base_model_weights
    INPUT_CHANNELS = config_data.resnet_params.input_channels

    LABELS_DIR = config_data.eda.clinical_data_dir
    EXTRACTED_FEATURES_DIR = config_data.feature_extraction_params.extracted_features_dir
    TARGET_SHAPE = tuple(config_data.feature_extraction_params.target_shape)
    BATCH_SIZE = config_data.feature_extraction_params.batch_size

    IS_PERFORM_CLASSIFICATION = config_data.svc_params.is_perform_classification
    SAVE_CLF_DIR = config_data.svc_params.save_clf_dir
    RESULTS_OUTPUT_DIR = config_data.svc_params.results_output_dir
    KERNEL = config_data.svc_params.kernel
    C = config_data.svc_params.C
    GAMMA = config_data.svc_params.gamma

    IS_PERFORM_PLOTTING = config_data.plot_params.is_perform_plotting
    FIGS_OUTPUT_DIR = config_data.plot_params.figs_output_dir

    # Step 0: Explanatory data analysis
    """
    Performs an explanatory data analysis on the raw images (data/raw_pt)
    and corresponding clinical data (data/clinical_data.csv).
    This step is optional and can be skipped if not necessary.
    """
    if IS_PERFORM_EDA:
        eda = SchizophreniaEDA(path_to_clinical_data=CLINICAL_DATA_DIR, 
                               path_to_raw_images=RAW_PT_DATA_DIR, 
                               output_path=OUTPUT_PATH_EDA)
        eda.plot_age_distribution()
        eda.plot_gender_distribution()
        eda.save_age_statistics()
        eda.save_raw_images_metrics()
        logger.info("EDA completed.")
    else:
        logger.info("EDA was skipped.")

    # Step 1: Select the data (train, test)
    """
    Randomly divides (ratio is user defined) images located in /data/raw_nii,
    Creates 2 folders: data/raw_nii/train_set, data/raw_nii/test_set,
    Copy-pastes previously divided images into these folders (e.g. 50 images into train, 12 images into test),
    Images are saved in a PyTorch format - .pt
    Original images are kept intact in data/raw_nii.

    For smoother workflow, before copying the images, 
    it resamples and normalizes them (both train and test sets).
    """
    if IS_PERFORM_DATASET_PREPARATION:
        prepare_dataset(train_ratio=TRAIN_SET_RATIO,
                        raw_nii_dir=RAW_NII_DATA_DIR,
                        train_set_output_dir=TRAIN_SET_DIR,
                        test_set_output_dir=TEST_SET_DIR,
                        resampling_voxel_size=VOXEL_SIZE,
                        normalize=IS_NORMALIZE_WHEN_PREPARING,
                        norm_method=NORMALIZATION_METHOD,
                        min_max_min_val=MIN_VAL,
                        min_max_max_val=MAX_VAL)
        logger.info("Dataset preparation completed.")
    else:
        logger.info("Dataset preparation was skipped.")

    # Step 2: Preprocessing
    """
    Preprocesses images located in data/raw_nii/train_set,
    Resamples to a common voxel size (if not done in the previous step),
    Normalizes intensity values (if not done in the previous step),
    Extracts brain regions using a custom algorithm (user-defined option),
    Crops the image (user-defined option),
    Smooths the image (user-defined option),
    Saves preprocessed images in data/raw_nii/train_set/preprocessed.

    Test images located in data/raw_nii/test_set 
    may also be resampled and normalized for consistency (user-defined option),
    Saves preprocessed test images in data/raw_nii/test_set/preprocessed.
    """
    if IS_PERFORM_PREPROCESSING:
        preprocess_images(
            raw_nii_dir=RAW_NII_DATA_DIR,
            train_set_dir=TRAIN_SET_DIR,
            test_set_dir=TEST_SET_DIR,
            is_normalize=IS_NORMALIZE_WHEN_PREPROCESS,
            norm_method=NORMALIZATION_METHOD,
            min_max_min_val=MIN_VAL,
            min_max_max_val=MAX_VAL,
            is_brain_extraction=IS_BRAIN_EXTRACTION,
            brain_extraction_modality=MODALITY,
            brain_extraction_verbose=VERBOSE,
            is_crop=IS_CROP,
            is_smooth=IS_SMOOTH,
            smooth_sigma=SIGMA,
            smooth_order=ORDER,
            smooth_mode=MODE,
            smooth_cval=CVAL,
            smooth_truncate=TRUNCATE,
            is_re_normalize_after_smooth=IS_RE_NORMALIZE_AFTER_SMOOTH,
            is_preprocess_test_set=IS_PREPROCESS_TEST_SET,
            output_dir=PREPROCESSED_DATA_DIR,
        )
        logger.info("Preprocessing completed.")
    else:
        logger.info("Preprocessing was skipped.")

    # Step 3: Augmentation
    """
    Applies specified augmentation type (translation, rotation, gaussian_noise, or combined)
    to the preprocessed train set images located in data/raw_nii/train_set/preprocessed,
    Saves augmented images in data/raw_nii/train_set/preprocessed/augmented.
    """
    if IS_PERFORM_AUGMENTATION:
        # Construct augmentation list based on config
        augmentations = []

        if IS_TRANSLATION:
            augmentations.append(("translation", {"shift": TRANSLATION_SHIFT}))

        if IS_ROTATION:
            augmentations.append(("rotation", {"angle": ROTATION_ANGLE}))

        if IS_GAUSSIAN_NOISE:
            augmentations.append(("gaussian_noise", {
                "mean": GAUSSIAN_NOISE_MEAN,
                "std": GAUSSIAN_NOISE_STD
            }))

        augment_images(augmentations=augmentations,
                       num_augmentations=HOW_MANY_AUGMENTATIONS,
                       raw_nii_dir=RAW_NII_DATA_DIR,
                       train_set_dir=TRAIN_SET_DIR,
                       preprocessed_train_set_dir=PREPROCESSED_DATA_DIR,
                       output_dir=AUGMENTED_DATA_DIR)
        logger.info("Augmentation completed.")
    else:
        logger.info("Augmentation was skipped.")

    # Step 4: Feature extraction stage (ResNet)
    """
    Extracts features from both preprocessed and augmented images of the train set. 
    These images are located in 
    data/raw_nii/train_set/preprocessed and data/raw_nii/train_set/preprocessed/augmented
    Saves the extracted features in data/raw_nii/train_set/extracted_features.
    """
    if IS_PERFORM_FEATURE_EXTRACTION:
        feature_extraction_pipeline(
            raw_nii_dir=RAW_NII_DATA_DIR,
            train_set_dir=TRAIN_SET_DIR,
            test_set_dir=TEST_SET_DIR,
            preprocessed_set_dir=PREPROCESSED_DATA_DIR,
            labels_dir=LABELS_DIR,
            extracted_features_dir=EXTRACTED_FEATURES_DIR,
            target_shape=TARGET_SHAPE,
            batch_size=BATCH_SIZE,
            feature_extraction_model=BASE_MODEL_NAME,
            base_model_weights=BASE_MODEL_WEIGHTS,
            input_channels=INPUT_CHANNELS)
        logger.info("Feature extraction completed.")
    else:
        logger.info("Feature extraction was skipped.")

    # Step 5: Model training and validation (SVC)
    """
    Trains a classifier using the extracted features located in 
    data/raw_nii/train_set/extracted_features,
    Validates the trained classifier using the images of the test set located in 
    data/raw_nii/test_set/preprocessed,
    Saves the trained model in src/models.
    Saves the validation results in src/models/results.
    """
    if IS_PERFORM_CLASSIFICATION:
        train_and_evaluate(extracted_features_dir=EXTRACTED_FEATURES_DIR,
                           dir_to_save_clf=SAVE_CLF_DIR,
                           results_output_dir=RESULTS_OUTPUT_DIR,
                           clf_kernel=KERNEL,
                           clf_c_value=C,
                           clf_gamma=GAMMA)
        logger.info("Classifier training and validation completed.")
    else:
        logger.info("Classifier training and validation was skipped.")

    # Step 6: Plot model (SVC) metrics
    """
    Plots model metrics (accuracy, precision, recall, f1-score) on the test set.
    The results are loaded from src/models/results.
    Plots are saved in paper/figs.
    """
    if IS_PERFORM_PLOTTING:
        svm_visualizer = SVMVisualizer(results_json_path=RESULTS_OUTPUT_DIR,
                                       save_figs_path=FIGS_OUTPUT_DIR)
        svm_visualizer.plot_confusion_matrix()
        svm_visualizer.plot_roc_curve()
        logger.info("Classifier metrics have been plotted.")
    else:
        logger.info("Plotting was skipped.")

    logger.info("The program completed successfully.")


if __name__ == "__main__":
    main()
