#import os
import pandas as pd

def main(path):

    df = pd.read_csv(path)

    # Get the file_path of the scan with highest SNR value
    max_snr_scan = df.loc[df['snr'].idxmax()]['file_path']
    print(f"Scan with highest SNR value: {max_snr_scan}")
    # data/schizconnect_COBRE_images_22613/COBRE/sub-A00014839/ses-20090101/anat/sub-A00014839_ses-20090101_acq-mprage_run-02_T1w.nii.gz

    # Print other metrics for this scan
    print(df.loc[df['file_path'] == max_snr_scan])
    #file_path        snr        cnr  relative_psnr  relative_rmse
    #778  data/schizconnect_COBRE_images_22613/COBRE/sub...  13.081738  12.324631      20.805514     875.427017


if __name__ == "__main__":
    quality_check_file = 'tools/initial_quality_check_results/quality_check_all_results_COBRE.csv'

    main(quality_check_file)