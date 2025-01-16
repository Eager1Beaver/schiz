import os
import pandas as pd

def encode_diagnosis(input_path: str, output_path: str) -> None:
    """
    Reads a .tsv file, encodes diagnosis and saves un updated file as a CSV

    Args:
    input_path: str: path to the input.tsv file
    output_path: str: path to the output.csv file

    Returns:
    None: saves the updated DataFrame as a new.csv file
    """ 
    if not os.path.exists(input_path): # Check if the input file exists
        raise FileNotFoundError(f"The input file at {input_path} does not exist.")
    
    # Read the input.tsv file into a DataFrame
    try: 
        df = pd.read_csv(input_path, sep='\t') 
    except Exception as e: 
        raise Exception(f"Error reading the input file: {e}")

    # Create the 'dx_encoded' column based on the values in the 'dx' column 
    # Here we do not differentiate between diagnoses of schizophrenic specter (they all are labeled as 1)

    # Check if 'dx' column exists in the DataFrame 
    if 'dx' not in df.columns: 
        raise ValueError("'dx' column (diagnosis) is not present in the input file.")
    
    df['dx_encoded'] = df['dx'].apply(lambda x: 0 if x == 'No_Known_Disorder' else 1) 

    # Save the updated DataFrame as a new .csv file
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        raise Exception(f"Error writing the output file: {e}")

def main(datasets_names):
    tsv_name = "participants.tsv"
    csv_name = "participants.csv"

    # Get the current working directory of the script 
    current_dir = os.path.dirname(os.path.abspath(__file__))

    for dataset_name in datasets_names:
        dataset_full_name = "schizconnect_" + dataset_name + "_images" + "_22613"
        path_to_load_tsv = os.path.join(current_dir, '..', 'data', dataset_full_name, dataset_name , tsv_name)
        path_to_save_csv = os.path.join(current_dir, '..', 'data', dataset_full_name, dataset_name , csv_name)
        encode_diagnosis(path_to_load_tsv, path_to_save_csv)

if __name__ == '__main__':
    datasets_names = ['COBRE', 'MCICShare']
    main(datasets_names)    