import os
import pandas as pd

def encode_diagnosis(input_path, output_path):
    """
    Reads a .tsv file and encodes diagnosis
    """
    df = pd.read_csv(input_path, sep='\t')

    # Create the 'dx_encoded' column based on the values in the 'dx' column 
    # Here we do not differentiate between diagnoses of schizophrenic specter (they all are labeled as 1)
    df['dx_encoded'] = df['dx'].apply(lambda x: 0 if x == 'No_Known_Disorder' else 1) 

    # Save the updated DataFrame as a new .csv file
    df_updated = df.copy()
    df_updated.to_csv(output_path, index=False)

def main(): # TODO: upscale (automate) the script in case of more datasets
    tsv_name = "participants.tsv"
    csv_name = "participants.csv"

    # Get the current working directory of the script 
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the base .tsv file of the COBRE dataset
    COBRE_input_path = os.path.join(current_dir, '..', 'data', 'schizconnect_COBRE_images_22613/COBRE', tsv_name)

    # Path to save the .csv with encoded diagnosis
    COBRE_output_path = os.path.join(current_dir, '..', 'data', 'schizconnect_COBRE_images_22613/COBRE', csv_name)

    # Same for the MCICShare dataset
    MCICShare_input_path = os.path.join(current_dir, '..', 'data', 'schizconnect_MCICShare_images_22613/MCICShare', tsv_name)
    MCICShare_output_path = os.path.join(current_dir, '..', 'data', 'schizconnect_MCICShare_images_22613/MCICShare', csv_name)

    # Encode COBRE
    encode_diagnosis(COBRE_input_path, COBRE_output_path)

    # Encode MCICShare
    encode_diagnosis(MCICShare_input_path, MCICShare_output_path)

if __name__ == '__main__':
    main()    