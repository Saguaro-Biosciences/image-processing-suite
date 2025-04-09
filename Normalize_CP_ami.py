import argparse
import boto3
import pandas as pd
from io import StringIO
import numpy as np
from functools import reduce
from pycytominer import annotate, normalize, feature_select
import pathlib
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os

k = 3
alpha = 2.3538
def double_sigmoid(x):
    return (x/alpha)**k / np.sqrt(1 + (x/alpha)**(2*k))

def read_csv_from_s3(bucket_name, file_key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response['Body'].read().decode('utf-8')

    sample = csv_content[:1024]  # Read a small portion of the file
    dialect = csv.Sniffer().sniff(sample, delimiters=";,") 

    return pd.read_csv(StringIO(csv_content), sep=dialect.delimiter)
    

def concatenate_csv_from_s3(bucket_name, plates, base_folder_path, output_bucket, output_prefix, local_dir="temp_data"):
    """
    Concatenates and merges CSV files from an S3 bucket based on time points and image metadata.
    
    Parameters:
    - bucket_name: Name of the S3 bucket.
    - base_folder_path: Base path to the folder containing the experiment data.
    - output_bucket: S3 bucket where the final output will be stored.
    - output_prefix: Prefix for the output files in S3.
    - local_dir: Local directory for temporary storage.
    """
    s3 = boto3.client('s3')
    os.makedirs(local_dir, exist_ok=True)

    # Process each time point separately
    for plate in plates:
        print(plate)
        Image =read_csv_from_s3(bucket_name,f"{base_folder_path}/{plate}/Image.csv")
        nuclei = read_csv_from_s3(bucket_name, f"{base_folder_path}/{plate}/Nuclei.csv")
        cells = read_csv_from_s3(bucket_name, f"{base_folder_path}/{plate}/Cells.csv")
        cytoplasm = read_csv_from_s3(bucket_name, f"{base_folder_path}/{plate}/Cytoplasm.csv")
        
        if not'Metadata_Site' in nuclei.columns: 
            print("Adding ConcLevel to Image...")
            Image['Metadata_ConcLevel']= 1
        else:
            print("Adding ConcLevel to All...")
            Image['Metadata_ConcLevel']= 1
            nuclei['Metadata_ConcLevel']= 1
            cells['Metadata_ConcLevel']= 1
            cytoplasm['Metadata_ConcLevel']= 1


        if not'Metadata_Site' in nuclei.columns: 
                print("Adding Metadata...")
                nuclei = nuclei.merge(Image[['ImageNumber', 'Metadata_Site', 'Metadata_Well','Metadata_Timepoint','Metadata_Compound','Metadata_ConcLevel']], 
                                                            on='ImageNumber', how='left')
                cells = cells.merge(Image[['ImageNumber', 'Metadata_Site', 'Metadata_Well','Metadata_Timepoint','Metadata_Compound','Metadata_ConcLevel']], 
                                                            on='ImageNumber', how='left')
                cytoplasm = cytoplasm.merge(Image[['ImageNumber', 'Metadata_Site', 'Metadata_Well','Metadata_Timepoint','Metadata_Compound','Metadata_ConcLevel']], 
                                                            on='ImageNumber', how='left')
        
        required_metadata_cols = ['Metadata_Well', 'Metadata_Timepoint', 'Metadata_Compound']
        missing_cols = [col for col in required_metadata_cols if col not in nuclei.columns]
        if missing_cols:
            raise ValueError(f"Missing required metadata columns before groupby: {missing_cols}")
        
        #well level aggreggated, trouble shoout later the pycytomaner function
        cols_to_drop = ['ImageNumber', 'Metadata_Site', 'Metadata_ConcLevel']
        nuclei = nuclei.drop(columns=[col for col in cols_to_drop if col in nuclei.columns])
        cells = cells.drop(['ImageNumber','Metadata_Site','Metadata_ConcLevel'], axis=1)
        cytoplasm = cytoplasm.drop(['ImageNumber','Metadata_Site','Metadata_ConcLevel'], axis=1)
        Image = Image.drop(['ImageNumber'], axis=1)
        Image = Image.drop(columns=[col for col in Image.columns if Image[col].dtype == 'object' and not col.startswith('Metadata')])
        
        
        #well level aggreggated, trouble shoout later the pycytomaner function
        nuclei = nuclei.groupby(['Metadata_Well', 'Metadata_Timepoint','Metadata_Compound'], as_index=False).mean()
        cells = cells.groupby(['Metadata_Well', 'Metadata_Timepoint','Metadata_Compound'], as_index=False).mean()
        cytoplasm =cytoplasm.groupby(['Metadata_Well', 'Metadata_Timepoint','Metadata_Compound'], as_index=False).mean()
        Image = Image.groupby(['Metadata_Well', 'Metadata_Timepoint','Metadata_Compound'], as_index=False).mean()
        
        Image = Image.rename(columns=lambda x: 'Image_' + + x if not x.startswith('Metadata_') else x)
        nuclei = nuclei.rename(columns=lambda x: 'DNA_' + x if not x.startswith('Metadata_') else x)
        cells = cells.rename(columns=lambda x: 'Cell_' + x if not x.startswith('Metadata_') else x)
        cytoplasm = cytoplasm.rename(columns=lambda x: 'Cyto_' + x if not x.startswith('Metadata_') else x)
        df_CP_merged = reduce(lambda left, right: pd.merge(left, right, on=['Metadata_Well', 'Metadata_Timepoint','Metadata_Compound'], how='outer'), [cells, nuclei, Image, cytoplasm])
        del nuclei, cells, cytoplasm, Image
        
        # Remove columns that contain 'ExecutionTime'
        df_CP_merged = df_CP_merged.loc[:, ~df_CP_merged.columns.str.contains('ExecutionTime')]
        df_CP_merged = df_CP_merged.loc[:, ~df_CP_merged.columns.str.contains('ModuleError')]
        df_CP_merged = df_CP_merged.loc[:, ~df_CP_merged.columns.str.contains('URL')]
        #gc.collect()

        df_CP_merged["Metadata_Timepoint"] = plate
        df_CP_merged["Metadata_ConcLevel"] = 1
        features = df_CP_merged.columns[~df_CP_merged.columns.str.contains("Metadata")].to_list()

        normalized_exp = normalize(
            profiles=df_CP_merged,
            features=features,
            samples=f"Metadata_Compound == 'DMSO' and Metadata_Timepoint == '{plate}'",
            method="mad_robustize"
        )
        all_features_cp = normalized_exp.columns[~normalized_exp.columns.str.contains("Metadata")].to_list()
        normalized_exp[all_features_cp] = normalized_exp[all_features_cp].astype(float)
        normalized_exp[all_features_cp] = normalized_exp[all_features_cp].apply(double_sigmoid)
        normalized_exp.loc[:, all_features_cp] = normalized_exp.loc[:, all_features_cp].abs()

        # No need to read normalized_exp again
        csv_buffer = StringIO()
        normalized_exp.to_csv(csv_buffer, index=False)
        output_key = f"{output_prefix}/{plate}/Normalized_features.csv"
        s3.put_object(Bucket=output_bucket, Key=output_key, Body=csv_buffer.getvalue())
        print(f"Saved to S3: s3://{output_bucket}/{output_key}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Concatenate CSV files from S3 for multiple plates.")

    #parser.add_argument("--exp_ids", nargs="+", required=True, help="List of experiment IDs (plate numbers).")
    parser.add_argument("--bucket_name", required=True, help="S3 bucket containing the files.")
    parser.add_argument("--base_folder", required=True, help="Base folder path in S3 where experiment folders are stored.")
    parser.add_argument("--plates", nargs="+", required=True, help="List of plates list to process (prefix as they are from CP Feature extraction).")
    parser.add_argument("--output_bucket", required=True, help="S3 bucket where output files will be saved.")
    parser.add_argument("--output_prefix", required=True, help="Prefix for the output files in S3.")
    parser.add_argument("--local_dir", default="temp_data", help="Local directory for temporary storage.")

    args = parser.parse_args()
    print(f"Processing Plate {args.base_folder}...")
    
    concatenate_csv_from_s3(
        bucket_name=args.bucket_name,  # cellprofiler-resuts
        base_folder_path=args.base_folder, # IRIC/CQDM_CTL_Plate_Validation_202501/Plate_1
        plates= args.plates,
        output_bucket=args.output_bucket, #cellprofiler-resuts
        output_prefix=args.output_prefix, # CQDM/CTL_Plate/Plate_1
        local_dir=args.local_dir # Plate_1
    )

    #for plate in args.exp_ids:
    #    print(f"Processing Plate {plate}...")
    #    concatenate_csv_from_s3(
    #        bucket_name=args.bucket_name,
    #        base_folder_path=f"{args.base_folder}/Plate_{plate}",
    #        output_bucket=args.output_bucket,
    #        output_prefix=f"{args.output_prefix}/{plate}",
    #        local_dir=args.local_dir
    #    )
