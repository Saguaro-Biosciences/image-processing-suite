import argparse
import boto3
import pandas as pd
from io import StringIO
import numpy as np
from functools import reduce
from pycytominer import annotate, normalize
import csv

def read_csv_from_s3(bucket_name, file_key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response['Body'].read().decode('utf-8')

    sample = csv_content[:1024]  # Read a small portion of the file
    dialect = csv.Sniffer().sniff(sample, delimiters=";,") # Detects the value fo separate the csv with

    return pd.read_csv(StringIO(csv_content), sep=dialect.delimiter)
    

def concatenate_csv_from_s3(bucket_name, plates,times, base_folder_path, output_bucket, DMSO,output_prefix,well_agg_func):
    """
    Concatenates and merges CSV files from an S3 bucket based on time points and image metadata.
    
    Parameters:
    - bucket_name: Name of the S3 bucket.
    - base_folder_path: Base path to the folder containing the experiment data.
    - output_bucket: S3 bucket where the final output will be stored.
    - output_prefix: Prefix for the output files in S3.
    """
    s3 = boto3.client('s3')

    # Process each time/plate point separately
    for plate in plates:
        print(f"Processing plate ID: {plate}")
        filtered_plateMap =read_csv_from_s3(bucket_name, f"{base_folder_path}/{plate}_PlateMap.csv")
        for time in times:
            print(f"Processing timepoint: {time}")
            #Assumming all tables have Metadata_Well, other colum not Well will be disregarded. Well is the only needed
            table_info = {
                'Image': 'Image_',
                'Nuclei': 'DNA_',
                'Cells': 'Cell_',
                'Cytoplasm': 'Cyto_'
            }

            drop_substrings = ['ExecutionTime', 'ModuleError', 'URL']
            tables = {}

            # Load and validate each table
            for name in table_info:
                df = read_csv_from_s3(bucket_name, f"{base_folder_path}/{plate}/{time}/{name}.csv")
                if 'Metadata_Well' not in df.columns:
                    raise ValueError(f"Missing required metadata well columns before groupby in {name}.csv")
                tables[name] = df

            # Clean, rename, and aggregate in-place
            for name, prefix in table_info.items():
                df = tables[name]

                df = df.drop(columns=[
                    col for col in df.columns
                    if col == 'ImageNumber'
                    or (col.startswith('Metadata') and col != 'Metadata_Well')
                    or any(sub in col for sub in drop_substrings)
                ])

                df = df.rename(columns=lambda x: prefix + x if not x.startswith('Metadata_') else x)

                df = df.groupby('Metadata_Well', as_index=False).agg(well_agg_func)

                tables[name] = df  # Overwrite with cleaned version

            # Merge all cleaned tables on Metadata_Well
            df_CP_merged = reduce(lambda left, right: pd.merge(left, right, on='Metadata_Well', how='outer'), tables.values())

            # Cleanup
            del tables

            #Annotating with platemap propagating Compound ConcLevel , each of them harbouring the Metadatata prefix
            df_CP_merged = annotate(df_CP_merged,filtered_plateMap,join_on=[["Metadata_Well"], ["Metadata_Well"]])

            df_CP_merged["Metadata_Timepoint"] = time
            features = df_CP_merged.columns[~df_CP_merged.columns.str.contains("Metadata")].to_list()

            normalized_exp = normalize(
                profiles=df_CP_merged,
                features=features,
                samples=f"Metadata_Compound == '{DMSO}' and Metadata_Timepoint == '{time}'",
                method="mad_robustize"
            )
            
            all_features_cp = normalized_exp.columns[~normalized_exp.columns.str.contains("Metadata")].to_list()
            normalized_exp[all_features_cp] = normalized_exp[all_features_cp].astype(float)

            # No need to read normalized_exp again
            csv_buffer = StringIO()
            normalized_exp.to_csv(csv_buffer, index=False)
            output_key = f"{output_prefix}/{plate}/Normalized_features.csv"
            s3.put_object(Bucket=output_bucket, Key=output_key, Body=csv_buffer.getvalue())
            print(f"Saved to S3: s3://{output_bucket}/{output_key}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Normalizes each timepoint of a project folder, out puts the normalized tables against DMSO.")

    #parser.add_argument("--exp_ids", nargs="+", required=True, help="List of experiment IDs (plate numbers).")
    parser.add_argument("--bucket_name", required=True, help="S3 bucket containing the files.")
    parser.add_argument("--base_folder", required=True, help="Base folder path in S3 where experiment folders are stored.")
    parser.add_argument("--plates", nargs="+", required=True, help="List of plates list to process (prefix Plate/Time/csv).")
    parser.add_argument("--times", nargs="+", required=True, help="List of times to process (prefix Plate/Time/csv).")
    parser.add_argument("--DMSO", default="DMSO", help="DMSO nomenclature use to normalize in the plateMap")
    parser.add_argument("--output_bucket", required=True, help="S3 bucket where output files will be saved.")
    parser.add_argument("--output_prefix", required=True, help="Prefix for the output files in S3.")

    args = parser.parse_args()
    print(f"Processing Plate {args.base_folder}...")
    
    concatenate_csv_from_s3(
        bucket_name=args.bucket_name,  # cellprofiler-resuts
        base_folder_path=args.base_folder, # IRIC/CQDM_CTL_Plate_Validation_202501/Plate_1
        plates= args.plates, # List of plate IDS as in folders
        time=args.times, # List of timepoints as in folders
        DMSO=args.DMSO, # How is DMSO named in the platemap
        output_bucket=args.output_bucket, #cellprofiler-resuts
        output_prefix=args.output_prefix # CQDM/CTL_Plate/Plate_1
    )