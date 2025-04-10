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

def concatenate_csv_from_s3(bucket_name, base_folder_path, output_bucket, output_prefix, local_dir="temp_data"):
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

    datasets = ["nuclei", "cytoplasm", "image", "cells"]
    times = ["12h", "18h", "24h_2", "48h_2", "6h", "72h_2"]

    # Store paths for temporary CSV files
    local_files = {dtype: f"{local_dir}/{dtype}.csv" for dtype in datasets}
    

    # Clear any existing local files
    for file in local_files.values():
        if os.path.exists(file):
            os.remove(file)


    # Process each time point separately
    for time in times:
        # Load the image metadata for this time point
        image_metadata = pd.DataFrame()
        image_key = f"{base_folder_path}/{time}/Image.csv"
        try:
            response = s3.get_object(Bucket=bucket_name, Key=image_key)
            csv_stream = response['Body'].read().decode('utf-8')
            image_metadata = pd.read_csv(StringIO(csv_stream))

            # Append the image metadata to the local image CSV
            if not os.path.exists(local_files['image']):
                image_metadata.to_csv(local_files['image'], index=False, header=True)
            else:
                image_metadata.to_csv(local_files['image'], mode='a', index=False, header=False)

        except Exception as e:
            print(f"Error processing image file {image_key}: {e}")

        # After loading the image metadata, now process other files (nuclei, cytoplasm, cells)
        for file_type in datasets:
            if file_type != "image":
                key = f"{base_folder_path}/{time}/{file_type.capitalize()}.csv"
                try:
                    print(f"Processing: {key}")
                    response = s3.get_object(Bucket=bucket_name, Key=key)
                    csv_stream = response['Body'].read().decode('utf-8')
                    
                    # Load the entire CSV into memory
                    data_df = pd.read_csv(StringIO(csv_stream))
                    data_df["Metadata_Timepoint"] = time

                    # Merge with image metadata if 'ImageNumber' is in the data
                    if 'ImageNumber' in data_df.columns:
                        data_df = data_df.merge(image_metadata[['ImageNumber', 'Metadata_Plate', 'Metadata_Site', 'Metadata_Well']], 
                                                 on='ImageNumber', how='left')

                    # Write the merged data directly to the local file for this data type
                    if not os.path.exists(local_files[file_type]):
                        data_df.to_csv(local_files[file_type], index=False, header=True)
                    else:
                        data_df.to_csv(local_files[file_type], mode='a', index=False, header=False)

                except Exception as e:
                    print(f"Error processing {key}: {e}")

    # After processing all time points, upload the concatenated CSVs for each data type (including image)
    for file_type, local_path in local_files.items():
        if os.path.exists(local_path):
            output_key = f"{output_prefix}/concatenated_{file_type}.csv"
            with open(local_path, "rb") as data:
                s3.put_object(Bucket=output_bucket, Key=output_key, Body=data)
            print(f"Uploaded {file_type} to S3: s3://{output_bucket}/{output_key}")

    # Clean up local files after the upload
    for file in local_files.values():
        if os.path.exists(file):
            os.remove(file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Concatenate CSV files from S3 for multiple plates.")

    #parser.add_argument("--exp_ids", nargs="+", required=True, help="List of experiment IDs (plate numbers).")
    parser.add_argument("--bucket_name", required=True, help="S3 bucket containing the files.")
    parser.add_argument("--base_folder", required=True, help="Base folder path in S3 where experiment folders are stored.")
    parser.add_argument("--output_bucket", required=True, help="S3 bucket where output files will be saved.")
    parser.add_argument("--output_prefix", required=True, help="Prefix for the output files in S3.")
    parser.add_argument("--local_dir", default="temp_data", help="Local directory for temporary storage.")

    args = parser.parse_args()
    print(f"Processing Plate {args.base_folder}...")
    
    concatenate_csv_from_s3(
        bucket_name=args.bucket_name,
        base_folder_path=args.base_folder,
        output_bucket=args.output_bucket,
        output_prefix=args.output_prefix,
        local_dir=args.local_dir
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


