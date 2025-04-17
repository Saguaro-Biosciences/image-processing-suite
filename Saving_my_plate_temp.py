import argparse
import boto3
import pandas as pd
from io import StringIO
import numpy as np
from pycytominer import feature_select
import pathlib
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os
import io
import logging

# --- Logger Setup ---
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Parameters ---
# Initialize S3 client
s3 = boto3.client('s3')

def read_csv_from_s3(bucket_name, file_key):
    s3 = boto3.client('s3')
    logger.info(f"Reading CSV from s3://{bucket_name}/{file_key}")
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response['Body'].read().decode('utf-8')

    sample = csv_content[:1024]
    dialect = csv.Sniffer().sniff(sample, delimiters=";,") 
    return pd.read_csv(StringIO(csv_content), sep=dialect.delimiter)

# Iterate over both kits in kitnames
def annotate_well(bucket_name, kitnames, base_folder_path, times):
    s3 = boto3.client('s3')

    for kit in kitnames:
        print(f"Processing kit: {kit}")
    
    for time in times:
        print(f"Processing timepoint: {time}")
        Image_key = f"{base_folder_path}/{kit}/{time}/Image.csv"
        Nuclei_key = f"{base_folder_path}/{kit}/{time}/Nuclei.csv"
        Cells_key = f"{base_folder_path}/{kit}/{time}/Cells.csv"
        Cytoplasm_key = f"{base_folder_path}/{kit}/{time}/Cytoplasm.csv"

        Image = read_csv_from_s3(bucket_name, Image_key)
        print(f"Loaded Image: {time}")

        print(f"Loading Nuclei: {time}")
        Nuclei = read_csv_from_s3(bucket_name, Nuclei_key)
        print(f"Loaded Nuclei: {time}")
        
        if 'ImageNumber' in Image.columns and 'Metadata_Well' not in Nuclei.columns:
            Image_map = Image[['ImageNumber', 'Metadata_Well']]
            Nuclei = Nuclei.merge(Image_map, on='ImageNumber', how='left')
        
        csv_buffer = StringIO()
        Nuclei.to_csv(csv_buffer, index=False)
        output_key = f"{base_folder_path}/{kit}/{time}/Nuclei.csv"
        s3.put_object(Bucket=bucket_name, Key=output_key, Body=csv_buffer.getvalue())
        print(f"Saved Nuclei S3: s3://{base_folder_path}/{kit}/{time}/{output_key}")
        del Nuclei

        print(f"Loading Cells: {time}")
        Cells = read_csv_from_s3(bucket_name, Cells_key)
        print(f"Loaded Cells: {time}")
        if 'ImageNumber' in Image.columns and 'Metadata_Well' not in Cells.columns:
            Image_map = Image[['ImageNumber', 'Metadata_Well']]
            Cells = Cells.merge(Image_map, on='ImageNumber', how='left')
        
        csv_buffer = StringIO()
        Cells.to_csv(csv_buffer, index=False)
        output_key = f"{base_folder_path}/{kit}/{time}/Cells.csv"
        s3.put_object(Bucket=bucket_name, Key=output_key, Body=csv_buffer.getvalue())
        print(f"Saved Cells S3: s3://{base_folder_path}/{kit}/{time}/{output_key}")
        del Cells

        print(f"Loading Cytoplasm: {time}")
        Cytoplasm = read_csv_from_s3(bucket_name, Cytoplasm_key)
        print(f"Loaded Cytoplasm: {time}")
        if 'ImageNumber' in Image.columns and 'Metadata_Well' not in Cytoplasm.columns:
            Image_map = Image[['ImageNumber', 'Metadata_Well']]
            Cytoplasm = Cytoplasm.merge(Image_map, on='ImageNumber', how='left')

        csv_buffer = StringIO()
        Cytoplasm.to_csv(csv_buffer, index=False)
        output_key = f"{base_folder_path}/{kit}/{time}/Cytoplasm.csv"
        s3.put_object(Bucket=bucket_name, Key=output_key, Body=csv_buffer.getvalue())
        print(f"Saved Cytoplasm S3: s3://{base_folder_path}/{kit}/{time}/{output_key}")
        del Cytoplasm
    print(f"Loaded and grouped Image data for {kit} at {time}.")

# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate and normalize CellProfiler features from S3.")
    parser.add_argument("--bucket_name",type=str, required=True)
    parser.add_argument("--kitnames",type=str, required=True)
    parser.add_argument("--base_folder_path", type=str,required=True)
    parser.add_argument("--times", type=str, required=True)

    args = parser.parse_args()
    logger.info(f"Starting processing for base folder: {args.base_folder_path}")


    annotate_well(
        bucket_name=args.bucket_name,
        kitnames=args.kitnames,
        base_folder_path=args.base_folder_path,
        times=args.times,
    )

bucket_name = "cellprofiler-resuts"
kitnames = ['1.ChromaLIVE_MOA90']
times = ['12h']
base_folder_path ="IRIC/KitsChromaLIVE2.0_MOA90_outputs"
output_prefix = "IRIC/KitsChromaLIVE2.0_MOA90_outputs"