"""
Normalize_CP_ami_CellPainting.py

Concatenates and normalizes CellProfiler (Cell Painting) output CSVs for one
or more plates and timepoints, then writes normalized, well-aggregated
feature tables back to S3.

For each plate:
  - Reads the plate map (Plate_{plate}_PlateMap.csv).
  - For each timepoint, reads Image/Nuclei/Cells/Cytoplasm CSVs, optionally
    drops QC-failed images (--qc_drop) and site-count-scales features,
    aggregates to well level (--well_agg_func), merges all compartments,
    annotates with the plate map, and normalizes features against DMSO
    wells using pycytominer's mad_robustize method.
  - Saves the normalized table to
    {output_bucket}/{output_prefix}/{plate}/Normalized_features_{time}.csv

Folder layout flags control the expected key structure for the
Image/Nuclei/Cells/Cytoplasm CSVs:
  Default:              base_folder/{plate}/{time}/{Image,Nuclei,Cells,Cytoplasm}.csv
  --no_time_subFolder:   base_folder/{plate}/{Image,Nuclei,Cells,Cytoplasm}.csv
  --flat_folder:         base_folder/{Image,Nuclei,Cells,Cytoplasm}.csv
                         (use when base_folder is already scoped to a single
                         plate/timepoint run, with no plate subfolder at all)

The plate map is always read from base_folder/Plate_{plate}_PlateMap.csv
regardless of these flags.

Local data option (--local_data_dir):
  If set, CSVs are read from local disk first (e.g. a folder populated by an
  `aws s3 sync` at instance launch) instead of going to S3 for every file.
  The local folder is expected to be FLAT — i.e. it directly contains
  Image.csv, Nuclei.csv, Cells.csv, Cytoplasm.csv, and
  Plate_{plate}_PlateMap.csv, regardless of --flat_folder/--no_time_subFolder,
  since those flags only affect S3 key construction. If a file isn't found
  locally, this script falls back to reading it from S3 using the normal
  key logic, so partial local caches are safe.
"""

import argparse
import os
import boto3
from botocore.config import Config
import pandas as pd
from io import StringIO
import numpy as np
from functools import reduce
from pycytominer import annotate, normalize
import csv
import logging

# --- Logger Setup ---
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def sniff_delimiter(sample):
    """
    Tries to detect the CSV delimiter with csv.Sniffer(). Falls back to
    comma if detection fails (e.g. on files where Sniffer can't confidently
    tell ';' from ',', which previously caused a hard crash).
    """
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,").delimiter
    except csv.Error:
        logger.info("csv.Sniffer() could not confidently detect a delimiter; defaulting to ','")
        return ","


def read_csv_local(file_path):
    logger.info(f"Reading CSV from local disk: {file_path}")
    # Only read a small sample for delimiter sniffing — don't load the whole
    # file into memory as a string. For large tables (Cells.csv especially),
    # doing that first and then re-parsing via StringIO roughly doubles peak
    # memory and loses pandas' fast C-parser file streaming, which can turn
    # a multi-GB file into a multi-hour (or effectively hung) read.
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = f.read(1024)
    delimiter = sniff_delimiter(sample)
    return pd.read_csv(file_path, sep=delimiter)


def read_csv_from_s3(bucket_name, file_key, s3):
    logger.info(f"Reading CSV from s3://{bucket_name}/{file_key}")
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response['Body'].read().decode('utf-8')
    sample = csv_content[:1024]
    delimiter = sniff_delimiter(sample)
    return pd.read_csv(StringIO(csv_content), sep=delimiter)


def read_csv(bucket_name, file_key, s3, local_data_dir, local_filename):
    """
    Reads a CSV, preferring local disk (local_data_dir/local_filename) if
    local_data_dir is set and the file exists there. Falls back to S3
    (bucket_name/file_key) otherwise.
    """
    if local_data_dir:
        local_path = os.path.join(local_data_dir, local_filename)
        if os.path.isfile(local_path):
            return read_csv_local(local_path)
        logger.info(f"'{local_filename}' not found locally in {local_data_dir}, falling back to S3")
    return read_csv_from_s3(bucket_name, file_key, s3)


def concatenate_csv_from_s3(bucket_name, plates, times, base_folder_path, output_bucket, DMSO, output_prefix, well_agg_func, no_time_subFolder, qc_drop, flat_folder, local_data_dir):
    custom_config = Config(
        connect_timeout=30,
        read_timeout=5000,
        retries={
            'max_attempts': 3,
            'mode': 'standard'
        }
    )
    s3 = boto3.client('s3', config=custom_config)

    for plate in plates:
        logger.info(f"Processing plate ID: {plate}")
        plate_clean = plate.lstrip('binned/')
        platemap_key = f"{base_folder_path}/Plate_{plate_clean}_PlateMap.csv"
        platemap_filename = f"Plate_{plate_clean}_PlateMap.csv"
        filtered_plateMap = read_csv(bucket_name, platemap_key, s3, local_data_dir, platemap_filename)
        filtered_plateMap = filtered_plateMap[['Metadata_Compound', 'Metadata_ConcLevel', 'Metadata_Well', 'Metadata_Plate']]  # plate map
        filtered_plateMap["Metadata_Compound"] = filtered_plateMap["Metadata_Compound"].apply(lambda x: str(x).upper())
        for time in times:
            logger.info(f"Processing timepoint: {time}")
            table_info = {
                'Image': 'Image_',
                'Nuclei': 'DNA_',
                'Cells': 'Cell_',
                'Cytoplasm': 'Cyto_'
            }

            drop_substrings = ['ExecutionTime', 'ModuleError', 'URL']
            tables = {}

            for name in table_info:
                if flat_folder:
                    file_key = f"{base_folder_path}/{name}.csv"
                elif no_time_subFolder:
                    file_key = f"{base_folder_path}/{plate}/{name}.csv"
                else:
                    file_key = f"{base_folder_path}/{plate}/{time}/{name}.csv"

                df = read_csv(bucket_name, file_key, s3, local_data_dir, f"{name}.csv")

                tables[name] = df  # Save immediately so Image is available

        # Now propagate Metadata_Well using Image table
            image_df = tables.get("Image")
            failing_images = image_df.loc[image_df.filter(like='ImageQC_').any(axis=1), 'ImageNumber']
            for name, df in tables.items():
                if 'Metadata_Well' not in df.columns:
                    logger.info(f"'Metadata_Well' missing in {name}, merging from Image.csv using ImageNumber")
                    df = df.merge(
                        image_df[['ImageNumber', 'Metadata_Well', 'Metadata_Site']],
                        on='ImageNumber',
                        how='left'
                    )
                    tables[name] = df
                if qc_drop:
                    logger.info(f"Removing QC failed images for {time}")
                    tables[name] = df[~df['ImageNumber'].isin(failing_images)]

            for name, prefix in table_info.items():
                df = tables[name]

                if qc_drop:
                    df = df.drop(columns=[
                        col for col in df.columns
                        if col == 'ImageNumber'
                        or (col.startswith('Metadata') and col not in {'Metadata_Well', 'Metadata_Site'})
                        or any(sub in col for sub in drop_substrings)
                    ])

                    df = df.rename(columns=lambda x: prefix + x if not x.startswith('Metadata_') else x)
                    # Get number of sites per well
                    site_counts = df.groupby("Metadata_Well")["Metadata_Site"].nunique()
                    max_sites = site_counts.max()

                    # Compute scaling factors
                    scaling_factors = (max_sites / site_counts).rename("scaling_factor")

                    # Merge scaling factor into df
                    df = df.merge(scaling_factors, on="Metadata_Well")

                    # Select integer columns not starting with 'Metadata'
                    features_to_scale = [
                        col for col in df.select_dtypes(include="integer").columns
                        if not col.startswith("Metadata")
                    ]

                    # Apply scaling
                    df[features_to_scale] = df[features_to_scale].multiply(df["scaling_factor"], axis=0)

                    # Clean up and aggregate
                    df.drop(columns=["scaling_factor", "Metadata_Site"], inplace=True)

                else:
                    df = df.drop(columns=[
                        col for col in df.columns
                        if col == 'ImageNumber'
                        or (col.startswith('Metadata') and col not in {'Metadata_Well'})
                        or any(sub in col for sub in drop_substrings)
                    ])

                    df = df.rename(columns=lambda x: prefix + x if not x.startswith('Metadata_') else x)
                df = df.groupby("Metadata_Well", as_index=False).agg(well_agg_func)
                tables[name] = df

            df_CP_merged = reduce(lambda left, right: pd.merge(left, right, on='Metadata_Well', how='outer'), tables.values())
            del tables

            df_CP_merged = annotate(df_CP_merged, filtered_plateMap, join_on=[["Metadata_Well"], ["Metadata_Well"]])
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

            csv_buffer = StringIO()
            normalized_exp.to_csv(csv_buffer, index=False)
            output_key = f"{output_prefix}/{plate}/Normalized_features_{time}.csv"
            s3.put_object(Bucket=output_bucket, Key=output_key, Body=csv_buffer.getvalue())
            logger.info(f"Saved to S3: s3://{output_bucket}/{output_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize each timepoint of a project folder, outputs normalized tables against DMSO.")

    parser.add_argument("--bucket_name", type=str, required=True, help="S3 bucket containing the files.")
    parser.add_argument("--base_folder", type=str, required=True, help="Base folder path in S3 where experiment folders are stored.")
    parser.add_argument("--plates", nargs="+", required=True, help="List of plates list to process (prefix Plate/Time/csv).")
    parser.add_argument("--times", nargs="+", help="List of times to process (prefix Plate/Time/csv).")
    parser.add_argument("--DMSO", type=str, default="DMSO", help="DMSO nomenclature used to normalize in the plateMap.")
    parser.add_argument("--output_bucket", type=str, required=True, help="S3 bucket where output files will be saved.")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for the output files in S3.")
    parser.add_argument("--well_agg_func", type=str, default="mean", help="Function to aggregate at well level. Default mean.")
    parser.add_argument("--no_time_subFolder", action='store_true', help="Set if there's no timepoint subfolder: base_folder/{plate}/{name}.csv")
    parser.add_argument("--flat_folder", action='store_true', help="Set if there's no plate or timepoint subfolder at all: base_folder/{name}.csv")
    parser.add_argument("--qc_drop", action='store_true')
    parser.add_argument("--local_data_dir", type=str, default=None, help="If set, read CSVs from this local flat folder first (e.g. populated by 'aws s3 sync' at launch), falling back to S3 for any file not found locally.")

    args = parser.parse_args()
    logger.info(f"Starting normalization for base folder: {args.base_folder}")

    concatenate_csv_from_s3(
        bucket_name=args.bucket_name,
        base_folder_path=args.base_folder,
        plates=args.plates,
        times=args.times,
        no_time_subFolder=args.no_time_subFolder,
        qc_drop=args.qc_drop,
        DMSO=args.DMSO,
        output_bucket=args.output_bucket,
        output_prefix=args.output_prefix,
        well_agg_func=args.well_agg_func,
        flat_folder=args.flat_folder,
        local_data_dir=args.local_data_dir
    )