import argparse
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


def read_csv_from_s3(bucket_name, file_key,s3):
    logger.info(f"Reading CSV from s3://{bucket_name}/{file_key}")
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response['Body'].read().decode('utf-8')

    sample = csv_content[:1024]
    dialect = csv.Sniffer().sniff(sample, delimiters=";,")
    return pd.read_csv(StringIO(csv_content), sep=dialect.delimiter)

def concatenate_csv_from_s3(bucket_name, plates, times, base_folder_path, output_bucket, DMSO,output_prefix, well_agg_func,no_time_subFolder):
    custom_config = Config(
    connect_timeout=30,  # Time to establish connection
    read_timeout=600     # Time to wait for data (increase as needed)
    )
    s3 = boto3.client('s3', config=custom_config)

    for plate in plates:
        logger.info(f"Processing plate ID: {plate}")
        filtered_plateMap = read_csv_from_s3(bucket_name, f"{base_folder_path}/{plate}_PlateMap.csv",s3)
        filtered_plateMap = filtered_plateMap[['Metadata_Compound', 'Metadata_ConcLevel', 'Metadata_Well', 'Metadata_Plate']]# plate map 
        filtered_plateMap["Metadata_Compound"] = filtered_plateMap["Metadata_Compound"].apply(lambda x: x.upper())
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
                if not no_time_subFolder:
                    file_key = f"{base_folder_path}/{plate}/{time}/{name}.csv"
                elif no_time_subFolder:
                    file_key = f"{base_folder_path}/{plate}/{name}.csv"
                df = read_csv_from_s3(bucket_name, file_key,s3)

                tables[name] = df  # Save immediately so Image is available

        # Now propagate Metadata_Well using Image table
            image_df = tables.get("Image")
            for name, df in tables.items():
                if 'Metadata_Well' not in df.columns:
                    logger.info(f"'Metadata_Well' missing in {name}, merging from Image.csv using ImageNumber")
                    df = df.merge(
                        image_df[['ImageNumber', 'Metadata_Well']],
                        on='ImageNumber',
                        how='left'
                    )
                    tables[name] = df

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

    parser.add_argument("--bucket_name", type=str,required=True, help="S3 bucket containing the files.")
    parser.add_argument("--base_folder",type=str, required=True, help="Base folder path in S3 where experiment folders are stored.")
    parser.add_argument("--plates", nargs="+", required=True, help="List of plates list to process (prefix Plate/Time/csv).")
    parser.add_argument("--times", nargs="+", help="List of times to process (prefix Plate/Time/csv).")
    parser.add_argument("--DMSO", type=str,default="DMSO", help="DMSO nomenclature used to normalize in the plateMap.")
    parser.add_argument("--output_bucket",type=str, required=True, help="S3 bucket where output files will be saved.")
    parser.add_argument("--output_prefix", type=str,required=True, help="Prefix for the output files in S3.")
    parser.add_argument("--well_agg_func",type=str, default="mean", help="Function to aggregate at well level. Default mean.")
    parser.add_argument("--no_time_subFolder", action='store_true')

    args = parser.parse_args()
    logger.info(f"Starting normalization for base folder: {args.base_folder}")

    concatenate_csv_from_s3(
        bucket_name=args.bucket_name,
        base_folder_path=args.base_folder,
        plates=args.plates,
        times=args.times,
        no_time_subFolder= args.no_time_subFolder,
        DMSO=args.DMSO,
        output_bucket=args.output_bucket,
        output_prefix=args.output_prefix,
        well_agg_func=args.well_agg_func
    )
