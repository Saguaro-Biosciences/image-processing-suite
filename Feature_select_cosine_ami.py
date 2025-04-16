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
k = 3
alpha = 2.3538

# --- Functions ---
def double_sigmoid(x):
    return (x/alpha)**k / np.sqrt(1 + (x/alpha)**(2*k))

def read_csv_from_s3(bucket_name, file_key):
    s3 = boto3.client('s3')
    logger.info(f"Reading CSV from s3://{bucket_name}/{file_key}")
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response['Body'].read().decode('utf-8')

    sample = csv_content[:1024]
    dialect = csv.Sniffer().sniff(sample, delimiters=";,") 
    return pd.read_csv(StringIO(csv_content), sep=dialect.delimiter)

def concatenate_normalized_csv_from_s3(bucket_name, plates, base_folder_path, per_time, output_bucket, output_prefix, exp,na_cutoff, corr_3hold, local_dir="temp_data"):
    s3 = boto3.client('s3')
    os.makedirs(local_dir, exist_ok=True)
    logger.info(f"Local directory created or exists: {local_dir}")

    for plate in plates:
        logger.info(f"Processing plate folder: {plate}")
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{base_folder_path}/{plate}/",Delimiter='/')
        matching_files = [obj['Key'] for obj in response.get('Contents', []) if 'Normalized_features' in obj['Key']]
        logger.info(f"Found {len(matching_files)} normalized feature files for plate {plate}")

        normalized_dfs = []
        for file_key in matching_files:
            df = read_csv_from_s3(bucket_name, file_key)
            normalized_dfs.append(df)

    normalized_exp = pd.concat(normalized_dfs, ignore_index=True)
    
    if per_time:
        logger.info("Performing feature selection per timepoint...")
        all_timepoints_selected = []

        for timepoint in normalized_exp["Metadata_Timepoint"].unique():
            df_time = normalized_exp[normalized_exp["Metadata_Timepoint"] == timepoint]
            features = df_time.columns[~df_time.columns.str.contains("Metadata")].tolist()

            feature_select_file = pathlib.Path(f"{local_dir}/normalized_cpfeature_select_{timepoint}.csv")
            feature_select(
                profiles=df_time,
                features=features,
                samples="all",
                na_cutoff=na_cutoff,
                corr_threshold=corr_3hold,
                operation=["variance_threshold", "drop_na_columns", "correlation_threshold", "drop_outliers"],
                output_file=feature_select_file,
                output_type="csv"
            )

            df_selected = pd.read_csv(feature_select_file)
            df_selected["Metadata_Timepoint"] = timepoint
            all_timepoints_selected.append(df_selected)
            feature_select_file.unlink()
            logger.info(f"Selected features saved and removed for timepoint {timepoint}")

        normalized_exp_selected = pd.concat(all_timepoints_selected, ignore_index=True)

    else:
        logger.info("Performing global feature selection...")
        feature_select_file = pathlib.Path(f"{local_dir}/normalized_cpfeature_select.csv")
        features = normalized_exp.columns[~normalized_exp.columns.str.contains("Metadata")].tolist()

        feature_select(
            profiles=normalized_exp,
            features=features,
            samples="all",
            na_cutoff=na_cutoff,
            corr_threshold=corr_3hold,
            operation=["variance_threshold", "drop_na_columns", "correlation_threshold", "drop_outliers"],
            output_file=feature_select_file,
            output_type="csv"
        )

        normalized_exp_selected = pd.read_csv(feature_select_file)
        feature_select_file.unlink()
        logger.info("Global feature selection complete.")

    csv_buffer = StringIO()
    normalized_exp_selected.to_csv(csv_buffer, index=False)
    output_key = f"{output_prefix}/{exp}_CP_features_selected_allTimes_raw.csv"
    s3.put_object(Bucket=output_bucket, Key=output_key, Body=csv_buffer.getvalue())
    logger.info(f"Saved raw selected features to s3://{output_bucket}/{output_key}")

    features = normalized_exp_selected.columns[~normalized_exp_selected.columns.str.contains("Metadata")].tolist()
    normalized_exp_selected[features] = normalized_exp_selected[features].apply(double_sigmoid).abs()
    csv_buffer = StringIO()
    normalized_exp_selected.to_csv(csv_buffer, index=False)
    output_key = f"{output_prefix}/{exp}_CP_features_selected_allTimes_dSig.csv"
    s3.put_object(Bucket=output_bucket, Key=output_key, Body=csv_buffer.getvalue())
    logger.info(f"Saved double sigmoid transformed features to s3://{output_bucket}/{output_key}")

    # Cosine similarity
    averaged_similarities = []
    logger.info("Computing cosine similarities...")

    cpfeature_cos=normalized_exp_selected.drop(columns=['Metadata_Plate','Metadata_Well'])

    for (compound_code, timepoint, compound_concentration) in cpfeature_cos[['Metadata_Compound', 'Metadata_Timepoint', 'Metadata_ConcLevel']].drop_duplicates().values:
        group = cpfeature_cos[
            (cpfeature_cos['Metadata_Compound'] == compound_code) &
            (cpfeature_cos['Metadata_Timepoint'] == timepoint) &
            (cpfeature_cos['Metadata_ConcLevel'] == compound_concentration)
        ]

        if group.empty:
            logger.warning(f"Empty group: {compound_code}, {timepoint}, {compound_concentration}")
            continue

        features = group.drop(columns=['Metadata_Compound', 'Metadata_Timepoint', 'Metadata_ConcLevel']).fillna(0)
        logger.debug(f"Group shape for {compound_code}, {timepoint}, {compound_concentration}: {features.shape}")

        pairwise_similarities = cosine_similarity(features)
        triu_indices = np.triu_indices_from(pairwise_similarities, k=1)
        pairwise_similarities_values = pairwise_similarities[triu_indices]

        avg_similarity = np.mean(pairwise_similarities_values) if len(pairwise_similarities_values) > 0 else np.nan

        averaged_similarities.append({
            'Metadata_Compound': compound_code,
            'Metadata_Timepoint': timepoint,
            'Metadata_ConcLevel': compound_concentration,
            'average_cosine_similarity': avg_similarity
        })

    df_averaged_similarities = pd.DataFrame(averaged_similarities)

    csv_buffer = StringIO()
    output_key = f"{output_prefix}/{exp}_Average_cosine_similarity.csv"
    df_averaged_similarities.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=output_bucket, Key=output_key, Body=csv_buffer.getvalue())
    logger.info(f"Saved cosine similarity to s3://{output_bucket}/{output_key}")

# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate and normalize CellProfiler features from S3.")
    parser.add_argument("--bucket_name",type=str, required=True)
    parser.add_argument("--base_folder", type=str,required=True)
    parser.add_argument("--plates", nargs="+", required=True)
    parser.add_argument("--exp", nargs="+", required=True)
    parser.add_argument("--na_cutoff", type=float, default=0.5)
    parser.add_argument("--corr_3hold", type=float, default=0.9)
    parser.add_argument("--per_time", action='store_true')
    parser.add_argument("--output_bucket", type=str,required=True)
    parser.add_argument("--output_prefix", type=str,required=True)
    parser.add_argument("--local_dir", type=str,default="temp_data")

    args = parser.parse_args()
    logger.info(f"Starting processing for base folder: {args.base_folder}")

    concatenate_normalized_csv_from_s3(
        bucket_name=args.bucket_name,
        base_folder_path=args.base_folder,
        plates=args.plates,
        exp=args.exp,
        na_cutoff=args.na_cutoff,
        corr_3hold=args.corr_3hold,
        per_time=args.per_time,
        output_bucket=args.output_bucket,
        output_prefix=args.output_prefix,
        local_dir=args.local_dir
    )
