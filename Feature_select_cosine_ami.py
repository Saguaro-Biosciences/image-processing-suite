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
import pathlib
import io

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

def concatenate_normalized_csv_from_s3(bucket_name, plates, base_folder_path, output_bucket, output_prefix,na_cutoff, corr_3hold,local_dir="temp_data"):
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

    normalized_dfs = []

    for plate in plates:
        print(f"Processing Plate folder: {plate}")
        #reading every file onder the plate folder to get the normalized features
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{base_folder_path}/{plate}/")
        matching_files = [
        obj['Key'] for obj in response.get('Contents', []) if 'Normalized_features' in obj['Key']
        ]
        normalized_dfs = []
        for file_key in matching_files:
            df = read_csv_from_s3(bucket_name, file_key)
            normalized_dfs.append(df)

        # Concatenate all plates into one DataFrame
        normalized_exp = pd.concat(normalized_dfs, ignore_index=True)

        feature_select_file = pathlib.Path(f"{local_dir}/normalized_cpfeature_select.csv")
        feature_select_opts = ["variance_threshold", "drop_na_columns", "correlation_threshold", "drop_outliers"]
        features = normalized_exp.columns[~normalized_exp.columns.str.contains("Metadata")].tolist()
        # na_cutoff=0.05, corr_threshold=0.9
        feature_select(
            profiles=normalized_exp,
            features=features,
            samples="all",
            na_cutoff=na_cutoff,
            corr_threshold= corr_3hold,
            operation=feature_select_opts,
            output_file=feature_select_file,
            output_type="csv"
        )
    
        normalized_exp_selected = pd.read_csv(feature_select_file)
        if feature_select_file.exists():
            feature_select_file.unlink()  # This deletes the file
            print(f"Deleted local file: {feature_select_file}")

        csv_buffer = StringIO()
        normalized_exp_selected.to_csv(csv_buffer, index=False)
        output_key = f"{output_prefix}/CP_features_selected_allTimes_raw.csv"
        s3.put_object(Bucket=output_bucket, Key=output_key, Body=csv_buffer.getvalue())
        print(f"Saved to S3: s3://{output_bucket}/{output_key}")


        normalized_exp_selected[features]=normalized_exp_selected[features].apply(double_sigmoid)
        # Absolute value
        normalized_exp_selected.loc[:,features]=normalized_exp_selected.loc[:,features].abs()
        csv_buffer = StringIO()
        normalized_exp_selected.to_csv(csv_buffer, index=False)
        output_key = f"{output_prefix}/CP_features_selected_allTimes_dSig.csv"
        s3.put_object(Bucket=output_bucket, Key=output_key, Body=csv_buffer.getvalue())
        print(f"Saved to S3: s3://{output_bucket}/{output_key}")

        # Compute cosine similarities for each plate individually
        averaged_similarities = []
    
    
        print(f"Processing cosine similarity")
        
        # Filter the selected data for the current plate
        cpfeature_cos = normalized_exp_selected[normalized_exp_selected['Metadata_Timepoint'] == plate].drop(columns=['Metadata_Well'])
        
        for (compound_code, timepoint, compound_concentration) in cpfeature_cos[['Metadata_Compound', 'Metadata_Timepoint', 'Metadata_ConcLevel']].drop_duplicates().values:
            # Filter the data based on the unique combination of metadata values
            group = cpfeature_cos[
                (cpfeature_cos['Metadata_Compound'] == compound_code) &
                (cpfeature_cos['Metadata_Timepoint'] == timepoint) &
                (cpfeature_cos['Metadata_ConcLevel'] == compound_concentration)
            ]
            if group.empty:
                print(f"Empty group: {compound_code}, {timepoint}, {compound_concentration}")
                continue  # Skip to the next iteration if group is empty

            # Extract the feature columns
            features = group.drop(columns=['Metadata_Compound', 'Metadata_Timepoint', 'Metadata_ConcLevel'])
            features = features.fillna(0)
            print(f"Features shape for {compound_code}, {timepoint}, {compound_concentration}: {features.shape}")

            # Compute the pairwise cosine similarity within the same metadata group
            pairwise_similarities = cosine_similarity(features)

            # Extract the upper triangular part of the similarity matrix, excluding diagonal
            triu_indices = np.triu_indices_from(pairwise_similarities, k=1)
            pairwise_similarities_values = pairwise_similarities[triu_indices]

            # Calculate the average cosine similarity for this subset
            avg_similarity = np.mean(pairwise_similarities_values) if len(pairwise_similarities_values) > 0 else np.nan

            # Store the average similarity along with subset information
            averaged_similarities.append({
                'Metadata_Compound': compound_code,
                'Metadata_Timepoint': timepoint,
                'Metadata_ConcLevel': compound_concentration,
                'average_cosine_similarity': avg_similarity
            })


        # Convert the results into a DataFrame for the current plate
        df_averaged_similarities = pd.DataFrame(averaged_similarities)

        # Save Average_cosine_similarity to S3
        csv_buffer = StringIO()
        df_averaged_similarities.to_csv(csv_buffer, index=False)
        output_key = f"{output_prefix}/{plate}/Average_cosine_similarity.csv"
        s3.put_object(Bucket=output_bucket, Key=output_key, Body=csv_buffer.getvalue())
        print(f"Saved to S3: s3://{output_bucket}/{output_key}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Concatenate CSV files from S3 for multiple plates.")

    #parser.add_argument("--exp_ids", nargs="+", required=True, help="List of experiment IDs (plate numbers).")
    parser.add_argument("--bucket_name", required=True, help="S3 bucket containing the files.")
    parser.add_argument("--base_folder", required=True, help="Base folder path in S3 where experiment folders are stored.")
    parser.add_argument("--plates", nargs="+", required=True, help="List of plates list to process (prefix as they are from CP Feature extraction).")
    parser.add_argument("--na_cutoff", default=0.5, help="Absent value threshold for feature selection.")
    parser.add_argument("--corr_3hold", default=0.9, help="Correlation threshold for feature selection.")
    parser.add_argument("--output_bucket", required=True, help="S3 bucket where output files will be saved.")
    parser.add_argument("--output_prefix", required=True, help="Prefix for the output files in S3.")
    parser.add_argument("--local_dir", default="temp_data", help="Local directory for temporary storage.")

    args = parser.parse_args()
    print(f"Processing Plate {args.base_folder}...")
    
    concatenate_normalized_csv_from_s3(
        bucket_name=args.bucket_name,  # cellprofiler-resuts
        base_folder_path=args.base_folder, # IRIC/CQDM_CTL_Plate_Validation_202501/Plate_1
        plates= args.plates,
        na_cutoff=int(args.na_cutoff),
        corr_3hold=int(args.corr_3hold),
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