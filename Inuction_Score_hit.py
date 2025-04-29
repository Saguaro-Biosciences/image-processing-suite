import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import matplotlib.patches as mpatches
import seaborn as sns
import re
import boto3
import csv
from io import StringIO

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
def extract_timepoint_numeric(tp):
        match = re.search(r'(\d+)', str(tp))
        return int(match.group(1)) if match else float('inf')  # put unrecognized at the end

def read_csv_from_s3(bucket_name, file_key):
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response["Body"].read().decode("utf-8")
    sample = csv_content[:1024]
    dialect = csv.Sniffer().sniff(sample, delimiters=";,")
    return pd.read_csv(StringIO(csv_content), sep=dialect.delimiter)

def upload_image_to_s3(bucket, key, image_path):
    s3 = boto3.client("s3")
    with open(image_path, "rb") as f:
        s3.put_object(Bucket=bucket, Key=key, Body=f)
    logger.info(f"Uploaded image to s3://{bucket}/{key}")

def main(
    bucket_name,
    features_key,
    bioactive_threshold_quantile,
    induction_threshold,
    output_prefix,
    DMSO
):
    logger.info("Reading features file from S3")
    sig = read_csv_from_s3(bucket_name, features_key)

    non_metadata_cols = [col for col in sig.columns if not col.startswith("Metadata_")]
    logger.info(f"Total feature col to account: {len(non_metadata_cols)}")
    sig["induction"] = (sig[non_metadata_cols] > induction_threshold).sum(axis=1) / len(non_metadata_cols)

    sig_ind = sig[[
        "Metadata_Plate", "Metadata_Well", "Metadata_Timepoint",
        "Metadata_Compound", "Metadata_ConcLevel", "induction"
    ]]

    logger.info("Computing per-timepoint DMSO thresholds")
    ind_zpe_all = sig_ind[sig_ind["Metadata_Compound"] == f"{DMSO}"]
    bioactive_thresholds = (
        ind_zpe_all.groupby("Metadata_Timepoint")["induction"]
        .quantile(bioactive_threshold_quantile)
        .to_dict()
    )
    logger.info(f"Computed thresholds: {bioactive_thresholds}")

    # Plot per-timepoint induction distributions
    plt.figure(figsize=(12, 8))
    timepoints_sorted = sorted(bioactive_thresholds.keys(), key=extract_timepoint_numeric)
    for i, tp in enumerate(timepoints_sorted):
        tp_data = ind_zpe_all[ind_zpe_all["Metadata_Timepoint"] == tp]["induction"]
        sns.histplot(tp_data, bins=100, kde=True, label=f"{tp} (thresh={bioactive_thresholds[tp]:.2f})", alpha=0.6)

        plt.axvline(x=bioactive_thresholds[tp], color="black", linestyle="--", linewidth=1)

    plt.xlabel("Induction")
    plt.ylabel("Frequency")
    plt.title("Per-Timepoint DMSO Induction Distributions")
    plt.legend()
    dist_img = "induction_distribution_per_timepoint.png"
    plt.savefig(dist_img, dpi=300)
    plt.close()
    upload_image_to_s3(bucket_name, f"{output_prefix}/induction_distribution_per_timepoint.png", dist_img)


    # Bioactivity analysis
    ind_mean = (
    sig_ind[sig_ind["Metadata_Compound"] != f"{DMSO}"]
    .groupby(["Metadata_Timepoint", "Metadata_Compound", "Metadata_ConcLevel"])
    .agg(induction_mean=("induction", "mean"))
    .reset_index()
    )

    ind_mean["Bioactive"] = ind_mean.apply(lambda row: int(row["induction_mean"] >= bioactive_thresholds.get(row["Metadata_Timepoint"], np.inf)), axis=1)
    compound_bioactivity = (
        ind_mean.groupby(["Metadata_Timepoint", "Metadata_Compound"])["Bioactive"]
        .max()
        .reset_index()
    )
    csv_buffer = StringIO()
    output_key = f"{output_prefix}/Bioactivities.csv"
    heatmap_data.to_csv(csv_buffer, index=False)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket_name, Key=output_key, Body=csv_buffer.getvalue())
    logger.info(f"Saved cosine similarity to s3://{bucket_name}/{output_key}")

    logger.info("Generating Venn diagrams")

    # All compound codes
    all_compounds = set(compound_bioactivity["Metadata_Compound"])
    bioactive_compounds = set(compound_bioactivity.loc[compound_bioactivity["Bioactive"] == 1, "Metadata_Compound"])

    # Venn 1: All compounds vs Bioactive compounds
    plt.figure(figsize=(8, 5))
    venn2([all_compounds, bioactive_compounds], set_labels=("All Compounds", f"Bioactive {int(len(bioactive_compounds)/len(all_compounds)*100)}%"))
    plt.title("Bioactivity Overview")
    venn_all_vs_bioactive = "venn_all_vs_bioactive.png"
    plt.savefig(venn_all_vs_bioactive)
    plt.close()
    upload_image_to_s3(bucket_name, f"{output_prefix}/venn_all_vs_bioactive.png", venn_all_vs_bioactive)

    # Identify 48h column (int 48, or '48', or '15' fallback)
    tp48_col = next((h for h in compound_bioactivity.Metadata_Timepoint.unique().tolist() if str(h) in ["48", "48h", "15","Time_14"]), None)
    
    if tp48_col:
        tp48_induction = set(
            compound_bioactivity.loc[
                (compound_bioactivity["Metadata_Timepoint"] == tp48_col) & 
                (compound_bioactivity["Bioactive"] == 1),
                "Metadata_Compound"
            ]
        )
        
        plt.figure(figsize=(8, 6))
        venn2([all_compounds, tp48_induction], set_labels=("All Bioactive", f"48h Bioactive {int(len(tp48_induction)/len(all_compounds)*100)}%"))
        plt.title("Bioactive Compounds at 48h vs All Timepoints")
        venn_48_vs_allbio = "venn_48_vs_allbio.png"
        plt.savefig(venn_48_vs_allbio)
        plt.close()
        upload_image_to_s3(bucket_name, f"{output_prefix}/venn_48_vs_allbio.png", venn_48_vs_allbio)

    
    logger.info("Performing heatmap.")

    # Ensure compound names are uppercase
    compound_bioactivity["Metadata_Compound"] = compound_bioactivity["Metadata_Compound"].apply(lambda x: x.upper())

    # Normalize timepoint labels for consistent sorting
    compound_bioactivity["Metadata_Timepoint_Numeric"] = compound_bioactivity["Metadata_Timepoint"].apply(extract_timepoint_numeric)

    # Sort timepoints for consistent plotting
    timepoint_order = (
        compound_bioactivity[["Metadata_Timepoint", "Metadata_Timepoint_Numeric"]]
        .drop_duplicates()
        .sort_values("Metadata_Timepoint_Numeric")["Metadata_Timepoint"]
        .tolist()
    )

    # Pivot the table and aggregate duplicate compounds using max (logical OR)
    heatmap_data = (
        compound_bioactivity
        .pivot_table(
            index="Metadata_Compound",
            columns="Metadata_Timepoint",
            values="Bioactive",
            aggfunc='max',
            fill_value=0
        )
    )

    # Reorder columns
    heatmap_data = heatmap_data[timepoint_order]

    # Add a Bioactive summary column: 1 if the compound was ever active
    heatmap_data["Bioactive"] = (heatmap_data > 0).any(axis=1).astype(int)
    

    # Set up the figure
    plt.figure(figsize=(10, min(20, 0.2 * len(heatmap_data))))
    sns.heatmap(
        heatmap_data,
        cmap=sns.color_palette(["lightgrey", "black"]),
        linewidths=0.5,
        linecolor='black',
        cbar=False,
        annot=False,
        xticklabels=True,
        yticklabels=True
    )

    # Add a custom legend
    plt.title("Compound Bioactivity by Timepoint", fontsize=12, pad=10)
    plt.xlabel("Timepoint")
    plt.ylabel("Compound")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=6)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='black', label='Active', edgecolor='black')]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()

    # Save and upload
    bioheat_img = "compound_bioactivity_heatmap.png"
    plt.savefig(bioheat_img, dpi=300)
    plt.close()

    upload_image_to_s3(bucket_name, f"{output_prefix}/compound_bioactivity_heatmap.png", bioheat_img)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bioactivity Analysis with Venn Diagrams and Heatmaps.")
    parser.add_argument("--bucket_name", required=True, help="S3 bucket with feature and platemap files.")
    parser.add_argument("--features_key", required=True, help="S3 key to the normalized selected feature CSV.")
    parser.add_argument("--bioactive_threshold_quantile", type=float, default=0.95, help="Quantile threshold for ZPE/DMSO induction.")
    parser.add_argument("--induction_threshold", type=float, default=0.95, help="Threshold to consider a feature induced.")
    parser.add_argument("--output_prefix", required=True, help="S3 prefix where output images will be saved.")
    parser.add_argument("--DMSO", type=str,default="DMSO", help="DMSO nomenclature used to normalize in the plateMap.")
    args = parser.parse_args()

    main(
        bucket_name=args.bucket_name,
        features_key=args.features_key,
        bioactive_threshold_quantile=args.bioactive_threshold_quantile,
        induction_threshold=args.induction_threshold,
        output_prefix=args.output_prefix,
        DMSO=args.DMSO
    )
