import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns
import boto3
import csv
from io import StringIO

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
    sig["induction"] = (sig[non_metadata_cols] > induction_threshold).sum(axis=1) / len(non_metadata_cols)

    sig_ind = sig[[
        "Metadata_Plate", "Metadata_Well", "Metadata_Timepoint",
        "Metadata_Compound", "Metadata_ConcLevel", "induction"
    ]]

    logger.info("Computing bioactive threshold using ZPE control")
    ind_zpe = sig_ind[sig_ind["Metadata_Compound"] == f"{DMSO}"]
    bioactive_threshold = np.quantile(ind_zpe["induction"], bioactive_threshold_quantile)
    logger.info(f"Bioactive threshold: {bioactive_threshold}")

    # Plot induction distribution
    dist_img = "induction_distribution.png"
    ind_zpe.induction.hist(bins=100)
    plt.xlabel("Induction")
    plt.ylabel("Frequency")
    plt.axvline(x=bioactive_threshold, color="red", linestyle="dashed", linewidth=2, label=f"Threshold {bioactive_threshold:.2f}")
    plt.title("Distribution of ZPE Induction Score")
    plt.legend()
    plt.savefig(dist_img)
    plt.close()
    upload_image_to_s3(bucket_name, f"{output_prefix}/induction_distribution.png", dist_img)

    # Bioactivity analysis
    ind_mean = (
        sig_ind[sig_ind["Metadata_Compound"] != f"{DMSO}"]
        .groupby(["Metadata_Timepoint", "Metadata_Compound", "Metadata_ConcLevel"])
        .agg(induction_mean=("induction", "mean"))
        .reset_index()
    )

    wide = ind_mean.pivot(index=["Metadata_Timepoint", "Metadata_Compound"],
                          columns="Metadata_ConcLevel",
                          values="induction_mean").reset_index()

    wide["bioactivity"] = (wide.iloc[:, 2:] >= bioactive_threshold).any(axis=1).astype(int)

    wide_tp = wide.pivot(index="Metadata_Compound",
                         columns="Metadata_ConcLevel",
                         values="bioactivity").reset_index()

    wide_tp["Bioactive"] = (wide_tp.iloc[:, 1:] > 0).any(axis=1).astype(int)

    logger.info("Generating Venn diagrams")

    # All compound codes
    all_compounds = set(wide_tp["Metadata_Compound"])
    bioactive_compounds = set(wide_tp.loc[wide_tp["Bioactive"] == 1, "Metadata_Compound"])

    # Venn 1: All compounds vs Bioactive compounds
    plt.figure(figsize=(5, 5))
    venn2([all_compounds, bioactive_compounds], set_labels=("All Compounds", "Bioactive"))
    plt.title("Bioactivity Overview")
    venn_all_vs_bioactive = "venn_all_vs_bioactive.png"
    plt.savefig(venn_all_vs_bioactive)
    plt.close()
    upload_image_to_s3(bucket_name, f"{output_prefix}/venn_all_vs_bioactive.png", venn_all_vs_bioactive)

    # Identify 48h column (int 48, or '48', or '15' fallback)
    tp48_col = next((col for col in wide_tp.columns if str(col) in ["48", "48h", "15"]), None)
    
    if tp48_col:
        tp48_induction = set(wide_tp.loc[wide_tp[tp48_col] == 1, "Metadata_Compound"])

        plt.figure(figsize=(6, 6))
        venn2([bioactive_compounds, tp48_induction], set_labels=("All Bioactive", "48h Bioactive"))
        plt.title("Bioactive Compounds at 48h vs All Timepoints")
        venn_48_vs_allbio = "venn_48_vs_allbio.png"
        plt.savefig(venn_48_vs_allbio)
        plt.close()
        upload_image_to_s3(bucket_name, f"{output_prefix}/venn_48_vs_allbio.png", venn_48_vs_allbio)

    heatmap_img = "bioactivity_heatmap.png"
    heatmap_df.set_index("Metadata_Compound", inplace=True)
    heatmap_df = heatmap_df.sort_index()

    logger.info("Plotting bioactivity heatmap")
    plt.figure(figsize=(12, max(6, 0.4 * len(heatmap_df))))
    ax = sns.heatmap(heatmap_df, cmap="gray_r", linewidths=0.5, cbar_kws={"ticks": [0, 1]})
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Inactive", "Active"])
    plt.xlabel("Time Points")
    plt.ylabel("Compounds")
    plt.title("Binary Heatmap of Compound Bioactivity Over Time")
    plt.tight_layout()
    plt.savefig(heatmap_img)
    plt.close()
    upload_image_to_s3(bucket_name, f"{output_prefix}/bioactivity_heatmap.png", heatmap_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bioactivity Analysis with Venn Diagrams and Heatmaps.")
    parser.add_argument("--bucket_name", required=True, help="S3 bucket with feature and platemap files.")
    parser.add_argument("--features_key", required=True, help="S3 key to the normalized feature CSV.")
    parser.add_argument("--plate_map_key", required=True, help="S3 key to the plate map CSV.")
    parser.add_argument("--bioactive_threshold_quantile", type=float, default=0.95, help="Quantile threshold for ZPE/DMSO induction.")
    parser.add_argument("--induction_threshold", type=float, default=0.95, help="Threshold to consider a feature induced.")
    parser.add_argument("--output_prefix", required=True, help="S3 prefix where output images will be saved.")
    parser.add_argument("--DMSO", type=str,default="DMSO", help="DMSO nomenclature used to normalize in the plateMap.")
    args = parser.parse_args()

    main(
        bucket_name=args.bucket_name,
        features_key=args.features_key,
        plate_map_key=args.plate_map_key,
        bioactive_threshold_quantile=args.bioactive_threshold_quantile,
        induction_threshold=args.induction_threshold,
        output_prefix=args.output_prefix,
        DMSO=args.DMSO
    )
