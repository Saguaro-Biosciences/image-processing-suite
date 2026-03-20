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
    """Extracts a numeric value from a timepoint string for sorting."""
    match = re.search(r'(\d+)', str(tp))
    return int(match.group(1)) if match else float('inf')  # put unrecognized at the end

def read_csv_from_s3(bucket_name, file_key):
    """Reads a CSV file from an S3 bucket into a pandas DataFrame."""
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response["Body"].read().decode("utf-8")
    sample = csv_content[:1024]
    dialect = csv.Sniffer().sniff(sample, delimiters=";,")
    return pd.read_csv(StringIO(csv_content), sep=dialect.delimiter)

def upload_image_to_s3(bucket, key, image_path):
    """Uploads a local image file to an S3 bucket."""
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

    # Compute per-plate, per-timepoint DMSO thresholds
    logger.info("Computing per-plate, per-timepoint DMSO thresholds")
    ind_zpe_all = sig_ind[sig_ind["Metadata_Compound"] == f"{DMSO}"]
    bioactive_thresholds = (
        ind_zpe_all.groupby(["Metadata_Plate", "Metadata_Timepoint"])["induction"]
        .quantile(bioactive_threshold_quantile)
        .to_dict()
    )
    logger.info(f"Computed thresholds (Plate, Timepoint): {bioactive_thresholds}")

    # Generate a separate induction distribution plot for each plate
    logger.info("Generating induction distribution plots per plate.")
    unique_plates_dist = ind_zpe_all["Metadata_Plate"].unique()

    for plate_id in unique_plates_dist:
        plt.figure(figsize=(12, 8))
        
        plate_dmso_data = ind_zpe_all[ind_zpe_all["Metadata_Plate"] == plate_id]
        
        plate_timepoints_sorted = sorted(
            plate_dmso_data["Metadata_Timepoint"].unique(),
            key=extract_timepoint_numeric
        )
        
        for tp in plate_timepoints_sorted:
            tp_data = plate_dmso_data[plate_dmso_data["Metadata_Timepoint"] == tp]["induction"]
            threshold = bioactive_thresholds.get((plate_id, tp))
            
            label_text = f"Timepoint {tp}"
            if threshold is not None:
                label_text += f" (thresh={threshold:.2f})"
            
            ax = sns.histplot(tp_data, bins=100, kde=True, label=label_text, alpha=0.6)
            plot_color = ax.get_lines()[-1].get_c()
            
            if threshold is not None:
                plt.axvline(x=threshold, color=plot_color, linestyle="--", linewidth=2)

        plt.xlabel("Induction Score")
        plt.ylabel("Frequency")
        plt.title(f"DMSO Induction Distribution for Plate: {plate_id}")
        plt.legend()
        
        dist_img = f"induction_distribution_plate_{plate_id}.png"
        plt.savefig(dist_img, dpi=300, bbox_inches='tight')
        plt.close()
        
        upload_image_to_s3(bucket_name, f"{output_prefix}/{dist_img}", dist_img)

    # Bioactivity analysis now grouped by Plate, Timepoint, Compound, and Concentration
    ind_mean = (
        sig_ind[sig_ind["Metadata_Compound"] != f"{DMSO}"]
        .groupby(["Metadata_Plate", "Metadata_Timepoint", "Metadata_Compound", "Metadata_ConcLevel"])
        .agg(induction_mean=("induction", "mean"))
        .reset_index()
    )
    csv_buffer = StringIO()
    output_key = f"{output_prefix}/Bioactivities_per_plate_doses.csv"
    ind_mean.to_csv(csv_buffer, index=False)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket_name, Key=output_key, Body=csv_buffer.getvalue())
    logger.info(f"Saved Bioactivities s3://{bucket_name}/{output_key}")

    # Apply the per-plate, per-timepoint threshold for bioactivity
    ind_mean["Bioactive"] = ind_mean.apply(
        lambda row: int(row["induction_mean"] >= bioactive_thresholds.get((row["Metadata_Plate"], row["Metadata_Timepoint"]), np.inf)),
        axis=1
    )

    # This summary is for the overall Venn diagrams
    compound_bioactivity_summary = (
        ind_mean.groupby(["Metadata_Timepoint", "Metadata_Compound"])["Bioactive"]
        .max()
        .reset_index()
    )
    
    logger.info("Generating Venn diagrams")

    all_compounds = set(compound_bioactivity_summary["Metadata_Compound"])
    bioactive_compounds = set(compound_bioactivity_summary.loc[compound_bioactivity_summary["Bioactive"] == 1, "Metadata_Compound"])

    plt.figure(figsize=(8, 5))
    venn2([all_compounds, bioactive_compounds], set_labels=("All Compounds", f"Bioactive ({len(bioactive_compounds)})"))
    plt.title("Bioactivity Overview (All Plates)")
    venn_all_vs_bioactive = "venn_all_vs_bioactive.png"
    plt.savefig(venn_all_vs_bioactive)
    plt.close()
    upload_image_to_s3(bucket_name, f"{output_prefix}/venn_all_vs_bioactive.png", venn_all_vs_bioactive)

    # --- NEW: Generate Venn diagrams comparing each timepoint's actives to the total pool of actives ---
    logger.info("Generating per-timepoint Venn diagrams.")
    all_timepoints = compound_bioactivity_summary["Metadata_Timepoint"].unique()

    if not bioactive_compounds:
        logger.warning("No bioactive compounds found overall, skipping per-timepoint Venn diagrams.")
    else:
        for tp in all_timepoints:
            logger.info(f"Generating Venn diagram for timepoint: {tp}")
            
            timepoint_actives = set(
                compound_bioactivity_summary.loc[
                    (compound_bioactivity_summary["Metadata_Timepoint"] == tp) & 
                    (compound_bioactivity_summary["Bioactive"] == 1),
                    "Metadata_Compound"
                ]
            )
            
            plt.figure(figsize=(8, 6))
            bioactive_count = len(bioactive_compounds)
            percentage = int(len(timepoint_actives) / bioactive_count * 100) if bioactive_count > 0 else 0
            
            venn2(
                [bioactive_compounds, timepoint_actives],
                set_labels=("All Bioactive Compounds", f"Bioactive at {tp} ({percentage}%)")
            )
            plt.title(f"Bioactive Compounds at Timepoint {tp} vs. All Bioactive")

            sanitized_tp = str(tp).replace(" ", "_").replace("/", "_")
            venn_filename = f"venn_bioactive_vs_all_tp_{sanitized_tp}.png"
            plt.savefig(venn_filename, dpi=300)
            plt.close()
            upload_image_to_s3(bucket_name, f"{output_prefix}/{venn_filename}", venn_filename)

    # Heatmap generation is looped per plate
    logger.info("Performing heatmap generation per plate.")
    ind_mean["Metadata_Compound"] = ind_mean["Metadata_Compound"].str.upper()

    timepoint_order = sorted(
        ind_mean["Metadata_Timepoint"].unique(),
        key=extract_timepoint_numeric
    )
    
    unique_plates_heatmap = ind_mean["Metadata_Plate"].unique()

    for plate_id in unique_plates_heatmap:
        logger.info(f"Generating heatmap for plate: {plate_id}")
        
        plate_df = ind_mean[ind_mean["Metadata_Plate"] == plate_id].copy()

        plate_summary = (
            plate_df.groupby(["Metadata_Compound", "Metadata_Timepoint"])["Bioactive"]
            .max()
            .reset_index()
        )
        
        heatmap_data = plate_summary.pivot_table(
            index="Metadata_Compound",
            columns="Metadata_Timepoint",
            values="Bioactive",
            fill_value=0
        )

        heatmap_data = heatmap_data.reindex(columns=timepoint_order, fill_value=0)

        if heatmap_data.empty:
            logger.warning(f"No compound data for plate {plate_id}, skipping heatmap.")
            continue

        heatmap_data["Bioactive"] = (heatmap_data > 0).any(axis=1).astype(int)
        
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

        plt.title(f"Compound Bioactivity by Timepoint (Plate: {plate_id})", fontsize=12, pad=10)
        plt.xlabel("Timepoint")
        plt.ylabel("Compound")
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=6)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='black', label='Active', edgecolor='black')]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        plt.tight_layout()

        bioheat_img = f"compound_bioactivity_heatmap_plate_{plate_id}.png"
        plt.savefig(bioheat_img, dpi=300)
        plt.close()

        upload_image_to_s3(bucket_name, f"{output_prefix}/{bioheat_img}", bioheat_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bioactivity Analysis with per-plate normalization, Venn Diagrams, and Heatmaps.")
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