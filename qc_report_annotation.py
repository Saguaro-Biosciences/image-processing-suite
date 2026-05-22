import argparse
import boto3
import pandas as pd
import numpy as np
import logging
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os

def read_csv_from_s3(bucket_name, key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=key)
    csv_content = response['Body'].read().decode('utf-8')
    return pd.read_csv(StringIO(csv_content))

def main():
    parser = argparse.ArgumentParser(description="Generate QC Plots from S3 Data")
    parser.add_argument('--work_path', required=True, help="Base path in S3")
    parser.add_argument('--plate', required=True, help="Plate identifier")
    parser.add_argument('--time', required=True, help="Timepoint identifier")
    args = parser.parse_args()

    s3 = boto3.client('s3')
    bucket_name = "cellprofiler-resuts"
    
    # Construct the S3 key exactly based on Nextflow inputs
    features_key = f"{args.work_path}/{args.plate}/{args.time}/Image.csv"
    
    print(f"📦 Processing: {args.time} {args.plate}")
    
    try:
        df = read_csv_from_s3(bucket_name, features_key)
        print(f"Loaded {features_key}")
    except Exception as e:
        print(f" Error loading {features_key}: {e}")
        return

    # QC Feature settings
    feature_groups = {
        'ImageQuality_Power': {'color': 'orange', 'threshold': 'iqr'},
        'ImageQuality_PercentMax': {'color': 'blue', 'threshold': 'fixed', 'fixed_thresh': 0.001}
    }

    all_traces = []
    all_titles = []
    
    for group_prefix, settings in feature_groups.items():
        matching_columns = [col for col in df.columns if col.startswith(group_prefix)]
        for feature_col in matching_columns:
            values = df[feature_col].dropna()
            wells = df.loc[values.index, 'Metadata_Well']
            site = df.loc[values.index, 'Metadata_Site'].astype(str)

            lower_thresh = None
            upper_thresh = None

            # Threshold logic
            if 'threshold' in settings:
                if settings['threshold'] == 'iqr':
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_thresh = Q1 - 1.5 * IQR
                    upper_thresh = Q3 + 1.5 * IQR
                    fail_mask = (df[feature_col] < lower_thresh) | (df[feature_col] > upper_thresh)
                else:
                    lower_thresh = settings['fixed_thresh']
                    upper_thresh = None
                    fail_mask = df[feature_col] >= lower_thresh

                # Save QC column
                qc_col = f"ImageQC_{feature_col}"
                df[qc_col] = fail_mask.where(~df[feature_col].isna(), np.nan)

            # Plotting logic
            histogram = go.Histogram(x=values, nbinsx=100, marker_color=settings['color'], opacity=0.5, showlegend=False)
            all_traces.append([histogram])

            scatter = go.Scatter(
                x=values, y=[0.1] * len(values), mode='markers',
                marker=dict(color='lightgrey'), customdata=np.stack([wells + '_S' + site], axis=-1),
                hovertemplate="Well: %{customdata[0]}<br>Value: %{x:.2f}<extra></extra>", showlegend=False
            )
            all_traces[-1].append(scatter)

            if upper_thresh is not None:
                all_traces[-1].append(go.Scatter(x=[upper_thresh, upper_thresh], y=[0, 100], mode='lines', line=dict(color='red', width=2, dash='dot'), showlegend=False))
            if lower_thresh is not None and group_prefix != 'ImageQuality_PercentMax':
                all_traces[-1].append(go.Scatter(x=[lower_thresh, lower_thresh], y=[0, 100], mode='lines', line=dict(color='red', width=2, dash='dot'), showlegend=False))

            all_titles.append(f"{feature_col} / {args.time}")

    if not all_traces:
        print("No matching feature columns found to plot.")
        return

    # Create subplot
    fig = make_subplots(rows=len(all_traces), cols=1, subplot_titles=all_titles)
    for i, trace_group in enumerate(all_traces):
        for trace in trace_group:
            fig.add_trace(trace, row=i + 1, col=1)

    fig.update_layout(height=430 * len(all_traces), width=800, title_text=f"Feature Distributions with QC for {args.time}", template='simple_white', barmode='overlay')

    # Save HTML to the CURRENT WORKING DIRECTORY so Nextflow captures it
    html_filename = f"{args.work_path}/{args.plate}/{args.time}/qc_plots_{args.plate}_{args.time}.html"
    pio.write_html(fig, file=html_filename, auto_open=False)
    print(f"💾 HTML saved: {html_filename}")

    # Save modified CSV back to S3
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key=features_key, Body=csv_buffer.getvalue())
    print(f"Updated CSV with QC columns uploaded: {features_key}")

if __name__ == "__main__":
    main()