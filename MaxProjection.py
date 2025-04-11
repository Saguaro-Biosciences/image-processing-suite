import numpy as np
import boto3
import io
import pandas as pd
import imageio
import argparse
import csv
import posixpath
import logging
from io import StringIO

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def modify_imagepath(filepath):
    parts = filepath.split('/')
    if 'Images' not in parts:
        return filepath
    images_index = parts.index('Images')
    parts[images_index] = 'ImagesStacked'
    return '/'.join(parts)

def read_csv_from_s3(bucket_name, file_key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response['Body'].read().decode('utf-8')

    sample = csv_content[:1024]
    dialect = csv.Sniffer().sniff(sample, delimiters=";,")
    return pd.read_csv(StringIO(csv_content), sep=dialect.delimiter)

def max_projection(image_group, bucket_name, s3_client):
    images = []

    for image_key in image_group:
        response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
        image_data = response['Body'].read()
        image = imageio.imread(io.BytesIO(image_data))
        images.append(image)

    if not all(img.shape == images[0].shape for img in images):
        raise ValueError(f"Image shape mismatch in group: {image_group}")

    max_proj = np.maximum.reduce(images)

    output_stream = io.BytesIO()
    imageio.imwrite(output_stream, max_proj, format='tiff')
    output_stream.seek(0)

    output_key = modify_imagepath(image_group[0])
    s3_client.upload_fileobj(output_stream, bucket_name, output_key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image plates using ImageJ and upload results to S3.")
    parser.add_argument("--bucket_data_set",type=str, required=True, help="S3 bucket containing the data set.")
    parser.add_argument("--data_set", type=str, required=True,
                        help="Data set key location containing per PlateID images to process containing 'ChannelName', 'ChannelID', 'Image_FileName', 'Image_PathName', 'FieldID', 'PlaneID', 'PlateID', 'Row', 'Col', 'Timestamp'.")
    parser.add_argument("--channels", type=int, required=True, help="Number of channels per group")
    parser.add_argument("--planes", type=int, required=True, help="Number of planes per channel")
    parser.add_argument("--bucket_images", type=str,required=True, help="S3 bucket containing the raw images to max project.")
    args = parser.parse_args()

    bucket_data_set = args.bucket_data_set
    data_set = args.data_set
    num_channels = args.channels
    num_planes = args.planes
    bucket_images = args.bucket_images
    group_size = num_channels * num_planes

    s3_client = boto3.client('s3')
    
    df = read_csv_from_s3(bucket_data_set, data_set)

    for plate in df['PlateID'].unique():
        sub = df[df['PlateID'] == plate]

        for i in range(0, len(sub), group_size):
            chunk = sub.iloc[i: i + group_size]
            if len(chunk) < group_size:
                logger.warning(f"Skipping incomplete chunk in plate {plate} at index {i}")
                continue

            for j in range(num_channels):
                try:
                    image_group = [
                        posixpath.join(chunk.iloc[j + (p * num_channels)].Image_PathName,
                                       chunk.iloc[j + (p * num_channels)].Image_FileName)
                        for p in range(num_planes)
                    ]
                    max_projection(image_group, bucket_images, s3_client)
                except Exception as e:
                    logger.error(f"Error processing group {j} in chunk starting at {i} for plate {plate}: {e}")

        logger.info(f"Plate {plate} finished! Check images in bucket.")
