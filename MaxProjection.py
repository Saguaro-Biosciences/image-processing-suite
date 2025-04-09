import numpy as np
import imagej
import boto3
import io
import pandas as pd
import imageio
import argparse

def modify_imagepath(filepath):
    parts = filepath.split('/')
    if 'Images' in parts:
        images_index = parts.index('Images')
        parts[images_index] = 'ImagesStacked'
        return '/'.join(parts)
    else:
        return filepath

def max_projection(image_group, bucket_name, s3_client):
    images = []

    # Load each image from S3
    for image_key in image_group:
        response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
        image_data = response['Body'].read()
        image = imageio.imread(io.BytesIO(image_data))  # Load as NumPy array
        images.append(image)

    # Ensure all images have the same shape
    if not all(img.shape == images[0].shape for img in images):
        raise ValueError("All images must have the same dimensions.")

    # Compute max intensity projection
    max_projection = np.maximum.reduce(images)

    # Save as TIFF in-memory for S3 upload
    output_stream = io.BytesIO()
    imageio.imwrite(output_stream, max_projection, format='tiff')
    output_stream.seek(0)

    # Upload to S3
    output_key = modify_imagepath(image_group[0])
    s3_client.upload_fileobj(output_stream, bucket_name, output_key)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image plates using ImageJ and upload results to S3.")
    parser.add_argument("plate", type=str, help="Plate ID to process")
    args = parser.parse_args()

    plate = args.plate

    s3_client = boto3.client('s3')
    
    df = pd.read_csv("work/PfizerBucketImageMetadata.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='ISO8601')
    df = df.sort_values(by="Timestamp")
    df['Image_FileName'] = df['Tiff'].str.split('/').str[-1]
    df['Image_PathName'] = df['Tiff'].str.rsplit('/', n=1).str[0]
    df = df[['ChannelName', 'ChannelID', 'Image_FileName', 'Image_PathName', 'FieldID', 'PlaneID', 'PlateID', 'Row', 'Col', 'Timestamp']]

    sub = df[df.PlateID == plate]
    for i in range(0, len(sub), 12):
        chunk = sub.iloc[i: i + 12]
        for j in range(4):
            image_group = [
                chunk.iloc[j].Image_PathName + "/" + chunk.iloc[j].Image_FileName,
                chunk.iloc[j + 4].Image_PathName + "/" + chunk.iloc[j + 4].Image_FileName,
                chunk.iloc[j + 8].Image_PathName + "/" + chunk.iloc[j + 8].Image_FileName
            ]

            max_projection(image_group, "clientsdata", s3_client)

    print(f"Plate {plate} finished! Check Images in bucket")
