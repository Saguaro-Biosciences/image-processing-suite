# 1. Removed unused imports (numpy, pandas, etc.)
from PIL import Image
import boto3
import io
import argparse
import logging

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_image_in_memory(image_bytes, target_size=(1080, 1080)):
    """
    Reads image data from bytes, resizes it, and returns the compressed
    bytes of the new TIFF image.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        binned_img = img.resize(target_size, resample=Image.Resampling.LANCZOS)
        output_buffer = io.BytesIO()
        # Using 'tiff' as it's the more standard format name
        binned_img.save(output_buffer, format='tiff', compression='tiff_lzw')
        return output_buffer.getvalue()

def process_images_in_s3(bucket_name, image_folder, resolution):
    """
    Finds images in an S3 folder, processes them in memory, and saves
    them to another folder in the same bucket.
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    logger.info(f"🚀 Starting image processing for bucket: '{bucket_name}'")
    logger.info(f"📂 Source: '{image_folder}' -> Destination: '{image_folder.replace('Image', 'Image_binned')}'")

    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    processed_count = 0

    for obj in bucket.objects.filter(Prefix=image_folder):
        if obj.key.endswith('/') or not obj.key.lower().endswith(valid_extensions):
            continue

        # 3. (IMPROVEMENT) Use the logger
        logger.info(f"Processing 's3://{bucket_name}/{obj.key}'...")
        try:
            image_data = obj.get()['Body'].read()
            
            processed_data = process_image_in_memory(image_data, target_size=(resolution, resolution))
            
            # Using a more robust replacement that only acts on the prefix
            new_key = obj.key.replace('Image', 'Image_binned')
            
            bucket.put_object(Key=new_key, Body=processed_data, ContentType='image/tiff')
            
            logger.info(f"✅ Success! Saved to 's3://{bucket_name}/{new_key}'")
            processed_count += 1

        except Exception as e:
            # 3. (IMPROVEMENT) Use logger.error with exc_info for full traceback
            logger.error(f"❌ Failed to process '{obj.key}'", exc_info=True)
            
    logger.info(f"\n✨ Done! Processed {processed_count} images.")

# --- Run the script ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and re-bin images from an S3 folder.")
    parser.add_argument("--bucket_name", type=str, required=True, help="S3 bucket containing the files.")
    parser.add_argument("--image_folder", type=str, required=True, help="Source folder path in S3 (e.g., 'path/to/experiment/Image/').")
    parser.add_argument("--resolution", type=int, default=1080, required=False, help="Target resolution for the square image (e.g., 1080).")

    args = parser.parse_args()
            
    # 4. (FIX) Use the correct argument name: args.image_folder
    logger.info(f"Starting re-binning for image folder: {args.image_folder}")

    process_images_in_s3(
        bucket_name=args.bucket_name,
        image_folder=args.image_folder,
        resolution=args.resolution
    )