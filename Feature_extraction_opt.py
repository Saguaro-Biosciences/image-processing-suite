import boto3
import time
import math
import pandas as pd
import numpy as np
import boto3
import re
import string
from io import StringIO
import time
import os
import tifffile

# Function to convert position number to well position
def position_to_well(pos):
    row = string.ascii_uppercase[(pos - 1) // 24]  # Row letter
    col = (pos - 1) % 24 + 1                       # Column number
    return f"{row}{col:02}"  


def list_dmso_main_folders(bucket_name, prefix,tokens):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')

    folders = set()

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            path_parts = obj['Key'].strip('/').split('/')
            for i in range(1, len(path_parts)):  # Iterate through all subfolder levels
                subfolder = '/'.join(path_parts[:i]) + '/'
                if any(token in path_parts[i-1] for token in tokens):  # Stop at the first match
                    folders.add(subfolder) 
                    break  # Prevent capturing deeper nested folders

    return sorted(folders)

import string

def row_col_to_well(row_num, col_num):
    row_letter = string.ascii_uppercase[row_num - 1]
    return f"{row_letter}{col_num:02}"

# Example usage
bucket_name = 'iric'
folder='Subset1_10uM_Run03'

# --- 1. CONFIGURATION ---
# --- Fill in these values for your environment ---
AMI_ID = "ami-014dc5a064d0ab846"  # An AMI with Docker, AWS CLI, and SSM Agent installed
KEY_NAME = "dcamacho_laptop"   # Your EC2 Key Pair for SSH access (optional but good practice)
IAM_PROFILE_ARN = "arn:aws:iam::373342583720:instance-profile/ssmmmanager" # ARN of the IAM role
SUBNET_ID = "subnet-0f63c8c61d8bbc468" # A subnet for the instance to launch into
SECURITY_GROUP_IDS = ['sg-00ad2c01ac4319690'] # Security group(s) for the instance

BATCH_SIZE = 10 # Process 10 plates at a time per instance

# --- Job and Path Configuration ---
FOLDER = 'Subset1_10uM_Run03'
CCPIPE_NAME = 'Feature_Extraction_CL2.0.cppipe'
S3_PIPELINES_PATH = "s3://cellprofiler-resuts/IRIC/Image_Processing_Pipelines"
S3_BASE_OUTPUT_PATH = f"s3://cellprofiler-resuts/IRIC/Phenotypic_screen_HY-L022-custom_U2OS/{FOLDER}"
PLATES_TO_RUN = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06']
prefix = f'Phenotypic_screen_HY-L022-custom_U2OS/{FOLDER}/'
prefixes = list_dmso_main_folders(bucket_name, prefix,PLATES_TO_RUN)

TIMES_TO_RUN = ['6', '12', '24', '48']

# --- Boto3 Clients ---
ec2 = boto3.client('ec2')
ssm = boto3.client('ssm')

def run_batch_processing():
    # --- 2. Generate and Batch All Jobs ---
    all_jobs = [(plate, time) for plate in PLATES_TO_RUN for time in TIMES_TO_RUN]
    batches = [all_jobs[i:i + BATCH_SIZE] for i in range(0, len(all_jobs), BATCH_SIZE)]
    print(f"Generated {len(all_jobs)} total jobs, split into {len(batches)} batches of {BATCH_SIZE}.")

    for i, batch in enumerate(batches):
        print(f"\n--- Starting Batch {i+1} of {len(batches)} with {len(batch)} jobs ---")
        if len(batch)<5:
            INSTANCE_TYPE = "r6a.4xlarge"    # An instance that can handle 10 parallel jobs (~10x15GB RAM needed)
        else:
            INSTANCE_TYPE = "r6a.8xlarge"     # An instance that can handle 10 parallel jobs (~10x15GB RAM needed)

        # --- 3. Launch a New EC2 Instance for This Batch ---
        print(f"Launching a new EC2 instance... {INSTANCE_TYPE}")
        instance_response = ec2.run_instances(
            ImageId=AMI_ID,
            InstanceType=INSTANCE_TYPE,
            MinCount=1,
            MaxCount=1,
            IamInstanceProfile={'Arn': IAM_PROFILE_ARN},
            KeyName=KEY_NAME,
            SubnetId=SUBNET_ID,
            UserData="""#!/bin/bash
                sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/bin/config.json -s""",
            SecurityGroupIds=SECURITY_GROUP_IDS,
            InstanceInitiatedShutdownBehavior='terminate', # This is crucial!
            TagSpecifications=[{'ResourceType': 'instance', 'Tags': [{'Key': 'Name', 'Value': f'CellProfiler-Batch-{i+1}'}]}],
            BlockDeviceMappings=[{
                'DeviceName': '/dev/sda1',  # Default root device name
                'Ebs': {
                    'VolumeSize': 120 * BATCH_SIZE,  
                    'VolumeType': 'gp3',
                    'Iops': 16000,
                    'Throughput': 500,  # General Purpose SSD
                    'DeleteOnTermination': True  # Delete EBS volume when the instance is terminated
                }}]
        )
        instance_id = instance_response['Instances'][0]['InstanceId']
        print(f"Instance {instance_id} launched for Batch {i+1}.")

        # --- 4. Wait for the Instance and SSM Agent to be Ready ---
        print(f"Waiting for instance {instance_id} to be ready...")
        while True:
            try:
                ssm_info = ssm.describe_instance_information(
                        InstanceInformationFilterList=[
                            {'key': 'InstanceIds', 'valueSet': [instance_id]}
                        ]
                    )
                if ssm_info['InstanceInformationList']:
                    print(f"Instance {instance_id} is online and ready for commands.")
                    break
            except Exception as e:
                pass # Ignore errors while waiting
            time.sleep(120) # Wait 15 seconds before checking again

        # --- 5. Build and Send the Parallel Command Script ---
        master_command_list = build_ssm_command(batch,FOLDER,prefixes)
        
        print(f"Sending batch command to instance {instance_id}...")
        response = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={'commands': master_command_list},
            OutputS3BucketName="cellprofiler-resuts",
            OutputS3KeyPrefix=f"Batch_Run_Logs/Batch_{i+1}",
            CloudWatchOutputConfig={
                "CloudWatchLogGroupName": "/aws/ssm/MySSMCommandLogs",
                "CloudWatchOutputEnabled": True
            })
        command_id = response['Command']['CommandId']
        print(f"Command {command_id} sent successfully to instance {instance_id}.")

def build_ssm_command(batch_jobs,FOLDER,prefixes):
    # This function builds the specific shell script for a given batch of jobs
    commands = ['#!/bin/bash', 'set -e', 'echo "--- SCRIPT STARTED ---"']
    # Example usage
    
    for plate, time in batch_jobs:
        image_folder=list(filter(lambda x:f'{time}h_{plate}' in x, prefixes))[0].split('/')[-2]
        s3_image_path = f"s3://iric/Phenotypic_screen_HY-L022-custom_U2OS/{FOLDER}/{image_folder}/Image/"
        s3_csv_path = f"{S3_BASE_OUTPUT_PATH}/load_data_{plate}_{time}_illum.csv"
        s3_cppipe_path = f"{S3_PIPELINES_PATH}/{CCPIPE_NAME}"
        local_dir = f"/data/{plate}_{time}h"

        command_block = f"""
echo "[JOB: {plate}_{time}h] Staging data..."
mkdir -p {local_dir}/input {local_dir}/output
aws s3 sync {s3_image_path} {local_dir}/input/
aws s3 cp {s3_csv_path} {local_dir}/input/
aws s3 cp {s3_cppipe_path} {local_dir}/input/
echo "[JOB: {plate}_{time}h] Starting analysis in background..."
sudo docker run --rm -v {local_dir}/input:/mnt/input -v {local_dir}/output:/mnt/output cellprofiler:4.2.8 \
-c -r -p /mnt/input/{CCPIPE_NAME} -o /mnt/output/ --data-file /mnt/input/load_data_{plate}_{time}_illum.csv &
"""
        commands.append(command_block)

    commands.extend(['echo "WAITING FOR ALL JOBS TO FINISH..."', 'wait'])
    
    commands.append('echo "SYNCING ALL RESULTS..."')
    for plate, time in batch_jobs:
        local_output_dir = f"{local_dir}/output" # The local_dir variable from the last iteration will be incorrect here
        local_output_dir = f"/data/{plate}_{time}h/output" # Corrected path
        s3_final_output_path = f"{S3_BASE_OUTPUT_PATH}/{plate}/{time}"
        commands.append(f"aws s3 sync {local_output_dir} {s3_final_output_path}")

    commands.extend(['echo "JOB COMPLETE. INSTANCE WILL TERMINATE IN 1 MINUTE."', 'sudo shutdown -h +1'])
    return commands

if __name__ == "__main__":
    run_batch_processing()