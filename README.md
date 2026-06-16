# Image Processing Suite

> Part of the **Saguaro Biosciences** cell-profiling platform.

A collection of tools for processing high-content phenotypic-profiling microscopy images: illumination quality control, Cellpose-based segmentation, and morphological feature / embedding extraction. The suite is packaged to run inside a Docker container, deployed either on an on-prem server (**P620**) or in the AWS EC2 cloud.

These tools are the compute backend behind the [`lembeddingscellprofileling`](https://github.com/Saguaro-Biosciences/lembeddingscellprofileling) Nextflow/WDL pipeline, which orchestrates them per plate and timepoint.

> [!IMPORTANT]
> **Proprietary software.** This repository and its associated code are the property of **Saguaro Biosciences** and are intended for internal use only. All rights reserved. It may not be copied, redistributed, published, or disclosed outside Saguaro Biosciences without prior written authorisation.

## Repository contents

### Used by the `lembeddingscellprofileling` pipeline

| Script | Purpose |
| ------ | ------- |
| [`Illumination_QC_mult.py`](Illumination_QC_mult.py) | CellProfiler-matched, image-level quality control (production). |
| [`qc_report_annotation.py`](qc_report_annotation.py) | Generates QC plots and reports from results stored in S3. |
| [`Cellpose_GPU_s3fs.py`](Cellpose_GPU_s3fs.py) | GPU Cellpose analysis: image-level QC, XGBoost dead-cell assessment, and single-cell / well-level embedding extraction. |

### Segmentation & image preparation

| Script | Purpose |
| ------ | ------- |
| [`Cellpose_GPU_s3fs_export_tiffs.py`](Cellpose_GPU_s3fs_export_tiffs.py) | Segment cell-profiling images with Cellpose and export masked per-channel TIFFs. |
| [`MaxProjection.py`](MaxProjection.py) | Maximum-intensity projection of image plates via ImageJ, with results uploaded to S3. |
| [`Image_re-binning.py`](Image_re-binning.py) | Re-bin images from an S3 folder. |
| [`mask_printer.py`](mask_printer.py) | Cellpose analysis variant for rendering / exporting segmentation masks. |

### Feature processing & analysis

| Script | Purpose |
| ------ | ------- |
| [`Feature_extraction_opt.py`](Feature_extraction_opt.py) | Morphological feature extraction. |
| [`ConcatCP_exp.py`](ConcatCP_exp.py) | Concatenate CSV outputs from S3 across multiple plates. |
| [`Pycyto_pertime.py`](Pycyto_pertime.py) | Concatenate CSV outputs from S3 across plates, per timepoint. |
| [`Normalize_CP_ami.py`](Normalize_CP_ami.py) | Normalize each timepoint of a project folder against DMSO controls. |
| [`Feature_select_cosine_ami.py`](Feature_select_cosine_ami.py) | Concatenate, normalize, and select CellProfiler features from S3. |
| [`Induction_Score_hit.py`](Induction_Score_hit.py) | Bioactivity analysis with per-plate normalization, Venn diagrams, and heatmaps. |

## Environment

The suite is built on `python:3.10` (see [`Dockerfile`](Dockerfile)); Python dependencies are pinned in [`requirements.txt`](requirements.txt) (boto3, pandas, numpy, pycytominer, scikit-learn, umap-learn, matplotlib / seaborn / plotly, imageio, and others).

Build and run the container locally:

```bash
docker build -t image-processing-suite .
docker run -dit --name pycyto_container image-processing-suite
```

## AWS EC2 setup

The tools can be run on a freshly launched EC2 instance. The example below uses `boto3` to launch an instance that, via its `UserData` script, installs Docker, clones this repository, builds the image, and starts the container:

```python
import boto3

ec2 = boto3.client("ec2")

response = ec2.run_instances(
    ImageId='<ami-id>',  # e.g. an Ubuntu core image + RAM monitor
    MinCount=1,
    MaxCount=1,  # Launch n instances
    InstanceType='r5a.2xlarge',
    IamInstanceProfile={
        'Name': "<iam-role-name>"  # IAM role with SSM access
    },
    KeyName='',  # Replace with your key name
    SecurityGroupIds=['<security-group-id>'],  # Replace with your security group ID
    SubnetId='<subnet-id>',  # Replace with your subnet ID
    BlockDeviceMappings=[{
        'DeviceName': '/dev/sda1',  # Default root device name
        'Ebs': {
            'VolumeSize': 120,
            'VolumeType': 'gp3',  # General Purpose SSD
            'DeleteOnTermination': True  # Delete EBS volume when the instance is terminated
        }
    }],
    UserData="""#!/bin/bash
                # Configure the CloudWatch agent, then install Docker
                sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/bin/config.json -s
                sudo apt-get update
                sudo apt-get install -y docker.io git

                # Clone this repository
                git clone https://github.com/Saguaro-Biosciences/image-processing-suite.git /home/ubuntu/image-processing-suite
                cd /home/ubuntu/image-processing-suite
                git checkout dev

                # Build and run the container
                sudo docker build -t image-processing-suite .
                sudo docker run -dit --name pycyto_container image-processing-suite
    """,
    TagSpecifications=[
        {
            'ResourceType': 'instance',
            'Tags': [{'Key': 'Name', 'Value': "GitPycy5"}]
        }
    ]
)
```

## Running commands on EC2 via SSM

Once the instance is running, you can execute commands remotely with AWS Systems Manager (SSM). The example below runs a Python script inside the Docker container on the instance:

```python
response = ssm.send_command(
    InstanceIds=[InstanceID],
    DocumentName="AWS-RunShellScript",  # Shell script for Linux
    Parameters={'commands': ['sudo docker exec pycyto_container python3 Pycyto_pertime.py \
        --bucket_name  \
        --base_folder  \
        --times  \
        --output_bucket  \
        --output_prefix ']},
    OutputS3BucketName="",
    OutputS3KeyPrefix="",
    CloudWatchOutputConfig={
        "CloudWatchLogGroupName": "/aws/ssm/MySSMCommandLogs",
        "CloudWatchOutputEnabled": True
    }
)
```

## Expected storage layout

The tools assume a consistent folder structure for each project, organised by project → run → plate → timepoint:

```
📦 bucket/
├── 📁 project_1/
│   ├── 📁 Plate_1/
│   │   ├── 📁 0h/
│   │   │   └── results.csv
│   │   ├── 📁 12h/
│   │   └── 📁 24h/
│   └── 📁 Plate_2/
├── 📁 project_2/
...
```

A concrete example:

```
📦 bucket/
└── 📁 Phenotypic_screen_HY-L022-custom_U2OS/
    ├── 📁 Subset1_10uM_run1/
    │   ├── 📁 Plate_1/
    │   │   ├── 📁 6h/
    │   │   │   ├── Image.csv
    │   │   │   ├── CH_illumination.csv
    │   │   │   └── results_6h.csv
    │   │   ├── 📁 12h/
    │   │   ├── 📁 24h/
    │   │   └── 📁 48h/
    │   └── 📁 Plate_2/
    ├── 📁 Subset1_10uM_run2/
    └── 📁 Subset1_10uM_run3/
...
```
