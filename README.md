# Image Processing Suite

This repository contains tools for processing image profiles straight from feature extraction. The tools are designed to run inside a Docker container in the AWS EC2 cloud environment and can also be launched locally.

## AWS EC2 Setup

This code allows you to launch an EC2 instance and run the image processing tools. Below is an example of how to use the `boto3` library to interact with AWS services and launch the EC2 instance:

```python
import boto3

response = ec2.run_instances(
    ImageId='ami-06c62cd5979834f62',  # Ubuntu core image + RAM monitor
    MinCount=1,
    MaxCount=1,  # Launch n instances
    InstanceType='r5a.2xlarge',
    IamInstanceProfile={
        'Name': "ssmmmanager"  # Specify the IAM role name
    },
    KeyName='',  # Replace with your key name
    SecurityGroupIds=['sg-00ad2c01ac4319690'],  # Replace with your security group ID
    SubnetId='subnet-0f63c8c61d8bbc468',
    BlockDeviceMappings=[{
        'DeviceName': '/dev/sda1',  # Default root device name
        'Ebs': {
            'VolumeSize': 120,  
            'VolumeType': 'gp3',  # General Purpose SSD
            'DeleteOnTermination': True  # Delete EBS volume when the instance is terminated
        }
    }],
    UserData="""#!/bin/bash
                # Update package list and install Docker
                sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/bin/config.json -s
                sudo apt-get update
                sudo apt-get install -y docker.io git

                # Clone the GitHub repo containing the Dockerfile and code
                git clone https://github.com/Saguaro-Biosciences/image-processing-suite.git /home/ubuntu/image-processing-suite

                # Navigate to the repo directory
                cd /home/ubuntu/image-processing-suite          
                git checkout dev
                
                # Build the Docker image
                sudo docker build -t image-processing-suite .

                # Run the Docker container
                sudo docker run -dit --name pycyto_container image-processing-suite
    """,  # This is the UserData script
    TagSpecifications=[
        {
            'ResourceType': 'instance',
            'Tags': [{'Key': 'Name', 'Value': "GitPycy5"}]
        }
    ]
)
```

## Running Commands on EC2 Using SSM

Once your EC2 instance is running, you can execute commands remotely using AWS Systems Manager (SSM). Here's an example template for running a Python script inside the Docker container on the EC2 instance:
```python
response = ssm.send_command(
    InstanceIds=[InstanceID],
    DocumentName="AWS-RunShellScript",  # Use shell script for Linux
    Parameters={'commands': ['sudo docker exec pycyto_container python3 Pycyto_pertime.py \
        --bucket_name cellprofiler-results \
        --base_folder IRIC/Progressive_Cpd_Dispensing_outputs \
        --times Progressive_Dispensing_1add_6h \
        --output_bucket cellprofiler-results \
        --output_prefix CQDM/Progressive_Dispensing/Add_1']},
    OutputS3BucketName="cellprofiler-results",
    OutputS3KeyPrefix="CQDM_Concat",
    CloudWatchOutputConfig={
        "CloudWatchLogGroupName": "/aws/ssm/MySSMCommandLogs",
        "CloudWatchOutputEnabled": True
    }
)
```

All of the codes relly in the general structure of the results where for a given project:

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
