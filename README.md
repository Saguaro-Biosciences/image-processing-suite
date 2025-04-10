# Image Processing Suite

This repository contains tools for processing image profiles straight from feature extraction. The tools are designed to run inside a Docker container in the AWS EC2 cloud environment and can also be launched locally.

## AWS EC2 Setup

This code allows you to launch an EC2 instance and run the image processing tools. Below is an example of how to use the `boto3` library to interact with AWS services and launch the EC2 instance:

```python
import boto3

# Initialize AWS clients
s3 = boto3.client('s3')
ec2 = boto3.client('ec2')
ssm = boto3.client('ssm')

instances = []

# Run EC2 instance
response = ec2.run_instances(
    ImageId='ami-053b0d53c279acc90',  # Ubuntu core image
    MinCount=1,
    MaxCount=1,  # Launch n instances
    InstanceType='t3.2xlarge',
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
                sudo apt-get update
                sudo apt-get install -y docker.io git

                # Clone the GitHub repo containing the Dockerfile and code
                git clone https://github.com/Saguaro-Biosciences/image-processing-suite.git /home/ubuntu/image-processing-suite

                # Navigate to the repo directory
                cd /home/ubuntu/image-processing-suite          

                # Build the Docker image
                sudo docker build -t image-processing-suite .

                # Run the Docker container
                sudo docker run -dit --name pycyto_container image-processing-suite
                sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/bin/config.json -s
    """,  # This is the UserData script
    TagSpecifications=[
        {
            'ResourceType': 'instance',
            'Tags': [{'Key': 'Name', 'Value': "GitPycy "}]
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
