# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/
import json
from pathlib import Path

from package import utils

current_folder = utils.get_current_folder(globals())
cfn_stack_outputs_filepath = Path(current_folder, '../../../stack_outputs.json').resolve()
assert cfn_stack_outputs_filepath.exists(), "Could not find stack_outputs.json file at {}".format(
    str(cfn_stack_outputs_filepath))

with open(cfn_stack_outputs_filepath) as f:
    cfn_stack_outputs = json.load(f)

AWS_ACCOUNT_ID = cfn_stack_outputs['AwsAccountId']
AWS_REGION = cfn_stack_outputs['AwsRegion']
STACK_NAME = cfn_stack_outputs['StackName']
SOLUTIONS_S3_BUCKET = cfn_stack_outputs['SolutionsS3Bucket']

PRIVACY_SAGEMAKER_IAM_ROLE = cfn_stack_outputs['IamRole']
S3_BUCKET = cfn_stack_outputs['ModelDataBucket']
SAGEMAKER_MODE = cfn_stack_outputs['SagemakerMode']
SAGEMAKER_PROCESSING_JOB_CONTAINER_BUILD = cfn_stack_outputs['SageMakerProcessingJobContainerBuild']
SAGEMAKER_PROCESSING_JOB_CONTAINER_NAME = cfn_stack_outputs['SageMakerProcessingJobContainerName']
TEST_OUTPUTS_S3_BUCKET = cfn_stack_outputs.get('TestOutputsS3Bucket', "")
SOLUTION_NAME = cfn_stack_outputs['SolutionName']
SOLUTION_PREFIX = cfn_stack_outputs['SolutionPrefix']
TRAINING_INSTANCE_TYPE = cfn_stack_outputs['TrainingInstanceType']
PROCESSING_INSTANCE_TYPE = cfn_stack_outputs['ProcessingInstanceType']
