from pathlib import Path
import os
import time
import logging

import boto3
import papermill as pm
import watchtower

from package import config, utils


if __name__ == "__main__":

    run_on_start = False if config.TEST_OUTPUTS_S3_BUCKET == "" else True

    if not run_on_start:
        exit()

    cfn_client = boto3.client('cloudformation', region_name=config.AWS_REGION)

    # Set up logging through watchtower
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    log_group = "/aws/sagemaker/NotebookInstances"
    stream_name = "{}/run-notebook.log".format(utils.get_notebook_name())
    logger.addHandler(
        watchtower.CloudWatchLogHandler(log_group=log_group, stream_name=stream_name))
    # Add papermill logging to CloudWatch as well
    pm_logger = logging.getLogger('papermill')
    pm_logger.addHandler(
        watchtower.CloudWatchLogHandler(log_group=log_group, stream_name=stream_name))

    # Wait for the stack to finish launching
    logger.info("Waiting for stack to finish launching...")
    waiter = cfn_client.get_waiter('stack_create_complete')

    waiter.wait(StackName=config.STACK_NAME)

    logger.info("Starting notebook execution through papermill")

    # Run the notebook
    bucket = config.TEST_OUTPUTS_S3_BUCKET


    solution_notebooks = [
            "1.Data_Privatization",
            "2.Model_Training"
            ]
    kernel_name = 'python3'
    test_prefix = "/home/ec2-user/SageMaker/test/"
    notebooks_directory = '/home/ec2-user/SageMaker/sagemaker/'

    for notebook_name in solution_notebooks:
        start_time = time.time()
        stdout_path = os.path.join(test_prefix, "{}-output_stdout.txt".format(notebook_name))
        stderr_path = os.path.join(test_prefix, "{}-output_stderr.txt".format(notebook_name))
        with open(stdout_path, 'w') as stdoutfile, open(stderr_path, 'w') as stderrfile:
            output_notebook_path = "{}-output.ipynb".format(os.path.join(test_prefix, notebook_name))
            try:
                nb = pm.execute_notebook(
                    "{}.ipynb".format(os.path.join(notebooks_directory, notebook_name)),
                    output_notebook_path,
                    cwd=notebooks_directory,
                    kernel_name=kernel_name,
                    stdout_file=stdoutfile, stderr_file=stderrfile, log_output=True
                )
            except pm.PapermillExecutionError as err:
                logger.warn("Notebook {} encountered execution error: {}".format(notebook_name, err))
                raise
            finally:
                end_time = time.time()
                logger.info("Notebook {} execution time: {} sec.".format(notebook_name, end_time - start_time))
                s3 = boto3.resource('s3')
                # Upload notebook output file to S3
                s3.meta.client.upload_file(output_notebook_path, bucket, Path(output_notebook_path).name)
                s3.meta.client.upload_file(stdout_path, bucket, Path(stdout_path).name)
                s3.meta.client.upload_file(stderr_path, bucket, Path(stderr_path).name)
