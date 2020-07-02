#!/bin/bash

set -e

if [ "$1" == "-h" ]; then
  echo "Usage: ./build.sh <trademarked-solution-name> <destination-s3-bucket> <region> [<branch-name>]"
  echo "The 'branch-name' is meant to be used to have multiple versions of the solution under the same bucket."
  echo "The solution will be uploaded to S3 at <destination-s3-bucket>-<region>/<trademarked-solution-name>[-<branch-name>]"
  echo "If 'mainline' is provided as the 'branch-name', no suffix is added to the S3 destination."
  exit 0
fi

# Check to see if input has been provided:
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Please provide the solution name as well as the base S3 bucket name and the region to run build script."
    echo "For example: ./build.sh trademarked-solution-name sagemaker-solutions-build us-west-2"
    exit 1
fi

if [ -z "$4" ] || [ "$4" == 'mainline' ]; then
    s3_prefix="s3://$2-$3/$1"
else
    s3_prefix="s3://$2-$3/$1-$4"
fi

base_dir="$PWD"
rm -rf build
mkdir -p build

cd $base_dir

# Prep solution assistant lambda environment
echo "Installing local requirements for solution assistant lambda function"
cd "$base_dir" || exit
cd ./deployment/solution-assistant && pip install -r requirements.txt -t ./src/site-packages
# Clean up pyc files, needed to avoid security issues. See: https://blog.jse.li/posts/pyc/
cd "$base_dir" || exit
find ./deployment/ -type f -name "*.pyc" -delete
find ./deployment/ -type d -name "__pycache__" -delete

# Package solution assistant lambda
cd "$base_dir" || exit
echo "Copying and packaging solution assistant lambda function"
cp -r ./deployment/solution-assistant/src build/
cd build/src || exit
zip -q -r9 "$base_dir"/build/solution_assistant.zip -- *

echo "Packaging data privatization code build source"
cd $base_dir
cp source/sagemaker/src/package/model/train.py source/sagemaker/src/package/data_privatization/container/
cp -r source/sagemaker/src/package/data_privatization/ build/sagemaker-data-privatization/
cd build/sagemaker-data-privatization/
zip -q -r9 $base_dir/build/sagemaker_data_privatization.zip *

cd $base_dir

rm -rf build/sagemaker-data-privatization/
rm -rf build/src

# Run cfn-nag and viperlight scan locally if available, but don't cause build failures
{
    if command -v cfn_nag &> /dev/null
    then
        echo "Running cfn_nag scan"
        for y in `find ./deployment/* -name "*.yaml"`;  do
            echo "============= $y ================" ;
            cfn_nag --fail-on-warnings $y || ec1=$?;
        done
    fi

    if command -v viperlight &> /dev/null
    then
        echo "Running viperlight scan"
        viperlight scan
    fi
} || true

echo "Clearing existing objects under $s3_prefix"
# aws s3 rm --recursive $s3_prefix
echo "Copying artifacts objects to $s3_prefix"
aws s3 sync s3://sagemaker-solutions-artifacts/sagemaker-privacy-for-nlp/ "${s3_prefix}/" --delete --source-region us-west-2 --region $3
echo "Copying source objects to $s3_prefix"
aws s3 sync --delete --source-region us-west-2 --region $3 build $s3_prefix/build
aws s3 sync --delete --source-region us-west-2 --region $3 metadata $s3_prefix/metadata
aws s3 sync --delete --source-region us-west-2 --region $3 deployment $s3_prefix/deployment
aws s3 sync --delete --source-region us-west-2 --region $3 docs $s3_prefix/docs
aws s3 sync --delete --source-region us-west-2 --region $3 source $s3_prefix/source --exclude "sagemaker/.vector_cache/*" \
  --exclude "sagemaker/.ipynb_checkpoints/*"
aws s3 sync --delete --source-region us-west-2 --region $3 test "$s3_prefix"/test
echo "Build sucessfull!"
