# This script will take the raw data as downloaded from kaggle, etc. and
# organize it in the way our code expects it to be organized. The raw data archives should be
# tracked with git-lfs. As you add data, please update this script to move it into place.
set -e

# cd to the dir of this script
cd "$(dirname "$0")"

# MIT CVPR 2019
cd cvpr_raw
unzip ../archives/mit_cvpr_2019.zip
