# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to create temp directory' >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

rsync -ah --stats /cluster/scratch/$USER/wachterberg-street65.tar.gz $TMPDIR
tar -xzf wachterberg-street65.tar.gz

echo "Data copied at:     $(date)"

# Binary or script to execute
# load modules
module load gcc/8.2.0 python/3.11.2 cuda/11.8.0 cudnn/8.8.1.3 eth_proxy curl jq

echo "Dependencies installed"

cd $HOME/road_segmentation