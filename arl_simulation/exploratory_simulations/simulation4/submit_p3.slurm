#!/bin/bash
#!
#! Dask job script for Darwin (Sandy Bridge, ConnectX3)
#! Tim Cornwell
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J ESIM4
#! Which project should be charged:
#SBATCH -A SKA-SDP
#! How many whole nodes should be allocated?
#SBATCH --nodes=16
#! How many (MPI) tasks will there be in total? (<= nodes*16)
#SBATCH --ntasks=61
#! Memory limit: P3 has roughly 107GB per node
#SBATCH --mem 26000
#! How much wallclock time will be required?
#SBATCH --time=23:59:59
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#! Do not change:
#SBATCH -p compute
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Modify the settings below to specify the application's environment, location
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
module purge                               # Removes all modules still loaded

#! Set up python
. $HOME/alaska-venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$ARL
echo "PYTHONPATH is ${PYTHONPATH}"

echo -e "Running python: `which python`"
echo -e "Running dask-scheduler: `which dask-scheduler`"

cd $SLURM_SUBMIT_DIR
echo -e "Changed directory to `pwd`.\n"

JOBID=${SLURM_JOB_ID}
echo ${SLURM_JOB_NODELIST}

#! Create a hostfile:
scontrol show hostnames $SLURM_JOB_NODELIST | uniq > hostfile.$JOBID

scheduler=$(head -1 hostfile.$JOBID)
hostIndex=0
for host in `cat hostfile.$JOBID`; do
    echo "Working on $host ...."
    if [ "$hostIndex" = "0" ]; then
        echo "run dask-scheduler"
        ssh $host dask-scheduler --port=8786 &
        sleep 5
    fi
    echo "run dask-worker"
    ssh $host dask-worker --host ${host} --nprocs 4 --nthreads 1  \
     --memory-limit 25GB  $scheduler:8786  &
        sleep 1
    hostIndex="1"
done
echo "Scheduler and workers now running"

#! We need to tell dask Client (inside python) where the scheduler is running
export ARL_DASK_SCHEDULER=${scheduler}:8786
echo "Scheduler is running at ${scheduler}"

# The signal bandwidth of 7MHz means that we would want to model about 14MHz, for example Seven DTV
# would require 170.5MHz to 184.50MHz at a resolution of 0.2MHz, spanning 280 frequency channels,
# time sampling would be about 0.2s.

CMD="python ../../simulate_low_rfi_visibility.py --rmax 3400 --npixel 1024 \
--station_skip 2 --noise False --attenuation 1.0 --use_agg True --declination -45 --ngroup_visibility 45 \
--nintegrations_per_chunk 256 --nchannels_per_chunk 256 --time_average 16 --channel_average 16 --integration_time 1.0 \
--frequency_range 170.5e6 184.5e6 | tee simulate_low_rfi_visibility.log"
echo "About to execute $CMD"

eval $CMD

