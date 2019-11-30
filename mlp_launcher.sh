cd ~/multiq
. ~/.bashrc
export OMP_NUM_THREADS=1
export PYTHONPATH=$PYTHONPATH:/home/user/multiq
python main.py "$@"
