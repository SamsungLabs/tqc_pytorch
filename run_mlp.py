#! /usr/bin/env python
import git
import os
import yaml
from pathlib import Path
from conf_utils import apply_common, expand_lists_in_task_configs
import argparse
from pprint import pprint
import subprocess


LOG_DIR = "/gpfs-volume/multiq_results"


def get_odc_params():
    result = subprocess.run(['odc', 'whoami'], stderr=subprocess.PIPE)
    user = result.stderr.decode('utf-8').strip()

    params = {
        '--pool': 'trainingPool',
        '--core': 2,
        '--image': f'{user}/sac15',
        '--gpu': 1,
        '--github-path': 'git@github.sec.samsung.net:grishin-a/multiq.git',
    }

    return params


def calculate_schedule(n_seeds, n_configs, n_per_job):
    n_tasks = n_seeds * n_configs
    n_jobs = (n_tasks + n_per_job - 1) // n_per_job
    per_job_short = n_tasks // n_jobs
    n_long_jobs = n_tasks % n_jobs
    return [per_job_short + 1] * n_long_jobs + [per_job_short] * (n_jobs - n_long_jobs)


# probably could be done with external tool
def dict_to_cmd_args(d, prefix_keys=None):
    if prefix_keys is None:
        prefix_keys = []

    if isinstance(d, dict):
        return ' '.join([dict_to_cmd_args(d[k], prefix_keys + [k]) for k in d])
    else:
        return f"--{'.'.join(prefix_keys)} {d}"


def get_config(conf_path):
    with Path(conf_path).open() as f:
        config = yaml.safe_load(f)
    config = apply_common(config)
    config = expand_lists_in_task_configs(config)
    return config


def check_git_get_hash():
    repo = git.Repo('.')
    # assert not repo.is_dirty()
    # if not repo.head.is_detached:  # if we are not running old commits
    #     repo.remote().fetch()   # check whether local commit is the same on github
    #     assert repo.active_branch.commit == repo.active_branch.tracking_branch().commit
    git_hash = repo.commit().__str__()
    return git_hash


if __name__ == '__main__':
    git_hash = check_git_get_hash()

    parser = argparse.ArgumentParser(description="Run jobs on MLP cluster")
    parser.add_argument('--config', type=str, help='Path to config_ant.yaml.'
                        'By default "common" section will be applied to all task_configs' 
                        ' and all lists will be expanded')
    args = parser.parse_args()
    config = get_config(args.config)

    n_seeds = config['n_seeds']
    tasks_per_gpu = config['tasks_per_gpu']
    job_name = config['job_name']
    tasks_configs = config['tasks_configs']
    n_configs = len(tasks_configs)

    tasks_configs_with_seeds = tasks_configs * n_seeds

    # calculate distribution of tasks per jobs
    schedule = calculate_schedule(n_seeds=n_seeds,
                                  n_configs=n_configs,
                                  n_per_job=tasks_per_gpu)

    odc_params = get_odc_params()
    odc_params_str = ' '.join([f'{k}={v}' for k, v in odc_params.items()])
    start_index = 0
    for job_id, cur_job_n_tasks in enumerate(schedule):
        assert cur_job_n_tasks == 1
        print()
        cur_tasks_configs = tasks_configs_with_seeds[start_index:start_index + cur_job_n_tasks]
        task_config = cur_tasks_configs[0]
        # if explicit index is presented use it
        if 'index' in config:
            job_id = config['index']
        pprint(task_config)
        cmd_line_option = dict_to_cmd_args(task_config)
        cmd_option = f"--save_model --seed {job_id} --prefix {job_name} --log_dir {LOG_DIR}  {cmd_line_option}"
        final_cmd = f'odc mlp create job {job_name}-{job_id} {odc_params_str} -e "cd ~/multiq && git fetch && git checkout {git_hash} && bash ~/multiq/mlp_launcher.sh {cmd_option}"'
        os.system(final_cmd)
        start_index += cur_job_n_tasks
