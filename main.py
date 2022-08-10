import os
import subprocess
import argparse
import logging

import configs.experiments as experiments
import configs.exp_analysis as exp_analysis
from utils.config_utils import save_config, configs_dict_unpack
from train import model_train
from configs.config_global import LOG_LEVEL
from configs.configs import BaseConfig

def train_cmd(config_):
    arg = '\'' + config_.save_path + '\''
    command = r'''python -c "import train; train.train_from_path(''' + arg + ''')"'''
    return command


def eval_cmd(eval_save_path):
    arg = '\'' + eval_save_path + '\''
    command = r'''python -c "import model_evaluation; model_evaluation.evaluate_from_path(''' + arg + ''')"'''
    return command


def analysis_cmd(exp_name):
    arg = exp_name + '_analysis()'
    command = r'''python -c "import configs.exp_analysis as exp_analysis; exp_analysis.''' + arg + '''"'''
    return command


def get_jobfile(cmd, job_name, dep_ids, email=False,
                sbatch_path='./sbatch/', hours=8, partition='normal', 
                mem=32, gpu_constraint='high-capacity', cpu=2):
    """
    Create a job file.
    adapted from https://github.com/gyyang/olfaction_evolution

    Args:
        cmd: python command to be execute by the cluster
        job_name: str, name of the job file
        dep_ids: list, a list of job ids used for job dependency
        email: bool, whether or to send email about job status
        sbatch_path : str, Directory to store SBATCH file in.
        hours : int, number of hours to train
    Returns:
        job_file : str, Path to the job file.
    """
    assert type(dep_ids) is list, 'dependency ids must be list'
    assert all(type(id_) is str for id_ in dep_ids), 'dependency ids must all be strings'

    if len(dep_ids) == 0:
        dependency_line = ''
    else:
        dependency_line = '#SBATCH --dependency=afterok:' \
                          + ':'.join(dep_ids) + '\n'

    email_line = ''
    os.makedirs(sbatch_path, exist_ok=True)
    job_file = os.path.join(sbatch_path, job_name + '.sh')

    with open(job_file, 'w') as f:
        f.write(
            '#!/bin/bash\n'
            + '#SBATCH -t {}:00:00\n'.format(hours)
            + '#SBATCH -N 1\n'
            + '#SBATCH -n {}\n'.format(cpu)
            + '#SBATCH --mem={}G\n'.format(mem)
            + '#SBATCH --gres=gpu:1\n'
            + '#SBATCH --constraint={}\n'.format(gpu_constraint)
            + '#SBATCH -p {}\n'.format(partition)
            + '#SBATCH -e ./sbatch/slurm-%j.out\n'
            + '#SBATCH -o ./sbatch/slurm-%j.out\n'
            + dependency_line
            + email_line
            + '\n'
            + 'module load openmind/cuda/11.1\n'
            + 'source ~/.bashrc\n'
            + 'conda activate plastic\n'
            + 'cd /om2/user/duany19/generalized_hebbian\n'
            + cmd + '\n'
            + '\n'
            )
        print(job_file)
    return job_file


def train_experiment(experiment, on_cluster, use_exp_array, partition):
    """Train model across platforms given experiment name.
    adapted from https://github.com/gyyang/olfaction_evolution
    Args:
        experiment: str, name of experiment to be run
            must correspond to a function in experiments.py
        on_cluster: bool, whether to run experiments on cluster
        use_exp_array: use dependency between training of different exps

    Returns:
        return_ids: list, a list of job ids that are last in the dependency sequence
            if not using cluster, return is an empty list
    """
    print('Training {:s} experiment'.format(experiment))
    if experiment in dir(experiments):
        # Get list of configurations from experiment function
        exp_configs = getattr(experiments, experiment)()
    else:
        raise ValueError('Experiment config not found: ', experiment)

    return_ids = []
    if not use_exp_array:
        # exp_configs is a list of configs
        exp_configs = configs_dict_unpack(exp_configs)
        assert isinstance(exp_configs[0], BaseConfig), \
            'exp_configs should be list of configs'

        if on_cluster:
            for config in exp_configs:
                if not save_config(config, config.save_path):
                    continue
                python_cmd = train_cmd(config)
                job_n = config.experiment_name + '_' + config.model_name
                cp_process = subprocess.run(['sbatch', get_jobfile(python_cmd,
                                                                   job_n,
                                                                   dep_ids=[],
                                                                   hours=config.hours, 
                                                                   partition=partition,
                                                                   mem=config.mem,
                                                                   gpu_constraint=config.gpu_constraint,
                                                                   cpu=config.cpu,
                                                                   sbatch_path=config.save_path
                                                                   )],
                                            capture_output=True, check=True)
                cp_stdout = cp_process.stdout.decode()
                print(cp_stdout)
                job_id = cp_stdout[-9:-1]
                return_ids.append(job_id)
        else:
            for config in exp_configs:
                if save_config(config, config.save_path):
                    model_train(config)
    else:
        exp_configs = [configs_dict_unpack(cfg_dict) for cfg_dict in exp_configs]
        # exp_configs is a list of lists of configs
        assert isinstance(exp_configs[0], list) \
               and isinstance(exp_configs[0][0], BaseConfig), \
               'exp_configs should a list of lists of configs'

        if on_cluster:
            send_email = False
            pre_job_ids = []
            for group_num, config_group in enumerate(exp_configs):
                group_job_ids = []
                for config in config_group:
                    if not save_config(config, config.save_path):
                        continue

                    if group_num == len(exp_configs) - 1:
                        send_email = True

                    python_cmd = train_cmd(config)
                    job_n = config.experiment_name + '_' + config.model_name
                    cp_process = subprocess.run(['sbatch',
                                                 get_jobfile(python_cmd, job_n,
                                                             dep_ids=pre_job_ids,
                                                             email=send_email,
                                                             hours=config.hours, 
                                                             partition=partition)], # TODO: Update this line
                                                capture_output=True, check=True)
                    cp_stdout = cp_process.stdout.decode()
                    print(cp_stdout)
                    job_id = cp_stdout[-9:-1]
                    group_job_ids.append(job_id)
                pre_job_ids = group_job_ids

            return_ids = pre_job_ids

        else:
            for config_group in exp_configs:
                for config in config_group:
                    if save_config(config, config.save_path):
                        model_train(config)

    return return_ids

def analyze_experiment(experiment, prev_ids, on_cluster, partition):
    """analyze experiments
     adapted from https://github.com/gyyang/olfaction_evolution

     Args:
         experiment: str, name of experiment to be analyzed
             must correspond to a function in exp_analysis.py
         prev_ids: list, list of previous job ids
         on_cluster: bool, if use on the cluster
     """
    print('Analyzing {:s} experiment'.format(experiment))
    if (experiment + '_analysis') in dir(exp_analysis):
        if on_cluster:
            python_cmd = analysis_cmd(experiment)
            job_n = experiment + '_analysis'
            slurm_cmd = ['sbatch', get_jobfile(python_cmd, job_n,
                                               dep_ids=prev_ids, email=False, partition=partition,
                                               hours=24, mem=64)]
            cp_process = subprocess.run(slurm_cmd, capture_output=True,
                                        check=True)
            cp_stdout = cp_process.stdout.decode()
            print(cp_stdout)
        else:
            getattr(exp_analysis, experiment + '_analysis')()
    else:
        raise ValueError('Experiment analysis not found: ', experiment + '_analysis')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', nargs='+', help='Train experiments', default=[])
    parser.add_argument('-a', '--analyze', nargs='+', help='Analyze experiments', default=[])
    parser.add_argument('-c', '--cluster', action='store_true', help='Use batch submission on cluster')
    parser.add_argument('-p', '--partition', default='normal', help='Partition of resource on cluster to use')
    
    args = parser.parse_args()
    experiments2train = args.train
    experiments2analyze = args.analyze
    use_cluster = args.cluster
    # on openmind cluster
    # use_cluster = 'node' in platform.node() or 'dgx' in platform.node()
    logging.basicConfig(level=LOG_LEVEL)

    # evaluation jobs are executed after training jobs,
    # analysis jobs are executed after training and evaluation jobs.
    train_ids = []
    if experiments2train:
        for exp in experiments2train:
            exp_array = '_exp_array' in exp
            exp_ids = train_experiment(exp, on_cluster=use_cluster,
                                       use_exp_array=exp_array, partition=args.partition)
            train_ids += exp_ids

    if experiments2analyze:
        for exp in experiments2analyze:
            analyze_experiment(exp, prev_ids=train_ids,
                               on_cluster=use_cluster, partition=args.partition)
