import utils.commons.single_thread_env  # NOQA
import glob
import subprocess
from textgrid import TextGrid
import os
from text_to_speech.utils.commons.hparams import hparams, set_hparams


def train_mfa_align(mfa_outputs="mfa_outputs",
                    mfa_inputs="mfa_inputs",
                    model_name=None, pretrain_model_name=None,
                    mfa_cmd='train'):
    CORPUS = hparams['processed_data_dir'].split("/")[-1]
    NUM_JOB = int(os.getenv('N_PROC', os.cpu_count()))
    env_vars = [f'CORPUS={CORPUS}', f'NUM_JOB={NUM_JOB}']
    if mfa_outputs is not None:
        env_vars.append(f'MFA_OUTPUTS={mfa_outputs}')
    if mfa_inputs is not None:
        env_vars.append(f'MFA_INPUTS={mfa_inputs}')
    if model_name is not None:
        env_vars.append(f'MODEL_NAME={model_name}')
    if pretrain_model_name is not None:
        env_vars.append(f'PRETRAIN_MODEL_NAME={pretrain_model_name}')
    if mfa_cmd is not None:
        env_vars.append(f'MFA_CMD={mfa_cmd}')
    env_str = ' '.join(env_vars)
    print(f"| Run MFA for {CORPUS}. Env vars: {env_str}")
    subprocess.check_call(f'{env_str} bash mfa_usr/run_mfa_train_align.sh', shell=True)
    mfa_offset = hparams['preprocess_args']['mfa_offset']
    if mfa_offset > 0:
        for tg_fn in glob.glob(f'{hparams["processed_data_dir"]}/{mfa_outputs}/*.TextGrid'):
            tg = TextGrid.fromFile(tg_fn)
            max_time = tg.maxTime
            for tier in tg.tiers:
                for interval in tier.intervals:
                    interval.maxTime = min(interval.maxTime + mfa_offset, max_time)
                    interval.minTime = min(interval.minTime + mfa_offset, max_time)
                tier.intervals[0].minTime = 0
                tier.maxTime = min(tier.maxTime + mfa_offset, max_time)
            tg.write(tg_fn)
            TextGrid.fromFile(tg_fn)


if __name__ == '__main__':
    set_hparams(print_hparams=False)
    train_mfa_align()
