# Guided Strategy Discovery

Public implementation for the paper [Generalized Behavior Learning from Diverse Demonstrations](https://openreview.net/pdf?id=5uEkcZZCnk), published at the [OOD Workshop](https://sites.google.com/stanford.edu/corl2023ood/home?authuser=0) at [CoRL 2023](https://www.corl2023.org/).

## Setup

Our implementation uses pytorch. We recommend setting up a conda environment. `env.yaml` provides the environment file.

```
conda env create -f env.yaml
```

Additionally, mujoco210 is required. Please download and setup from the official source.

## Training

To run models for any of the domains, first use the provided base commands and then proceed to modify command line arguments as indicated.

Importance weight of diversity lambda_I is implemented as 1-dr_cc. For example, to set lambda_I=0.2, use `--dr_cc 0.8`. For sweep range, please refer to the paper appendix.

For regularization methods (SN, GSD), magnitude of regularization lambda_C is implemented as dl_scale. For example, to set lambda_C=0.1, use `--dl_scale 0.1`. For sweep ranges for each domain, please refer to the paper appendix.

### Base commands

We provide the base commands for each domain to train InfoGAIL (IG).

#### PointMaze
```
python code/vild_main.py \
    --env_id -5 --c_data 1 --v_data 1 \
    --max_step 2500000 --big_batch_size 1000 --bc_step 0 \
    --il_method infogsdr --rl_method ppo \
    --encode_dim 2 --encode_sampling normal \
    --info_loss_type bce --clip_discriminator 0 --offset_reward 0 \
    --ac_dec 0 --ac_rew 1 \
    --dr_cc 0.9 --gp_lambda 0.01 --learning_rate_d 1e-3 \
    --p_step 1 --lr_p 1e-3 --wd_p 0 --lr_dk 0 \
    --reg_dec 0 --sn_dec 0 --dl_l2m 0 \
    --dl_scale 0.0001 --dl_linit 100 --dl_llr 1e-3 --dl_slack 1e-6 \
    --seed 1
```

#### InvertedPendulum
```
python code/vild_main.py \
    --env_id -71 --c_data 1 --max_step 10000000 --v_data 2 \
    --il_method infogsdr --rl_method ppo \
    --encode_dim 2 --encode_sampling normal \
    --info_loss_type bce --clip_discriminator 0 --offset_reward 0 \
    --ac_dec 1 --ac_rew 1 \
    --dr_cc 0.9 --gp_lambda 0.1 --learning_rate_d 1e-3 \
    --p_step 1 --lr_p 1e-3 --wd_p 0 --lr_dk 0 \
    --reg_dec 0 --sn_dec 0 \
    --dl_scale 0.001 --dl_linit 100 --dl_llr 1e-3 --dl_slack 1e-6 --dl_l2m 0 \
    --seed 1 --nthreads 2
```

#### HalfCheetah
```
python code/vild_main.py \
    --env_id -20 --c_data 1 --max_step 10000000 \
    --il_method infogsdr --rl_method ppo \
    --encode_dim 2 --encode_sampling normal \
    --info_loss_type bce --clip_discriminator 0 --offset_reward 0 \
    --ac_dec 1 --ac_rew 1 \
    --dr_cc 0.9 --gp_lambda 0.1 --learning_rate_d 1e-3 \
    --p_step 1 --lr_p 1e-3 --wd_p 0 --lr_dk 0 \
    --reg_dec 0 --sn_dec 0 \
    --dl_scale 0.001 --dl_linit 10 --dl_llr 1e-4 --dl_slack 1e-6 --dl_l2m 0 \
    --seed 1 --nthreads 2
```

#### FetchPickPlace
```
python code/vild_main.py \
    --env_id -43 --c_data 1 --max_step 10000000 \
    --il_method infogsdr --rl_method ppobc --bc_cf 0.1 --norm_obs 1 \
    --encode_dim 2 --encode_sampling normal \
    --info_loss_type bce --clip_discriminator 0 --offset_reward 0 \
    --ac_dec 1 --ac_rew 1 \
    --dr_cc 0.9 --gp_lambda 0.1 --learning_rate_d 1e-3 \
    --p_step 1 --lr_p 1e-3 --wd_p 0 --lr_dk 0 \
    --reg_dec 0 --sn_dec 0 \
    --dl_scale 0.001 --dl_linit 100 --dl_llr 1e-3 --dl_slack 1e-6 --dl_l2m 0 \
    --seed 1 --nthreads 2
```

### Modifications to run various models

Modifications of base command for IG+M
PointMaze
```
--clip_discriminator 5
```
Others
```
--clip_discriminator 10
```

Modifications of base command for IG+M+SN
```
--clip_discriminator 5/10
--sn_dec 1
```

Modifications of base command for GSD (Our regularization)
```
--clip_discriminator 5/10
--reg_dec 1 --dl_l2m 1
```

## Evaluation

Please use the below command to perform evaluation as in the submission, by setting the correct env_id, v_data and policy path path as indicated below. The sampling process is parallelized by default. To set the desired number of threads, modify `code/run_model.py:L436` to set
`NPARALLEL=<desired_nthreads>`.

```
python code/run_model.py \
    --env_id -71 --c_data 1 --v_data 2 \
    --mode prior --num_eps 1 --bgt_info `seq 10 10 51` --num_info 1500 \
    --test_seed 1 \
    --ckptpath results_IL/path/to/dir/ckpt_policy_T10000000.pt
```
InvertedPendulum: `--env_id -71 --v_data 2`  
HalfCheetah: `--env_id -20 --v_data 0`  
FetchPickPlace: `--env_id -43 --v_data 0`  

The script will log a significant amount of information to the console. Among them, lines of interest take the form:
- `VL<K>-<GTD> <mean> <std>`
    - Denotes the recovery metric (least MAE) reported in the submission.
- `RT-VL<K>-<GTD> <mean> <std> <max> <min>`
    - Denotes the reward obtained by the behavior corresponding to the z that minimized MAE with the desired GT factor value

Here, `<K>` denotes the number of samples considered, `<GTD>` corresponds to the desired GT factor value among `[1, 2, 3, 4, 5]` (canonicalized across domains).

Such information should be averaged over train seeds to construct the figures in the submission. The below command does so from three log files, each corresponding to the three domains in the paper. Each individual log file contains the console outputs corresponding to the four methods and 5 train seeds, i.e., [IG, IG+M, IG+M+SN, GSD] x [1, 2, 3, 4, 5].

`python code/print_latex.py runp.log runc.log runf.log`

## License
Code is available under MIT License.

## Citation
If you use this work and/or this codebase in your research, please cite as shown below:

```
@inproceedings{sreeramdass2023generalized,
    title={Generalized Behavior Learning from Diverse Demonstrations},
    author={Sreeramdass, Varshith and Paleja, Rohan R and Chen, Letian and Van Waveren, Sanne and Gombolay, Matthew},
    booktitle={First Workshop on Out-of-Distribution Generalization in Robotics at CoRL 2023},
    year={2023},
    url={https://openreview.net/pdf?id=5uEkcZZCnk}
}
```
