import argparse
import os
import json
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def str2bool(s):
    if s.lower() == 'false':
        return False
    elif s.lower() == 'true':
        return True
    else:
        raise TypeError


def get_args():
    parser = argparse.ArgumentParser(description='S3Delta arguments.')

    parser.add_argument("--unfrozen_modules", type=str,nargs='+', default=["deltas"], help="")
    parser.add_argument("--model_name_or_path", type=str,default="t5-large", help="")
    parser.add_argument("--task_name", type=str, help="")
    parser.add_argument("--delta_strategy", type=str, default='default_delta_strategy', help="")
    


    parser.add_argument("--search_train_batch_size", type=int, default=1, help="")
    parser.add_argument("--search_valid_batch_size", type=int, default=1, help="")
    parser.add_argument("--search_train_epochs", type=int, default=1, help="")
    parser.add_argument("--search_valid_steps", type=int, default=1, help="")
    
    parser.add_argument("--delta_tuning_train_batch_size", type=int, default=1, help="")
    parser.add_argument("--delta_tuning_valid_batch_size", type=int, default=1, help="")
    parser.add_argument("--delta_tuning_train_epochs", type=int, default=1, help="")
    parser.add_argument("--delta_tuning_valid_steps", type=int, default=1, help="")

    parser.add_argument("--learning_rate", type=float, default=0.0003, help="")
    parser.add_argument("--sparsity_decay_step",default=-1, type=int, help="")

    parser.add_argument("--seed", type=int, default=0, help="")


    parser.add_argument("--clip_grad_norm", type=float, default=2.0, help="")
    parser.add_argument("--w_warmup_steps", type=int, default=0, help="")
    parser.add_argument("--alpha_warmup_steps", type=int, default=0, help="")
    parser.add_argument("--w_weight_decay", type=float, default=0, help="")

    parser.add_argument("--r", type=float, default=1, help="")
    parser.add_argument("--l", type=float, default=-0, help="")
    parser.add_argument("--beta", type=float, default=1, help="")
    parser.add_argument("--lambda0", type=float, default=0, help="")
    parser.add_argument("--alpha_learning_rate",type=float, default=0.1, help="")

    parser.add_argument("--how_to_calc_max_num_param", type=str, default="abs",choices=['abs','relative'], help="")
    parser.add_argument("--max_num_param_abs", type=float, default=0, help="")
    parser.add_argument("--max_num_param_relative", type=float, default=0, help="")

    parser.add_argument("--use_cuda", type=str2bool,default=True, help="")
    parser.add_argument("--compute_memory", type=str2bool,default=True, help="")
    parser.add_argument("--using_gumbel", type=str2bool, default=True, help="")

    parser.add_argument("--global_strategy", type=str, default='shift',choices=['tau_theta','root','shift','l0-norm'] ,help="")
    parser.add_argument("--detemining_structure_strategy", type=str, default='p',choices=['p','random'] ,help="")
    parser.add_argument("--hard_forward", type=str2bool, default=False, help="")
    parser.add_argument("--lr_decay", type=str2bool, default=True, help="")
    parser.add_argument("--shift_detach", type=str2bool, default=True, help="")
    
    parser.add_argument("--sample_once_per_step", type=str2bool, default=True, help="")
    parser.add_argument("--using_darts", type=str2bool, default=True, help="")

    parser.add_argument("--if_search", type=str2bool,
                        default=True, help="")
    parser.add_argument("--train_after_search",
                        type=str2bool, default=True, help="")
    parser.add_argument("--for_baseline", type=str2bool,
                        default=False, help="")
    parser.add_argument("--for_baseline_finetuning", type=str2bool,
                        default=False, help="")
    
    parser.add_argument("--modify_configs_from_json", type=str, help="")
    parser.add_argument("--output_dir", type=str, default='test', help="")
    args = parser.parse_args()

    if args.for_baseline_finetuning:
        assert args.for_baseline

    if 'superglue' in args.task_name:
        if 'superglue-record' == args.task_name:
            args.max_source_length = 512
        else:
            args.max_source_length = 256
    else:
        if "rte" == args.task_name:
            args.max_source_length = 256
        elif "web_nlg" == args.task_name:
            args.max_source_length = 300
        else:
            args.max_source_length = 128
    logger.info(f"max_source_length: {args.max_source_length}")

    args.output_dir = os.path.join(args.output_dir, str(args.seed))

    assert 0<= args.max_num_param_relative <=1
    assert 0<= args.max_num_param_abs

    if args.for_baseline:
        args.train_batch_size = args.delta_tuning_train_batch_size
        args.valid_batch_size = args.delta_tuning_valid_batch_size
        args.train_epochs = args.delta_tuning_train_epochs
        args.valid_steps = args.delta_tuning_valid_steps
    else:
        args.train_batch_size = args.search_train_batch_size
        args.valid_batch_size = args.search_valid_batch_size
        args.train_epochs = args.search_train_epochs
        args.valid_steps = args.search_valid_steps

    args.modified_configs = json.load(open(args.modify_configs_from_json)) 
    
    return args


if __name__ == '__main__':
    args = get_args()
    import json
    with open("test.json", 'w') as fout:
        string = json.dump(args.__dict__, fout, indent=4, sort_keys=False)