import functools
import json
import logging
import os
import sys
import time
import torch
from examples_seq2seq.data_processors import AutoPostProcessor, AutoTask, TaskDataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, set_seed
from transformers.models.t5.modeling_t5 import T5Config, T5ForConditionalGeneration
import numpy as np
from argparser import get_args
from s3delta import S3Delta, alphas, named_alphas, named_weights, print_alphas, weights
from trainer import s3deltaTrainer
from sparse_structure_search import get_sparse_structure

logger = logging.getLogger(__name__)

TASK_TO_METRICS = {"mrpc": ["accuracy", "f1"],
                   "cola": ['matthews_correlation'],
                   "stsb": ['pearson', 'spearmanr'],
                   'sst2': ['accuracy'],
                   "mnli": ["accuracy"],
                   "mnli_mismatched": ["accuracy"],
                   "mnli_matched": ["accuracy"],
                   "qnli": ["accuracy"],
                   "rte": ["accuracy"],
                   "wnli": ["accuracy"],
                   "qqp": ["accuracy", "f1"],
                   "superglue-boolq": ["accuracy"],
                   "superglue-rte": ["accuracy"],
                   "superglue-cb": ["f1_multiclass", "accuracy"],
                   "superglue-copa": ["accuracy"],
                   "superglue-multirc": ["f1", "em"],
                   "superglue-wic": ["accuracy"],
                   "superglue-wsc.fixed": ["accuracy"],
                   "superglue-record": ["f1", "em"]
                   }
addable_delta_type = ['None', 'Adapter', 'BitFit',
                      'Compacter', 'Lora', 'LowRankAdapter', 'Prefix']


def main(args=None):

    # get args
    if args is None:
        args = get_args()

    # check if output dir exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        if_continue = input(
            f"output dir exists: {args.output_dir}\ncontinue to overwrite? [Y]yes, [else]No: ")
        if if_continue != 'Y':
            raise RuntimeError

    with open(os.path.join(args.output_dir, "config.json"), 'w') as fout:
        json.dump(args.__dict__, fout, indent=4, sort_keys=False)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(
            os.path.join(args.output_dir, 'log.out'), 'w+')],
    )
    logger.setLevel(logging.INFO)

    logger.info(f"outputdir: {args.output_dir}")
    logger.info(f"task name: {args.task_name}")

    # Set seed before initializing model.
    set_seed(args.seed)

    # config, tokenizer, model
    config = T5Config.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    if (not args.for_baseline and args.using_darts) and args.if_search:
        v_model = T5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    if (not args.for_baseline and args.using_darts) and args.if_search:
        v_model.resize_token_embeddings(len(tokenizer))

    # delta model
    # modify model from modified_configs
    delta_model = S3Delta(model, args, common_structure=False)
    for modified_configs in args.modified_configs:
        delta_model.modify_backbone_from_config(modified_configs=modified_configs)
    if not args.for_baseline_finetuning:
        delta_model.freeze_module(exclude=args.unfrozen_modules)
        delta_model.calculate_params_num()
    with open(os.path.join(args.output_dir, 'structure.out'), 'w+') as file:
        delta_model.log(delta_ratio=False, trainable_ratio=False,visualization=True, file=file)
        delta_model.log(delta_ratio=False, trainable_ratio=False, visualization=True)

    # num of params
    logger.info(
        f"Backbone params {delta_model.num_backbone_parameters()} M , Tunable params {delta_model.num_trainable_parameters(model)/1024**2:.10f} M , ratio {delta_model.num_trainable_parameters(model)/(delta_model.num_backbone_parameters()*1024*1024)*100:.10f}%")
    args.total_params = delta_model.num_trainable_parameters(model)/1024**2
    
    if args.how_to_calc_max_num_param == 'abs':
        args.max_num_param = args.max_num_param_abs
    elif args.how_to_calc_max_num_param == 'relative':
        args.max_num_param = delta_model.num_backbone_parameters()*(args.max_num_param_relative)
    else:
        raise NotImplementedError
    assert delta_model.num_trainable_parameters(model)/1024**2 > args.max_num_param, "max param is larger than total tunable param1"
    logger.info(f"max_num_param: {args.max_num_param}")

    # cuda
    if args.use_cuda:
        model = model.cuda()

    # virtual model
    if (not args.for_baseline and args.using_darts) and args.if_search:
        v_delta_model = S3Delta(v_model, args=args, common_structure=False)
        for modified_configs in args.modified_configs:
            v_delta_model.modify_backbone_from_config(
                modified_configs=modified_configs)
        v_delta_model.freeze_module(exclude=args.unfrozen_modules)
        v_delta_model.calculate_params_num()
        if args.use_cuda:
            v_model = v_model.cuda()
    else:
        v_delta_model = None
        v_model = None
#===============================================================
    # Data collator
    data_collator = TaskDataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    # function for preprocessing the dataset
    def preprocess_function(examples, max_target_length):
        model_inputs = tokenizer(examples['source'], max_length=args.max_source_length,
                                 padding=False, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target'], max_length=max_target_length, padding=False, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["extra_fields"] = examples['extra_fields']
        return model_inputs

    column_names = ['source', 'target', 'extra_fields']
    performance_metrics = {}

    # training dataset
    train_datasets = AutoTask.get(args.task_name,["en"],seed=args.seed).get(
        split="train",
        split_validation_test=True,
        add_prefix=True,
        n_obs=None)
    max_target_length = AutoTask.get(args.task_name, ["en"]).get_max_target_length(
        tokenizer=tokenizer, default_max_length=128)

    train_dataset = train_datasets.map(
        functools.partial(preprocess_function,max_target_length=max_target_length),
        batched=True,
        remove_columns=column_names,
    )
    logger.info(f"train dataset: {len(train_dataset)}")

    # eval dataset
    eval_dataset = AutoTask.get(args.task_name, ["en"],seed=args.seed).get(
        split="validation",
        split_validation_test=True,
        add_prefix=True)

    max_target_length = AutoTask.get(args.task_name, ["en"]).get_max_target_length(
        tokenizer=tokenizer, default_max_length=128)

    eval_dataset = eval_dataset.map(
        functools.partial(preprocess_function, max_target_length=max_target_length),
        batched=True,
        remove_columns=column_names
    )
    logger.info(f"eval dataset: {len(eval_dataset)}\n")

    # test dataset
    test_dataset = AutoTask.get(args.task_name, ["en"], seed=args.seed).get(
        split="test",
        split_validation_test=True,
        add_prefix=True)

    max_target_length = AutoTask.get(args.task_name, ["en"]).get_max_target_length(
        tokenizer=tokenizer, default_max_length=128)

    test_dataset = test_dataset.map(
        functools.partial(preprocess_function,max_target_length=max_target_length),
        batched=True,
        remove_columns=column_names
    )
    logger.info(f"test dataset: {len(test_dataset)}\n")

    data_info = {"eval": eval_dataset['extra_fields'],
                 "test": test_dataset['extra_fields']}

    # split the dataset
    if not args.for_baseline and args.using_darts:
        train_dataset_ = train_dataset.remove_columns(['task', 'extra_fields'])
        train_dataset_length = len(train_dataset_)
        from torch.utils.data import random_split
        if train_dataset_length % 2 == 0:
            train_dataset_train, train_dataset_eval = random_split(train_dataset_, [train_dataset_length//2, train_dataset_length//2])
        else:
            train_dataset_train, train_dataset_eval, _ = random_split(train_dataset_, [train_dataset_length//2, train_dataset_length//2, 1])
        assert len(train_dataset_train) == len(train_dataset_eval)
        assert len(train_dataset_train) + len(train_dataset_eval) >= len(train_dataset_)-1

        eval_dataset_ = eval_dataset.remove_columns(['task', 'extra_fields'])
        test_dataset_ = test_dataset.remove_columns(['task', 'extra_fields'])
    else:
        train_dataset_train = train_dataset.remove_columns(['task', 'extra_fields'])
        train_dataset_eval = eval_dataset.remove_columns(['task', 'extra_fields'])
        eval_dataset_ = eval_dataset.remove_columns(['task', 'extra_fields'])
        test_dataset_ = test_dataset.remove_columns(['task', 'extra_fields'])

    # dataloader
    train_dataloader = DataLoader(train_dataset_train, batch_size=args.train_batch_size, shuffle=True, collate_fn=data_collator)
    eval_dataloader = DataLoader(train_dataset_eval, batch_size=args.train_batch_size, shuffle=True, collate_fn=data_collator)
    eval_dataloader_not_shuffle = DataLoader(eval_dataset_, batch_size=args.valid_batch_size, shuffle=False, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset_, batch_size=args.valid_batch_size, shuffle=False, collate_fn=data_collator)
    logger.info(f"{len(train_dataloader)} {len(eval_dataloader)} {len(eval_dataloader_not_shuffle)} {len(test_dataloader)}")

    # Metric
    eval_metrics = AutoTask.get(args.task_name, ["en"]).metric

    def compute_metrics(eval_preds):
        preds, labels, data_info = eval_preds
        post_processor = AutoPostProcessor.get(args.task_name, tokenizer, True)
        decoded_preds, decoded_labels = post_processor.process(preds, labels, data_info)
        result = {}
        for metric in eval_metrics:
            result.update(metric(decoded_preds, decoded_labels))
        return result
    
    # Initialize our Trainer
    trainer = s3deltaTrainer(model=model,
                             delta_model=delta_model,
                             v_model=v_model,
                             v_delta_model=v_delta_model,
                             data_info=data_info,
                             tokenizer=tokenizer,
                             compute_metrics=compute_metrics,
                             args=args,
                             train_dataloader=train_dataloader,
                             eval_dataloader=eval_dataloader,
                             eval_dataloader_not_shuffle=eval_dataloader_not_shuffle,
                             test_dataloader=test_dataloader
                             )

    # Training
    if args.if_search:
        trainer.train()
        os.rename(os.path.join(args.output_dir, 'best.ckpt'),
                  os.path.join(args.output_dir, 'best_search.ckpt'))

    # Test
    results = {}
    logger.info("*** Test ***")
    metrics = trainer.evaluate(
        test_dataloader, split='test', PATH=os.path.join(args.output_dir, 'best_search.ckpt'))
    results['test'] = metrics
    results['main_score'] = next(iter(metrics.values()))
    logger.info(results)

    # save final alphas
    if not args.for_baseline:
        p_alphas = print_alphas(delta_model)
        with open(f"{args.output_dir}/final_alphas.txt", 'w') as f:
            f.write(p_alphas)

    # log peak memory use
    if torch.cuda.is_available() and args.compute_memory and args.use_cuda:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        logger.info(f"Memory utilization {peak_memory} GB")

    # retrain
    if not args.for_baseline and args.train_after_search:

        logger.info("============retrain============")
        # modify the args: modified_configs, for_baseline, train_batch_size, valid_batch_size, train_epochs, valid_steps
        new_modified_modules_list = get_sparse_structure(
            args, model, delta_model)
        logger.info(new_modified_modules_list)
        args.modified_configs = new_modified_modules_list

        args.for_baseline = True
        args.train_batch_size = args.delta_tuning_train_batch_size
        args.valid_batch_size = args.delta_tuning_valid_batch_size
        args.train_epochs = args.delta_tuning_train_epochs
        args.valid_steps = args.delta_tuning_valid_steps
        
        with open(os.path.join(args.output_dir, "config_retrain.json"), 'w') as fout:
            json.dump(args.__dict__, fout, indent=4, sort_keys=False)

        del model, delta_model, v_model, v_delta_model, trainer
        torch.cuda.empty_cache()


        # new model, delta model
        config = T5Config.from_pretrained(args.model_name_or_path,)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,)
        model = T5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path, config=config,)
        model.resize_token_embeddings(len(tokenizer))
        delta_model = S3Delta(model, args, common_structure=False)
        for modified_configs in args.modified_configs:
            delta_model.modify_backbone_from_config(
                modified_configs=modified_configs)
        delta_model.freeze_module(exclude=args.unfrozen_modules)
        with open(os.path.join(args.output_dir, 'structure_retrain.out'), 'w+') as file:
            delta_model.log(delta_ratio=False, trainable_ratio=False,
                            visualization=True, file=file)
            delta_model.log(delta_ratio=False, trainable_ratio=False,
                            visualization=True)

        logger.info(
            f"Backbone params {delta_model.num_backbone_parameters()} M , Tunable params {delta_model.num_trainable_parameters(model)/1024**2:.10f} M , ratio {delta_model.num_trainable_parameters(model)/(delta_model.num_backbone_parameters()*1024*1024)*100:.10f}%")
        results['num_param']=delta_model.num_trainable_parameters(model)/(delta_model.num_backbone_parameters()*1024*1024)*100
        if args.use_cuda:
            model = model.cuda()
        v_model = None
        v_delta_model = None

        # dataset, dataloader
        data_info = {"eval": eval_dataset['extra_fields'],
                     "test": test_dataset['extra_fields']}

        train_dataset_ = train_dataset.remove_columns(['task', 'extra_fields'])
        eval_dataset_ = eval_dataset.remove_columns(['task', 'extra_fields'])
        test_dataset_ = test_dataset.remove_columns(['task', 'extra_fields'])

        train_dataloader = DataLoader(
            train_dataset_, batch_size=args.train_batch_size, shuffle=True, collate_fn=data_collator)
        eval_dataloader = DataLoader(
            eval_dataset_, batch_size=args.valid_batch_size, shuffle=False, collate_fn=data_collator)
        test_dataloader = DataLoader(
            test_dataset_, batch_size=args.valid_batch_size, shuffle=False, collate_fn=data_collator)

        # Initialize our Trainer
        trainer = s3deltaTrainer(model=model,
                                 delta_model=delta_model,
                                 v_model=v_model,
                                 v_delta_model=v_delta_model,
                                 data_info=data_info,
                                 tokenizer=tokenizer,
                                 compute_metrics=compute_metrics,
                                 args=args,
                                 train_dataloader=train_dataloader,
                                 eval_dataloader=None,
                                 eval_dataloader_not_shuffle=eval_dataloader,
                                 test_dataloader=test_dataloader
                                 )
        # re-training
        trainer.train()

        # Test
        logger.info("*** Test ***")
        metrics = trainer.evaluate(
            test_dataloader, split='test')
        results['test_retrain'] = metrics
        results['main_score'] = next(iter(metrics.values()))
        os.rename(os.path.join(args.output_dir, 'best.ckpt'), os.path.join(
            args.output_dir, 'best_retrain.ckpt'))  # try remove & move
        logger.info(results)

    if torch.cuda.is_available() and args.compute_memory and args.use_cuda:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        logger.info(f"Memory utilization {peak_memory} GB")
        performance_metrics.update({"peak_memory": peak_memory})

    results['output_dir'] = args.output_dir
    return results


if __name__ == "__main__":

    start = time.time()

    args = get_args()
    result = main(args)

    with open("./collect_result.jsonl", 'a') as fout:
        string = json.dumps(result, indent=4, sort_keys=True)
        fout.write(string+"\n")
        
    np.save(os.path.join(args.output_dir, "main_score.npy"),
            np.array(result['main_score']))

    logger.info(result)
    end = time.time()
    logger.info(f"time: {(end-start)/60} min")
    logger.info("=============finished=============")
