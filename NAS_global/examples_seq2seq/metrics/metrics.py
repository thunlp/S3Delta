# several of the evaluation metrics are from https://github.com/google-research/text-to-text-transfer-transformer/blob/a1352e625db7ec114062f99d99b0565b9e45c155/t5/evaluation/metrics.py
"""Defines different metrics used for evaluation of tasks."""
from turtle import pos
import numpy as np
import scipy
import math
import sklearn
import collections
from logging import getLogger
from .qa_utils import normalize_squad, qa_metrics
import sklearn.metrics
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# import evaluate
logger = getLogger(__name__)
def chrf(predictions, targets) -> dict:
    metrics_ = evaluate.load('chrf')
    # check for empty lists
    seen = {'ref':[],'hyp':[]}
    unseen = {'ref':[],'hyp':[]}
    all_seen = {'ref':[],'hyp':[]}

    for i, refs in enumerate(targets):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            if predictions[i][1]:
                seen['hyp'].append(predictions[i][0])
                seen['ref'].append([ref for ref in refs_])
            else:
                unseen['hyp'].append(predictions[i][0])
                unseen['ref'].append([ref for ref in refs_])
            all_seen['hyp'].append(predictions[i][0])
            all_seen['ref'].append([ref for ref in refs_])

    if len(seen['hyp']) > 0:
        seen_result = metrics_.compute(references=seen['ref'], predictions=seen['hyp'],word_order=2,lowercase=True,eps_smoothing=True)
    else:
        seen_result=-1
    if len(unseen['hyp']) > 0:
        unseen_result = metrics_(references=unseen['ref'], predictions=unseen['hyp'],word_order=2,lowercase=True,eps_smoothing=True)
    else:
        unseen_result=-1
    
    all_seen_result = metrics_.compute(references=all_seen['ref'], predictions=all_seen['hyp'],word_order=2,lowercase=True,eps_smoothing=True)
    return {"chrf++_all": all_seen_result,"chrf++_seen": seen_result,"chrf++_unseen": unseen_result}
    
def bleu(predictions, targets) -> dict:
    metrics_ = evaluate.load('bleu')
    # check for empty lists
    seen = {'ref':[],'hyp':[]}
    unseen = {'ref':[],'hyp':[]}
    all_seen = {'ref':[],'hyp':[]}

    for i, refs in enumerate(targets):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            if predictions[i][1]:
                seen['hyp'].append(predictions[i][0])
                seen['ref'].append([ref for ref in refs_])
            else:
                unseen['hyp'].append(predictions[i][0])
                unseen['ref'].append([ref for ref in refs_])
            all_seen['hyp'].append(predictions[i][0])
            all_seen['ref'].append([ref for ref in refs_])

    if len(seen['hyp']) > 0:
        seen_result = metrics_.compute(references=seen['ref'], predictions=seen['hyp'],smooth=True)
    else:
        seen_result=-1
    if len(unseen['hyp']) > 0:
        unseen_result = metrics_(references=unseen['ref'], predictions=unseen['hyp'],smooth=True)
    else:
        unseen_result=-1
    
    all_seen_result = metrics_.compute(references=all_seen['ref'], predictions=all_seen['hyp'],smooth=True)
    return {"bleu_all": all_seen_result,"bleu_seen": seen_result,"bleu_unseen": unseen_result}

def meteor(predictions, targets) -> dict:
    meteor = evaluate.load('meteor')
    # check for empty lists
    seen = {'ref':[],'hyp':[]}
    unseen = {'ref':[],'hyp':[]}
    all_seen = {'ref':[],'hyp':[]}
    references_, hypothesis_ = [], []
    for i, refs in enumerate(targets):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            if predictions[i][1]:
                seen['hyp'].append(predictions[i][0])
                seen['ref'].append([ref for ref in refs_])
            else:
                unseen['hyp'].append(predictions[i][0])
                unseen['ref'].append([ref for ref in refs_])
            all_seen['hyp'].append(predictions[i][0])
            all_seen['ref'].append([ref for ref in refs_])

    if len(seen['hyp']) > 0:
        seen_result = meteor.compute(references=seen['ref'], predictions=seen['hyp'])
    else:
        seen_result=-1
    if len(unseen['hyp']) > 0:
        unseen_result = meteor(references=unseen['ref'], predictions=unseen['hyp'])
    else:
        unseen_result=-1
    
    all_seen_result = meteor.compute(references=all_seen['ref'], predictions=all_seen['hyp'])
    return {"meteor_all": all_seen_result,"meteor_seen": seen_result,"meteor_unseen": unseen_result}

def bleu_nltk(predictions, targets):
    # check for empty lists
    seen = {'ref':[],'hyp':[]}
    unseen = {'ref':[],'hyp':[]}
    all_seen = {'ref':[],'hyp':[]}
    references_, hypothesis_ = [], []
    for i, refs in enumerate(targets):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            if predictions[i][1]:
                seen['hyp'].append(predictions[i][0].split())
                seen['ref'].append([ref.split() for ref in refs_])
            else:
                unseen['hyp'].append(predictions[i][0].split())
                unseen['ref'].append([ref.split() for ref in refs_])
            all_seen['hyp'].append(predictions[i][0].split())
            all_seen['ref'].append([ref.split() for ref in refs_])

    if len(seen['hyp']) > 0:
        seen_result = corpus_bleu(seen['ref'], seen['hyp'], smoothing_function=SmoothingFunction().method3)
    else:
        seen_result=-1
    if len(unseen['hyp']) > 0:
        unseen_result = corpus_bleu(unseen['ref'], unseen['hyp'], smoothing_function=SmoothingFunction().method3)
    else:
        unseen_result=-1
    
    all_seen_result = corpus_bleu(all_seen['ref'], all_seen['hyp'], smoothing_function=SmoothingFunction().method3)
    return {"bleu_nltk_all": 100 * all_seen_result,"bleu_nltk_seen": 100 * seen_result,"bleu_nltk_unseen": 100 * unseen_result}

def accuracy(predictions, targets) -> dict:
    """Computes the average accuracy."""
    return {"accuracy": 100 * ((np.array(predictions) == np.array(targets)).mean())}

def pearson_corrcoef(predictions, targets) -> dict:
    """Computes Pearson correlation coefficient."""
    from examples_seq2seq.data_processors.postprocessors import string_to_float
    targets = [string_to_float(target) for target in targets]
    predictions= [string_to_float(prediction) for prediction in predictions]
    pearson_corrcoef = 100 * scipy.stats.pearsonr(targets, predictions)[0]

    # Note that if all the predictions will be the same, spearman
    # correlation is nan, to gaurad against this, we check the output
    # and return 0 in this case.
    if math.isnan(pearson_corrcoef):
        pearson_corrcoef = 0
    return {"pearson": pearson_corrcoef}


def spearman_corrcoef(predictions, targets) -> dict:
    """Computes Spearman correlation coefficient."""
    # TODO: we need to do postprocessors in a clean way for each dataset.
    from examples_seq2seq.data_processors.postprocessors import string_to_float
    targets = [string_to_float(target) for target in targets]
    predictions= [string_to_float(prediction) for prediction in predictions]
    spearman_corrcoef = 100 * scipy.stats.spearmanr(targets, predictions)[0]

    # Note that if all the predictions will be the same, spearman
    # correlation is nan, to gaurad against this, we check the output
    # and return 0 in this case.
    if math.isnan(spearman_corrcoef):
        spearman_corrcoef = 0
    return {"spearmanr": spearman_corrcoef}


def f1_score_with_invalid(predictions, targets) -> dict:
    """Computes F1 score,  with any prediction != 0 or 1 is counted as incorrect.
    Args:
      targets: list of targets, either 0 or 1
      predictions: list of predictions, any integer value
    Returns:
      F1 score, where any prediction != 0 or 1 is counted as wrong.
    """
    def binary_reverse(labels):
       return ['0' if label == '1' else '1' for label in labels]
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    # Get indices of invalid predictions.
    invalid_idx_mask = np.logical_and(predictions != '0', predictions != '1')
    # For any prediction != 0 or 1, we set the prediction to the opposite of its corresponding target.
    predictions[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask])
    # targets = targets.astype(np.int32)
    # predictions = predictions.astype(np.int32)
    # print(targets,predictions)
    return {"f1": 100 * sklearn.metrics.f1_score(targets, predictions,pos_label="1")}

# TODO: maybe gaurd against invalid values https://stackoverflow.com/questions/56865344/how-do-i-calculate-the-matthews-correlation-coefficient-in-tensorflow
def matthews_corrcoef(predictions, targets) -> dict:
    """Computes the Matthews correlation coefficient."""
    return {"matthews_correlation": 100 * sklearn.metrics.matthews_corrcoef(targets, predictions)}

def squad(predictions, targets):
  """Computes SQuAD metrics, maximizing over answers per question.
  Args:
    targets: list of lists of strings
    predictions: list of strings
  Returns:
    dict with score_key: squad score across all targets and predictions
  """

  targets = [[normalize_squad(t) for t in u] for u in targets]
  predictions = [normalize_squad(p) for p in predictions]
  return qa_metrics(targets, predictions)


def exact_match(predictions, targets):
  """Computes whether the targets match predictions exactly."""
  return {"em": 100 * float(np.array_equal(targets, predictions))}


def sklearn_metrics_wrapper(metric_str,
                            metric_dict_str=None,
                            metric_post_process_fn=None,
                            **metric_fn_kwargs):
  """Wraps any sklearn.metric function and returns a t5 metric function.
  Args:
    metric_str: string, the function from `sklearn.metrics` to use.
    metric_dict_str: optional string, if not specified `metric_str` is used as
      the key in the returned dictionary.
    metric_post_process_fn: callable, if specified the final computed metric
      will be passed through this.
    **metric_fn_kwargs: kwargs, passed to the metric function we are calling.
  Returns:
    the function that calculates the metric in a dict.
  """
  if not hasattr(sklearn.metrics, metric_str):
    raise ValueError("sklearn.metrics does not have: %s" % metric_str)

  def fn(predictions, targets):
    metric_fn = getattr(sklearn.metrics, metric_str)
    metric_val = metric_fn(targets, predictions, **metric_fn_kwargs)
    if metric_post_process_fn is not None:
      metric_val = metric_post_process_fn(metric_val)
    return {metric_dict_str or metric_str: metric_val}
  return fn


def mean_multiclass_f1(num_classes, **metric_fn_kwargs):
  """Computes the unweighted average of the F1 per class."""
  return sklearn_metrics_wrapper(
      "fbeta_score",
      metric_dict_str="f1_multiclass",
      metric_post_process_fn=lambda x: 100 * x,
      beta=1,
      labels=range(num_classes),
      average="macro",
      **metric_fn_kwargs)


def multirc_f1_over_all_answers(targets, predictions):
  """Special metric for MultiRC which computes F1 score over all examples.
  This is necessary because the targets/predictions for MultiRC are dicts and
  the f1_score_with_invalid expects a list of True/False labels, not dicts. As
  a result we just need to key in the "value" for each of the example dicts
  before feeding into f1_score_with_invalid.
  Args:
    targets: list of dicts, where each dict has a "value" key.
    predictions: list of dicts, where each dict has a "value" key.
  Returns:
    F1 score over values, where any prediction != 0 or 1 is counted as wrong.
  """
  return f1_score_with_invalid(
      [t["value"] for t in targets], [p["value"] for p in predictions]
  )


def mean_group_metric(metric_fn, group_key="group", value_key="value"):
  """Returns a metric that averages `metric_fn` on sub-groups of results.
  The sub-groups are defined by aggregating results (targets and predictions)
  by accessing the feature specified by `group_key` in the target dicts.
  **WARNING**: Using this function can produce unreliable results if you do not
  pass in full groups. For example, if you evaluate over a random subsample of a
  validation set and do not retain all of the examples in each group, you may
  get results which aren't directly comparable to using the full validation set.
  Args:
    metric_fn: function, the metric to compute on the subgroups.
    group_key: string, the key for the grouping value in the target dictionary.
    value_key: string, the key for the value in the dictionaries.
  """
  def my_metric(targets, predictions):
    """Computes mean of `metric_fn` over subgroups of results."""
    grouped_values = collections.defaultdict(lambda: ([], []))
    for targ, pred in zip(targets, predictions):
      g = targ[group_key]
      grouped_values[g][0].append(targ[value_key])
      grouped_values[g][1].append(pred[value_key])
    group_scores = collections.defaultdict(list)
    for (targets, predictions) in grouped_values.values():
      for metric, score in metric_fn(targets, predictions).items():
        group_scores[metric].append(score)
    return {metric: np.mean(scores) for metric, scores in group_scores.items()}
  return my_metric
