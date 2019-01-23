import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_accuracy(preds, target):
    """
    Accuracy for multiclass model.
    :param preds: predictions
    :param labels: ground truth labelt
    :return: average accuracy
    """
    omg = torch.sum(target,0)
    len_omg = len(torch.nonzero(omg))
    preds = torch.max(preds, 0)[1].float()
    target = torch.max(target, 0)[1].float()

    correct_prediction = torch.mul(omg, (preds == target).float())
    return torch.sum(correct_prediction)/len_omg


def rmse(logits, labels):
    """
    Computes the mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth labels for the ratings, 1-D array containing 0-num_classes-1 ratings
    :param class_values: rating values corresponding to each class.
    :return: mse
    """

    omg = torch.sum(labels, 0).detach()
    len_omg = len(torch.nonzero(omg))

    pred_y = logits
    y = torch.max(labels, 0)[1].float() + 1.

    se = torch.sub(y, pred_y).pow_(2)
    mse= torch.sum(torch.mul(omg, se))/len_omg
    rmse = torch.sqrt(mse)

    return rmse

def expected_rmse(logits, labels, class_values=None):
    """
    Computes the root mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth label
    :param class_values: rating values corresponding to each class.
    :return: rmse
    """

    omg = torch.sum(labels, 0).float().detach()
    len_omg = len(torch.nonzero(omg))

    pred_y = logits
    if class_values is None:
        scores = tf.to_float(tf.range(start=0, limit=logits.get_shape()[1]) + 1)
        y = tf.to_float(labels) + 1.  # assumes class values are 1, ..., num_classes
    else:
        #scores = class_values
        y = class_values[torch.max(labels, 0)[1]]
        #y = labels.float()+1.

    se = torch.sub(y, pred_y).pow_(2)
    mse= torch.sum(torch.mul(omg, se))/len_omg
    rmse = torch.sqrt(mse)

    return rmse


def softmax_cross_entropy(input, target):
    """ computes average softmax cross entropy """

    input = input.view(input.size(0),-1).t()
    target = target.view(target.size(0),-1).t()

    omg = torch.sum(target,1).detach()
    len_omg = len(torch.nonzero(omg))
    target = torch.max(target, 1)[1]

    loss = F.cross_entropy(input=input, target=target, reduction='none')
    loss = torch.sum(torch.mul(omg, loss))/len_omg

    return loss

def cal_ndcg(input, target):
    full, top_k = self._subjects, self._top_k
    top_k = full[full['rank']<=top_k]
    test_in_top_k =top_k[top_k['test_item'] == top_k['item']]
    test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
    return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()
