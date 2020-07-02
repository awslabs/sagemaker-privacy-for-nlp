# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

import argparse
import subprocess
import json
import sys
import os
import pathlib
import tarfile
import logging

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import numpy as np
import torch
from torchtext import vocab
from torchtext.data import TabularDataset, BucketIterator
import seaborn as sns
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath("/opt/ml/processing/code"))
from train import create_fields, create_model, binary_accuracy

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)


def evaluation_metrics(output_dir, epochs, precision, recall, accuracy, roc_auc, filename="MIA-metrics.txt"):
    f1score = 2 * precision * recall / (precision + recall)
    mia_result = "Results for " + str(epochs) + " epochs \n"\
                 + f"F1 score = {f1score:.2f}, Precision = {precision:.2f}, " \
                 + f"Recall = {recall:.2f}, AUC = {roc_auc:.2f}"

    LOG.info(mia_result)
    print(mia_result)

    with open(os.path.join(output_dir, filename), "w") as text_file:
        text_file.write(mia_result)

def plot_roc_curve(output_dir, fpr, tpr, roc_auc, filename, title):
    sns.set(context='notebook', style='whitegrid')
    sns.utils.axlabel(xlabel="False Positive Rate", ylabel="True Positive Rate", fontsize=16)
    linelabel = 'ROC curve (auc = %.2f)' % (roc_auc)
    sns.lineplot(x=fpr, y=tpr, lw=2, label=linelabel).set_title(title)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')


def calculate_metrics(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    all_predictions = np.array([])
    all_labels = np.array([])
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.review).squeeze(1)

            loss = criterion(predictions, batch.sentiment)

            acc = binary_accuracy(predictions, batch.sentiment)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            all_predictions = np.concatenate(
                (all_predictions, torch.sigmoid(predictions).detach().cpu().numpy()), axis=0)
            all_labels = np.concatenate(
                (all_labels, batch.sentiment.detach().cpu().clone().numpy()), axis=0)

    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions, pos_label=1)
    prec = precision_score(all_labels, np.round(all_predictions, 0))
    rec = recall_score(all_labels, np.round(all_predictions, 0))

    return epoch_loss / iterator.iterations, epoch_acc / iterator.iterations, fpr, tpr, prec, rec


def evaluate_model(test_dataset, model_file, vocab_file, output_dir, embedding_size=300):
    TEXT, LABEL = create_fields()

    fields = [('sentiment', LABEL),
              ('title', None),
              ('review', TEXT)]

    test_reviews = TabularDataset(
        path=test_dataset, format='csv',
        fields=fields,
        skip_header=True)

    train_vocab = torch.load(vocab_file)

    TEXT.vocab = train_vocab
    LABEL.build_vocab(test_reviews)

    model = create_model(
        len(TEXT.vocab),
        embedding_size,
        TEXT.vocab.stoi[TEXT.pad_token],
        TEXT.vocab.stoi[TEXT.unk_token],
        torch.zeros((len(TEXT.vocab), embedding_size)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(model_file, map_location=device))

    model.to(device)

    batch_size = 64
    criterion = torch.nn.BCEWithLogitsLoss()
    test_iterator = BucketIterator(
        test_reviews, batch_size=batch_size, sort=False, device=device)

    test_loss, test_acc, fpr, tpr, prec, rec = calculate_metrics(model, test_iterator, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    with open(os.path.join(output_dir, "output.json"), 'w') as outfile:
        json.dump({"test_loss": test_loss, "test_accuracy": test_acc}, outfile)

    return test_loss, test_acc, fpr, tpr, prec, rec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default='/opt/ml/processing/data')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/processing/model')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--embedding-size', type=int, default=300)
    parser.add_argument('--training-epochs', type=int, default=5)
    args = parser.parse_args()

    tar = tarfile.open("{}/model.tar.gz".format(args.model_dir))
    tar.extractall(path=args.model_dir)
    tar.close()

    test_loss, test_acc, fpr, tpr, prec, rec = evaluate_model(
        "{}/test_examples.csv".format(args.data_dir),
        os.path.join(args.model_dir, "model.pt"),
        os.path.join(args.model_dir, "vocab.pt"),
        args.output_dir,
        args.embedding_size)

    # Save an ROC curve
    roc_auc = auc(fpr, tpr)
    evaluation_metrics(args.output_dir, args.training_epochs, prec, rec, test_acc, roc_auc, "accuracy-metrics.txt")
    plot_roc_curve(args.output_dir, fpr, tpr, roc_auc, "accuracy-ROC.png", "ROC Curve")
