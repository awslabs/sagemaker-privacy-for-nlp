# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

import argparse
import os
from os.path import join
import json
import random
import time
import logging
import subprocess
from pathlib import Path
from glob import glob

import torch
from torch import nn
from torch import optim

from torchtext import data
from torchtext.data import Field, TabularDataset

from torchtext.vocab import Vectors

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)


class FastText(nn.Module):
    # The model is taken from the excellent Torchtext tutorial at
    # https://github.com/bentrevett/pytorch-sentiment-analysis/
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        embedded = embedded.permute(1, 0, 2)

        # embedded = [batch size, sent len, emb dim]

        pooled = nn.functional.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)

        # pooled = [batch size, embedding_dim]

        return self.fc(pooled)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    for batch in iterator:
        predictions = model(batch.review).squeeze(1)

        loss = criterion(predictions, batch.sentiment)

        acc = binary_accuracy(predictions, batch.sentiment)

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / iterator.iterations, epoch_acc / iterator.iterations


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_one_epoch(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.review).squeeze(1)

        loss = criterion(predictions, batch.sentiment)

        acc = binary_accuracy(predictions, batch.sentiment)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train(model, train_iterator, valid_iterator, n_epochs, model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float('inf')

    model_path = join(model_dir, "model.pt")

    for epoch in range(n_epochs):

        print(f'Epoch: {epoch + 1:02} started...')

        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


def create_fields():
    TEXT = Field(sequential=True, tokenize="basic_english")

    LABEL = data.LabelField(dtype=torch.float)

    return TEXT, LABEL


def create_iterators(train_data, valid_data):
    # Create iterators
    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        sort=False,
        device=device)

    return train_iterator, valid_iterator


def create_model(input_dimensions, embedding_size, pad_idx, unk_idx, pretrained_embeddings):
    model = FastText(input_dimensions, embedding_size, output_dim=1, pad_idx=pad_idx)

    model.embedding.weight.data.copy_(pretrained_embeddings)

    # Set <unk> and <pad> token vectors to all zero
    model.embedding.weight.data[unk_idx] = torch.zeros(embedding_size)
    model.embedding.weight.data[pad_idx] = torch.zeros(embedding_size)

    return model


if __name__ == '__main__':
    SEED = 42

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--vocab-size', type=int, default=25_000)
    parser.add_argument('--embedding-size', type=int, default=300)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--vectors-dir', type=str, default=os.environ['SM_CHANNEL_VECTORS'])
    parser.add_argument('--vectors-filename', type=str, default='glove.6B.300d.txt.gz')
    parser.add_argument('--train-filename', type=str, default='train_examples.csv')

    args, _ = parser.parse_known_args()

    LOG.info("Loading data...")

    TEXT, LABEL = create_fields()

    fields = [('review', TEXT),
              ('sentiment', LABEL)]

    # Torchtext expects a single file, so we concatenate the partial output files
    train_file = Path("{}/{}".format(os.environ['SM_CHANNEL_TRAIN'], args.train_filename))
    if not train_file.exists():
        part_files = glob("{0}/part-*".format(os.environ['SM_CHANNEL_TRAIN']))
        subprocess.check_call(["cat"] + part_files, stdout=train_file.open(mode='w'))

    assert train_file.exists()
    reviews = TabularDataset(
        path=str(train_file), format='csv',
        fields=fields,
        skip_header=True)

    train_data, valid_data = reviews.split(
        split_ratio=[.9, .1], random_state=random.seed(SEED))

    # Create vocabs
    MAX_VOCAB_SIZE = args.vocab_size
    vectors = Vectors(args.vectors_dir + "/" + args.vectors_filename)

    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors=vectors,
                     unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(train_data)

    train_iterator, valid_iterator = create_iterators(train_data, valid_data)

    LOG.info("Instantiating model...")
    model = create_model(
        len(TEXT.vocab),
        args.embedding_size,
        TEXT.vocab.stoi[TEXT.pad_token],
        TEXT.vocab.stoi[TEXT.unk_token],
        TEXT.vocab.vectors)

    LOG.info("Starting training...")
    train(model, train_iterator, valid_iterator, args.epochs, args.model_dir)

    # Save vocab, we'll need them for testing later
    vocab_path = join(args.model_dir, "vocab.pt")
    torch.save(TEXT.vocab, vocab_path)

    # Keep track of experiment settings
    json_file = join(args.model_dir, "training-settings.json")
    with open(json_file, 'w') as f:
        f.write(json.dumps(vars(args)))


