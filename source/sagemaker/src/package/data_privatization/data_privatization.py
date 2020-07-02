# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

from pathlib import Path
from collections import defaultdict, Counter
import itertools
import re
import random
import argparse
import json
from os.path import join

import numpy as np
from numpy.random import normal

import torch
import torchtext
from torchtext import data
from torchtext.data import Field, TabularDataset
from torchtext.vocab import Vectors

from annoy import AnnoyIndex

from pyspark import SparkFiles
from pyspark.sql import SparkSession

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--vectors-dir', type=str, default='/opt/ml/processing/vectors')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--artifact-output-dir', type=str, default='/opt/ml/processing/artifacts')
    parser.add_argument('--sensitive-filename', type=str, default='train_examples.csv')
    parser.add_argument('--vectors-filename', type=str, default='glove.6B.300d.txt.gz')
    parser.add_argument('--privatized-dir', type=str, default='reviews-privatized')
    parser.add_argument('--sensitive-dir', type=str, default='reviews-sensitive')
    parser.add_argument('--epsilon', type=float, default=23)
    parser.add_argument('--embedding-size', type=int, default=300)

    return parser.parse_args()


def create_clean_counter(input_data, add_space_split=False):
    phrase_count = Counter()
    for example in input_data:
        review = example.review
        original_text = " ".join(review)
        text = original_text.replace(
            " ' ", "").replace("'", "").replace("/", " ").replace("  ", " ").replace('"', '')
        if add_space_split:
            text = re.split('\!|\,|\n|\.|\?|\-|\;|\:|\(|\)|\s', text)
        else:
            text = re.split('\!|\,|\n|\.|\?|\-|\;|\:|\(|\)', text)
        sentences = [x.strip() for x in text if x.strip()]
        for sentence in sentences:
            phrase_count[sentence] += 1
    return phrase_count


def replace_word(sensitive_word, vocab, epsilon, ann_index, embedding_dims, sensitivity=1.0):
    """
    Given a word will inject noise according to the provided epsilon value and return a perturbed word.
    """
    # Generate a noise vector
    noise = generate_laplacian_noise_vector(embedding_dims, sensitivity, epsilon)
    # Get vector of sensitive word
    original_vec = vocab.vectors[vocab.stoi[sensitive_word]]
    # Get perturbed vector
    noisy_vector = original_vec + noise
    # Get item closest to noisy vector
    closest_item = ann_index.get_nns_by_vector(noisy_vector, 1)[0]
    # Get word from item
    privatized_word = vocab.itos[closest_item]
    return privatized_word


def generate_laplacian_noise_vector(dimension, sensitivity, epsilon):
    rand_vec = normal(size=dimension)
    normalized_vec = rand_vec / np.linalg.norm(rand_vec)
    magnitude = np.random.gamma(shape=dimension, scale=sensitivity / epsilon)
    return normalized_vec * magnitude


def clean_example(example):
    # TODO: Docstring
    original_text = " ".join(example.review)
    clean_text = original_text.replace(
        "'", " ").replace("/", " ").replace("  ", " ").replace('"', '')
    text = re.split('\!|\,|\n|\.|\?|\-|\;|\:|\(|\)', clean_text)
    return text


def privatize_example(example, local_vocab, local_embedding_dims, local_epsilon):
    from annoy import AnnoyIndex

    # Load files
    local_index = AnnoyIndex(local_embedding_dims, 'euclidean')
    local_index.load(SparkFiles.get("index.ann"))

    sensitive_phrases = [x.strip() for x in clean_example(example) if x.strip()]

    privatized_phrases = []
    for sensitive_phrase in sensitive_phrases:
        privatized_words = []
        for sensitive_word in sensitive_phrase.split(' '):
            privatized_word = replace_word(
                sensitive_word, local_vocab, local_epsilon, local_index, local_embedding_dims)
            privatized_words.append(privatized_word)

        # Flatten nested list of words
        privatized_phrases.append(itertools.chain(*[privatized_words]))

    privatized_review = " ".join(list(itertools.chain(*privatized_phrases)))

    privatized_row = "\"{}\",{}".format(privatized_review, example.sentiment)

    return privatized_row


if __name__ == '__main__':
    args = parse_args()

    # Ensure artifact output dir exists
    Path(args.artifact_output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare sensitive data
    TEXT = Field(sequential=True, tokenize="basic_english", include_lengths=True)

    LABEL = data.LabelField(dtype=torch.float)

    fields = [('sentiment', LABEL),
              ('title', None),
              ('review', TEXT)]

    reviews = TabularDataset(
        path=args.data_dir + "/" + args.sensitive_filename, format='csv',
        fields=fields,
        skip_header=True)

    train_private = reviews

    phrase_count_complete = create_clean_counter(reviews, add_space_split=True)
    train_vocab = torchtext.vocab.Vocab(counter=phrase_count_complete)

    # Attach GloVe embeddings
    embedding_dims = args.embedding_size
    vectors = Vectors(args.vectors_dir + "/" + args.vectors_filename, max_vectors=100_000)
    train_vocab.load_vectors(vectors)

    # Create approximate nearest neighbor index
    num_trees = 50

    ann_index = AnnoyIndex(embedding_dims, 'euclidean')

    ann_filename = join(args.artifact_output_dir, "index.ann")
    for vector_num, vector in enumerate(train_vocab.vectors):
        ann_index.add_item(vector_num, vector)

    print("Building annoy index...")
    assert ann_index.build(num_trees)
    ann_index.save(ann_filename)
    print("Annoy index built")

    epsilon = args.epsilon

    spark = SparkSession.builder.appName("data-privatization").getOrCreate()

    # Save vocab object for later use
    torch.save(train_vocab, join(args.artifact_output_dir, "vocab.pt"))

    with spark.sparkContext as sc:
        sc.addFile(ann_filename)
        examples = sc.parallelize(train_private, numSlices=500)

        # Privatize each example in the dataset
        privatized_examples = examples.map(
            lambda example: privatize_example(example, train_vocab, embedding_dims, epsilon))
        privatized_examples.saveAsTextFile(args.output_dir + "/" + args.privatized_dir)

        # We also save the sensitive examples, to ensure we train on the same source data later
        examples.map(lambda example: "\"{}\",{}".format(
            " ".join(clean_example(example)), example.sentiment)).saveAsTextFile(
            args.output_dir + "/" + args.sensitive_dir
        )

    print("Privatization done!")
    # Keep track of experiment settings
    json_file = join(args.artifact_output_dir, "data-privatization-settings.json")
    with open(json_file, 'w') as f:
        f.write(json.dumps(vars(args)))
