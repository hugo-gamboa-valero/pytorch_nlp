import pandas as pd
import numpy as np
import collections
import argparse
import torch
import sys
import re
import os

def preprocess_text(text):
    if type(text) == float:
        print(text)
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)    # Add one whitespace before and after the characters listed.
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text) # Remove characters different to the ones listed.

    return text

parser = argparse.ArgumentParser()
parser.add_argument("-d","--directory",help="Input directory")
parser.add_argument("--train", help="Name of train data file. Default='raw_train.csv'", default="raw_train.csv")
parser.add_argument("--test", help="Name of test data file. Default='raw_test.csv'", default="raw_test.csv")

args = parser.parse_args()
train_in_file = os.path.join(args.directory, args.train)
test_in_file = os.path.join(args.directory, args.test)

try:
   train_reviews = pd.read_csv(train_in_file, header=None, names=["rating","review"])
   train_reviews = train_reviews[~pd.isnull(train_reviews)]
   train_reviews = train_reviews[:100]

   test_reviews = pd.read_csv(test_in_file, header=None, names=["rating","review"])
   test_reviews = test_reviews[~pd.isnull(test_reviews)]
   test_reviews = test_reviews[:100]
except IOError as e:
   print("\nIncorrect name of files and/or directory.\n") 
   print(parser.print_help())
   sys.exit()


train_split_by_rating = collections.defaultdict(list)

for _, row in train_reviews.iterrows():
    train_split_by_rating[row.rating].append(row.to_dict())

final_list = []
np.random.seed(42)

for _, item_list in train_split_by_rating.items():
    np.random.shuffle(item_list)
    n_total = len(item_list)
    n_train = int(0.8*n_total)
    n_val = int(0.1*n_total)
    
    for item in item_list[:n_train]:
        item["split"] = "train"
        
    for item in item_list[n_train:]:
        item["split"] = "val"
    
    final_list.extend(item_list)

for _, row in test_reviews.iterrows():
    new_row = row.to_dict()
    new_row["split"] = "test"
    final_list.append(new_row)

preprocess_reviews = pd.DataFrame(final_list)
preprocess_reviews.review = preprocess_reviews.review.apply(preprocess_text)
preprocess_reviews.rating = preprocess_reviews.rating.apply({1:"negative", 2:"positive"}.get)
preprocess_reviews.to_csv("preprocess_data.csv", index=False)


