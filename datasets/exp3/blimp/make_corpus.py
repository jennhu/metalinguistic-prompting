import pandas as pd
import json
from os import listdir

SEED = 1111

json_files = [f for f in listdir("raw") if f.endswith("jsonl")]
print("Number of raw BLIMP files:", len(json_files))

# Read each file into a dict.
blimp = []
for f in json_files:
    with open(f"raw/{f}", "r") as fp:
        lines = [l.strip() for l in fp.readlines()]
        dicts = [json.loads(l) for l in lines]
        df = pd.DataFrame(dicts)
        blimp.append(df)
blimp = pd.concat(blimp)
print(blimp.head())

simple_lm = blimp[blimp.simple_LM_method]

# Convert into our format.
corpus = []
for _, row in simple_lm.iterrows():
    corpus.append(dict(
        good_sentence=row.sentence_good,
        bad_sentence=row.sentence_bad,
        category=row.field,
        category_fine=row.linguistics_term,
        blimp_UID=row.UID,
        blimp_pairID=row.pairID,
    ))
corpus = pd.DataFrame(corpus)

print(corpus.category_fine.value_counts())
print("Number of categories:", corpus.category_fine.nunique())

# Take a random sample from each category.
all_rows = []
num_items_per_category = 30
for category in corpus.category_fine.unique():
    rows = corpus[corpus.category_fine==category].copy()
    sample = rows.sample(
        n=num_items_per_category,
        replace=False,
        random_state=SEED
    )
    all_rows.append(sample)
# Reset indices.
sample = pd.concat(all_rows).reset_index()
sample["item_id"] = sample.index + 1
sample.drop(columns=["index"], inplace=True)
sample.to_csv("corpus.csv", index=False)