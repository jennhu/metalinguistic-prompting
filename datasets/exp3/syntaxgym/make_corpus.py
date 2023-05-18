from conds import CONDITION_ORDERINGS
import pandas as pd

all_data = []
global_item_number = 0

for test_suite_name, cond_data in CONDITION_ORDERINGS.items():
    if len(cond_data) > 0:
        print(test_suite_name)
        df = pd.read_csv(f"syntaxgym_csv/{test_suite_name}.csv")
        
        item_numbers = sorted(df.item_number.unique())

        for item_number in item_numbers:
            rows = df[df.item_number == item_number]
            for good_cond, bad_cond in cond_data:
                global_item_number += 1
                good_rows = rows[rows.condition_name == good_cond]
                bad_rows = rows[rows.condition_name == bad_cond]
                good_sentence = " ".join(good_rows.dropna().content)
                bad_sentence = " ".join(bad_rows.dropna().content)
                if not good_sentence.endswith("."):
                    good_sentence += "."
                if not bad_sentence.endswith("."):
                    bad_sentence += "."
                all_data.append(dict(
                    good_sentence=good_sentence.replace(" ,", ",").replace(" .", "."),
                    bad_sentence=bad_sentence.replace(" ,", ",").replace(" .", "."),
                    suite=test_suite_name,
                    good_cond=good_cond,
                    bad_cond=bad_cond,
                    syntaxgym_id=item_number,
                    item_id=global_item_number
                ))

corpus = pd.DataFrame(all_data)
print(corpus.head())
corpus.to_csv("full_corpus.csv", index=False)

print(corpus.suite.value_counts())
print("Number of categories:", corpus.suite.nunique())

# Take a random sample from each category.
SEED = 1111
all_rows = []
num_items_per_category = 15
for category in corpus.suite.unique():
    rows = corpus[corpus.suite==category].copy()
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