import pandas as pd
from os import listdir

all_data = []
global_item_number = 0

test_names = [
    ("Classifier", "cls"),
    # Skip garden-path suites
    # ("GP_obj", "gpo"),
    # ("GP_sub", "gps"),
    ("Missing_Object", "mobj"),
    ("Subordination", "sd"),
    ("Verb_Noun", "vo")
]

def get_suite_name(file_name):
    if file_name.endswith(".txt") or file_name.endswith(".csv"):
        file_name = file_name[:-4]
    splits = file_name.split("_")
    if splits[-1] == "tkd":
        suite_name = "_".join(splits[:-1])
    else:
        suite_name = "_".join(splits)
    return suite_name

def readlines(file_name):
    with open(file_name, "r") as fp:
        lines = fp.readlines()
    return [l.strip() for l in lines]

for test, test_abv in test_names:
    files = sorted(listdir(f"wang2021/{test}"))
    files = [f for f in files if f.lower().startswith(test_abv) and f.endswith(".csv")]
    for f in files:
        # construct suite name based on file name
        suite_name = get_suite_name(f.lower())
        file_name = f"wang2021/{test}/{f}"
        
        # read test suite data
        _df = pd.read_csv(file_name)
        for _, row in _df.iterrows():
            # Formatting/data processing depends on each test suite
            if test == "Classifier":
                g1 = row.Grammatical_1
                u1 = row.Ungrammatical_1
                g2 = row.Grammatical_2
                u2 = row.Ungrammatical_2
                global_item_number += 1
                new_row_1 = dict(
                    good_sentence=g1,
                    bad_sentence=u1,
                    suite=suite_name,
                    suite_broad=test,
                    wang2021_id=row.ID,
                    item_id=global_item_number
                )
                all_data.append(new_row_1)

                global_item_number += 1
                new_row_2 = dict(
                    good_sentence=g2,
                    bad_sentence=u2,
                    suite=suite_name,
                    suite_broad=test,
                    wang2021_id=row.ID,
                    item_id=global_item_number
                )
                all_data.append(new_row_2)

            else:
                global_item_number += 1
                new_row = dict(
                    good_sentence=row.Grammatical,
                    bad_sentence=row.Ungrammatical,
                    suite=suite_name,
                    suite_broad=test,
                    wang2021_id=row.ID,
                    item_id=global_item_number
                )
                all_data.append(new_row)
            
corpus = pd.DataFrame(all_data)
print(corpus.head())
corpus.to_csv("corpus.csv", index=False)

print(corpus.suite.value_counts())
print("Number of categories:", corpus.suite.nunique())
