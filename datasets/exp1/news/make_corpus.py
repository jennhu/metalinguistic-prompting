import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

INCLUDE_TITLE = True
TITLE_SEP = " -- "
data = []
item_num = 0

df = pd.read_csv("cleaned.csv")
print(df.head())
print("Number of items:", len(df.index))
df.drop_duplicates(inplace=True) # in case one news article is double counted
print("Number of items after dropping duplicates:", len(df.index)) 
    
for _, row in tqdm(list(df.iterrows()), total=len(df.index)):
    title = row.title
    text = row.raw_text

    try:
        sentences = sent_tokenize(text)
    except:
        print("Skipping:", title)
        continue
        
    first_sentence = sentences[0]
    splits = first_sentence.split()

    # Obtain prefix (what language model conditions on).
    prefix = " ".join(splits[:-1])
    if INCLUDE_TITLE:
        prefix = title + TITLE_SEP + prefix

    # Obtain continuation (what language model predicts).
    continuation = splits[-1]
    for eos in [".", "?", "!"]:
        if continuation.endswith(eos):
            continuation = continuation[:-1]

    data.append(dict(
        item_id=item_num+1,
        prefix=prefix,
        continuation=continuation,
        category=row.category,
        published_date_gmt=row.published_date_gmt,
        download_date=row.download_date
    ))

    # Update counter.
    item_num += 1

print("Final number of items:", item_num)
corpus = pd.DataFrame(data)
print(corpus.head())
corpus.to_csv("corpus.csv", index=False)