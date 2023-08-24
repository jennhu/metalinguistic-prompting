import pandas as pd
from zh_sentence.tokenizer import tokenize as sent_tokenize
from langdetect import detect
import jieba
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
df.dropna(inplace=True)
print("Number of items after dropping nan", len(df.index))
    
for _, row in tqdm(list(df.iterrows()), total=len(df.index)):
    title = row.title
    text = row.raw_text
    
    if detect(title) == "en":
        print("Detected English title! Title:", title)
        continue
    elif detect(text) == "en":
        print("Detected English text! Text snippet:", text[:20])
        continue

    sentences = sent_tokenize(text)
    first_sentence = sentences[0]
    
    tokens = [
        tok[0] for tok in jieba.tokenize(first_sentence)
    ]
    # include both Chinese and English punctuation marks
    eos_marks = ["。", "？", "！", "」", "》", ".", "?", "!", " "]
    while tokens[-1] in eos_marks:
        tokens = tokens[:-1]

    # Obtain prefix (what language model conditions on).
    prefix = "".join(tokens[:-1])
    if INCLUDE_TITLE:
        prefix = title + TITLE_SEP + prefix

    # Obtain continuation (what language model predicts).
    continuation = tokens[-1]

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