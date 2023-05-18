from newspaper import Article
import pandas as pd
from tqdm import tqdm

from os import listdir

folder = "raw_newsdata"
files = [f for f in listdir(folder) if f.startswith("Newsdata_Records")]

data = []
for f in files:
    df = pd.read_csv(f"{folder}/{f}")
    date = f.split("_")[-1].replace(".csv", "")

    for _, row in tqdm(list(df.iterrows()), total=len(df.index)):
        url = row["ARTICLE LINK"]
        try:
            article = Article(url)
            article.download()
            article.parse()
        except:
            print(f"Encountered error for {f}, {url}")
            continue
        raw_text = article.text
        noads = raw_text.replace("\nAdvertisement\n", "")
        data.append(dict(
            download_date=date,
            published_date_gmt=row["PUBLISHED DATE (GMT)"],
            title=article.title,
            url=url,
            category=row.CATEGORY,
            raw_text=noads
        ))

texts = pd.DataFrame(data)
print(texts.head())
print("Number of texts:", len(texts.index))
texts.to_csv(f"cleaned.csv", index=False)