import pandas as pd

df = pd.read_csv("clean_DTFit_human_dat.csv")

data = []
for item_num in sorted(df.ItemNum.unique()):
    rows = df[df.ItemNum==item_num]
    good = rows[rows.Plausibility=="Plausible"].squeeze()
    bad = rows[rows.Plausibility=="Implausible"].squeeze()

    good_sentence = good.Sentence
    bad_sentence = bad.Sentence
    prefix = " ".join(good_sentence.split(" ")[:-1])
    good_continuation = good_sentence.split(" ")[-1].replace(".", "")
    bad_continuation = bad_sentence.split(" ")[-1].replace(".", "")
    
    data.append(dict(
        item_id=item_num,
        prefix=prefix,
        good_continuation=good_continuation,
        bad_continuation=bad_continuation,
        category="event_plausibility",
        good_human_score=good.Score,
        bad_human_score=bad.Score
    ))

clean_df = pd.DataFrame(data)
clean_df.to_csv("corpus.csv", index=False)