import pandas as pd

with open("stimuli_384sentences_dereferencedpronouns.txt", "r") as fp:
    lines = [l.strip() for l in fp.readlines()]

result = []
for i, sentence in enumerate(lines):
    if sentence.endswith("."):
        sentence_no_eos = sentence[:-1]
    else:
        print(f"Skipping sentence: \"{sentence}\"")
        continue

    # Rough way to chop off the final word.
    splits = sentence_no_eos.split(" ")
    prefix = " ".join(splits[:-1])
    final_word = splits[-1]
    result.append(dict(
        item_id=i+1, # 1-indexed sentence
        prefix=prefix,
        continuation=final_word
    ))
        
df = pd.DataFrame(result)
df.to_csv("corpus.csv", index=False)