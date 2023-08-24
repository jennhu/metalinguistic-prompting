The raw news data was downloaded from [NewsData](https://newsdata.io/search-news), a free provider of recent news articles.
The final corpus file was constructed in the following way:

At various dates in August 2023 (August 22, 2023 at around 9pm and August 24, 2023 around 1pm), 
I searched for news from the previous 48 hours. I set the language to “Chinese”,
and downloaded the first 10 results (there is a limit of 10 results for free downloads). 
These raw results, downloaded from NewsData, can be found in the folder [`raw_newsdata`](raw_newsdata).

I then ran the script `python scrape.py` to scrape the articles from the URLs provided by NewsData. This resulted in the file `cleaned.csv`.

Finally, I ran the script `python make_corpus.py` to create a corpus in the next-word-prediction task format, found at `corpus.csv`.