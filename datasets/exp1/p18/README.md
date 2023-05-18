The sentences are taken from Experiment 2 of [Pereira et al. (2016)](https://www.nature.com/articles/s41467-018-03068-4.epdf?author_access_token=OisH9T0MaUh7XRdwZeYENNRgN0jAjWel9jnR3ZoTv0OGyuVxUm7-S4Xbskw1RGuNizgjOecqSbJxCvHfp5njx91ag5KvXUY0uKnqGtgYu7PU-Jt20YASIcmRW2XOBXPzFpVxcJ-SUBv1kC7EpdLrxw%3D%3D). 
Data from the original paper can be found at [https://osf.io/crwz7/](https://osf.io/crwz7/).
The pronouns have been dereferenced, so we can test them on models at the sentence level 
(instead of at the passage level, where sentence context is required).

I ran the script `python make_corpus.py` to separate the final word for each sentence 
and construct the `corpus.csv` file, which is fed to the models.