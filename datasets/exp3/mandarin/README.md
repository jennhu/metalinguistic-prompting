The folder [wang2021](wang2021) contains data downloaded from [this GitHub folder](https://github.com/YiwenWang03/syntactic-generalization-mandarin/tree/main/test_suites), which contains the test suites released by [Wang et al. 2021](https://aclanthology.org/2021.emnlp-main.454/).

The script `make_corpus.py` formats the test suite CSVs into the format expected by our model evaluation wrapper.
Note that we do not consider the Garden-Path test suites.