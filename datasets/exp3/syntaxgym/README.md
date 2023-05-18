The script `make_corpus.py` is used to create a corpus in the format for sentence probability comparisons. 
This uses the condition mappings in `conds.py`, which was hand-written based on the success criteria specified in 
[Hu et al. (ACL, 2020)](https://aclanthology.org/2020.acl-main.158/).
It contains 15 items from 23 different categories in the Hu et al. test suites (e.g., number agreement, cleft, filler_gap_dependency).

To run this script, you will first need to download the SyntaxGym test suites in CSV format.
To do this, download the CSV files from the SyntaxGym repository [here](https://github.com/cpllab/syntactic-generalization/tree/master/test_suites/csv).
Then, place them in a folder called `syntaxgym_csv`.