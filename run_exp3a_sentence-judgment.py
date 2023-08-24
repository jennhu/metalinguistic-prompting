# ~~~~~~~~~~~~~~~~~~~ EXPERIMENT 3A: SENTENCE JUDGMENT

import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

from utils import io


if __name__ == "__main__":
    TASK = "sentence_judge"
    
    # Parse command-line arguments.
    args = io.parse_args()
    
    # Set random seed.
    np.random.seed(args.seed)

    # Meta information.
    meta_data = {
        "model": args.model,
        "lang": args.lang,
        "seed": args.seed,
        "task": TASK,
        "eval_type": args.eval_type,
        "data_file": args.data_file,
        "timestamp": io.timestamp()
    }
    
    # Set up model and other model-related variables.
    model = io.initialize_model(args)
    kwargs = {}

    # Read corpus data.
    df = pd.read_csv(args.data_file)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN LOOP
    # Initialize results and get model outputs on each item.
    results = []
    for _, row in tqdm(list(df.iterrows()), total=len(df.index)):
        good_sentence = row.good_sentence
        bad_sentence = row.bad_sentence
        
        if args.eval_type == "direct":
            # Get standard full-sentence probabilities.
            logprob_of_good_sentence = model.get_full_sentence_logprob(
                good_sentence
            )
            logprob_of_bad_sentence = model.get_full_sentence_logprob(
                bad_sentence
            )

            # Store results in dictionary.
            res = {
                "item_id": row.item_id,
                "good_sentence": good_sentence,
                "bad_sentence": bad_sentence,
                "logprob_of_good_sentence": logprob_of_good_sentence,
                "logprob_of_bad_sentence": logprob_of_bad_sentence
            }
            
        else:
            # Create "continuations". We're essentially asking the models
            # a yes/no question.
            if args.lang == "en":
                yes_continuation = "Yes"
                no_continuation = "No"
            elif args.lang == "zh":
                yes_continuation = "是"
                no_continuation = "否"
            else:
                raise ValueError("Language must be 'en' (English) or 'zh' (Chinese).")
                
            # Create prompt and get outputs (2x2).
            good_prompt_yes, logprob_of_yes_good, logprobs_good = \
                model.get_logprob_of_continuation(
                    good_sentence,
                    yes_continuation,
                    task=TASK,
                    return_dist=True,
                    **kwargs
                )
            _, logprob_of_no_good, _ = \
                model.get_logprob_of_continuation(
                    good_sentence,
                    no_continuation,
                    task=TASK,
                    return_dist=True,
                    **kwargs
                )
            _, logprob_of_yes_bad, logprobs_bad = \
                model.get_logprob_of_continuation( 
                    bad_sentence,
                    yes_continuation, 
                    task=TASK,
                    return_dist=True,
                    **kwargs
                )
            _, logprob_of_no_bad, _ = \
                model.get_logprob_of_continuation( 
                    bad_sentence,
                    no_continuation, 
                    task=TASK,
                    return_dist=True,
                    **kwargs
                )
            
            # Store results in dictionary.
            res = {
                "item_id": row.item_id,
                "good_prompt_yes": good_prompt_yes,
                "good_sentence": good_sentence,
                "bad_sentence": bad_sentence,
                "logprob_of_yes_good_sentence": logprob_of_yes_good,
                "logprob_of_yes_bad_sentence": logprob_of_yes_bad,
                "logprob_of_no_good_sentence": logprob_of_no_good,
                "logprob_of_no_bad_sentence": logprob_of_no_bad
            }
            
            # Deal with logprobs: different cases for OpenAI and Huggingface.
            # if args.model_type == "openai":
            #     res["top_logprobs"] = logprobs
            # elif args.dist_folder is not None:
            #     # Save full distribution over vocab items 
            #     # (only corresponding to the first subword token).
            #     model.save_dist_as_numpy(
            #         logprobs, 
            #         f"{args.dist_folder}/{row.item_id}.npy"
            #     )

        # Record results for this item.
        results.append(res)

    # Combine meta information with model results into one dict.
    output = {
        "meta": meta_data,
        "results": results
    }

    # Save outputs to specified JSON file.
    io.dict2json(output, args.out_file)