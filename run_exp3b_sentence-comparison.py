# ~~~~~~~~~~~~~~~~~~~ EXPERIMENT 3B: SENTENCE COMPARISON

import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

from utils import io


if __name__ == "__main__":
    TASK = "sentence_comparison"
    
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
        "option_order": args.option_order,
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
            # Present a particular order of the answer options.
            if args.option_order == "goodFirst":
                options = [good_sentence, bad_sentence]
            else:
                options = [bad_sentence, good_sentence]
                
            # Create "continuations". We're essentially asking the models
            # a multiple choice question. NOTE: this is language-independent.
            good_continuation = "1" if args.option_order == "goodFirst" else "2"
            bad_continuation = "2" if args.option_order == "goodFirst" else "1"
                
            # Create prompt and get outputs.
            good_prompt, logprob_of_good_continuation, logprobs_good = \
                model.get_logprob_of_continuation(
                    "", # no "prefix"
                    good_continuation,
                    task=TASK,
                    options=options,
                    return_dist=True,
                    **kwargs
                )
            bad_prompt, logprob_of_bad_continuation, logprobs_bad = \
                model.get_logprob_of_continuation( 
                    "", # no "prefix"
                    bad_continuation, 
                    task=TASK,
                    options=options,
                    return_dist=True,
                    **kwargs
                )
            
            # Store results in dictionary.
            res = {
                "item_id": row.item_id,
                "good_prompt": good_prompt,
                "good_sentence": good_sentence,
                "bad_sentence": bad_sentence,
                "good_continuation": good_continuation,
                "bad_continuation": bad_continuation,
                "logprob_of_good_continuation": logprob_of_good_continuation,
                "logprob_of_bad_continuation": logprob_of_bad_continuation
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