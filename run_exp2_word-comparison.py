# ~~~~~~~~~~~~~~~~~~~ EXPERIMENT 2: WORD COMPARISON

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import io


if __name__ == "__main__":
    TASK = "word_comparison"
    
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
        # Present a particular order of the answer options.
        if args.option_order == "goodFirst":
            options = [row.good_continuation, row.bad_continuation]
        else:
            options = [row.bad_continuation, row.good_continuation]

        # Create prompt and get outputs.
        good_prompt, logprob_of_good_continuation, logprobs_good = \
            model.get_logprob_of_continuation(
                row.prefix, 
                row.good_continuation, 
                task=TASK,
                options=options,
                return_dist=True,
                **kwargs
            )
        bad_prompt, logprob_of_bad_continuation, logprobs_bad = \
            model.get_logprob_of_continuation(
                row.prefix, 
                row.bad_continuation, 
                task=TASK,
                options=options,
                return_dist=True,
                **kwargs
            )
        
        # Store results in dictionary.
        res = {
            "item_id": row.item_id,
            "prefix": row.prefix,
            "good_prompt": good_prompt,
            "good_continuation": row.good_continuation,
            "bad_continuation": row.bad_continuation,
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

        # Record results for current item.
        results.append(res)

    # Combine meta information with model results into one dict.
    output = {
        "meta": meta_data,
        "results": results
    }

    # Save outputs to specified JSON file.
    io.dict2json(output, args.out_file)