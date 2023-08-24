# ~~~~~~~~~~~~~~~~~~~ EXPERIMENT 1: WORD PREDICTION

import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

from utils import io


if __name__ == "__main__":
    TASK = "word_pred"
    
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
        # Create prompt and get outputs.
        prompt, logprob_of_continuation, logprobs = \
            model.get_logprob_of_continuation(
                row.prefix,
                row.continuation,
                task=TASK,
                options=None,
                return_dist=True,
                **kwargs # defined above in model initialization
            )
        
        # Store results in dictionary.
        res = {
            "item_id": row.item_id,
            "prefix": row.prefix,
            "prompt": prompt,
            "gold_continuation": row.continuation,
            "logprob_of_gold_continuation": logprob_of_continuation,
        }
        
        # Deal with logprobs: different cases for OpenAI and Huggingface.
        if args.model_type == "openai":
            res["top_logprobs"] = logprobs
        elif args.dist_folder is not None:
            # Save full distribution over vocab items 
            # (only corresponding to the first subword token).
            model.save_dist_as_numpy(
                logprobs, 
                f"{args.dist_folder}/{row.item_id}.npy"
            )

        # Record results for current item.
        results.append(res)

    # Combine meta information with model results into one dict.
    output = {
        "meta": meta_data,
        "results": results
    }

    # Save outputs to specified JSON file.
    io.dict2json(output, args.out_file)