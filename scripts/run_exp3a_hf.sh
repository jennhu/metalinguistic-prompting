#!/bin/bash

CORPUS=$1 # "syntaxgym" or "blimp"
MODEL=$2 # e.g., "google/flan-t5-small"
SAFEMODEL=$3 # e.g., "flan-t5-small"; this should be safe for file-naming purposes

RESULTDIR="results/exp3a_sentence-judgment"
DATAFILE="datasets/exp3/${CORPUS}/corpus.csv"

mkdir -p $RESULTDIR

# Helper function
run_experiment () {
    # Capture relevant variables
    local EVAL_TYPE=$1

    # Define variable-dependent file/folder names
    OUTFILE="${RESULTDIR}/${CORPUS}_${SAFEMODEL}_${EVAL_TYPE}.json"
    
    # By default, we won't save the full vocab distributions.
    # Uncomment the two lines below if you'd like to.
    
    # DISTFOLDER="${RESULTDIR}/dists/${CORPUS}_${SAFEMODEL}_${EVAL_TYPE}"
    # mkdir -p $DISTFOLDER

    # Run the evaluation script
    echo "Running Experiment 3a (sentence judgment): model = ${MODEL}; eval_type = ${EVAL_TYPE}" >&2
    python run_exp3a_sentence-judgment.py \
        --model $MODEL \
        --model_type "hf" \
        --eval_type ${EVAL_TYPE} \
        --data_file $DATAFILE --out_file ${OUTFILE}
        # Uncomment the line below to save full distributions.
        # --dist_folder $DISTFOLDER
}

# NOTE: "direct" model is THE SAME across 3a and 3b
# FIRST, run the "direct" model.
run_experiment "direct"

# # NEXT, run the other models.
for EVAL_TYPE in "metaQuestionSimple" "metaInstruct" "metaQuestionComplex"; do
    run_experiment "${EVAL_TYPE}"
done
