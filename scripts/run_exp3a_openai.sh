#!/bin/bash

CORPUS=$1 # "syntaxgym" or "blimp"
MODEL=$2 # text-curie-001, text-davinci-002, text-davinci-003

KEYFILE="key.txt"
RESULTDIR="results/exp3a_sentence-judgment"
DATAFILE="datasets/exp3/${CORPUS}/corpus.csv"

mkdir -p $RESULTDIR

# Helper function
run_experiment () {
    # Capture relevant variables
    local EVAL_TYPE=$1

    # Define variable-dependent file/folder names
    OUTFILE="${RESULTDIR}/${CORPUS}_${MODEL}_${EVAL_TYPE}.json"

    # Run the evaluation script
    echo "Running Experiment 3a (sentence judgment): model = ${MODEL}; eval_type = ${EVAL_TYPE}" >&2
    python run_exp3a_sentence-judgment.py \
        --model $MODEL \
        --model_type "openai" \
        --key $KEYFILE \
        --eval_type ${EVAL_TYPE} \
        --data_file $DATAFILE --out_file ${OUTFILE}
}

# NOTE: "direct" model is THE SAME across 3a and 3b
# FIRST, run the "direct" model.
run_experiment "direct"

# # NEXT, run the other models.
for EVAL_TYPE in "metaQuestionSimple" "metaInstruct" "metaQuestionComplex"; do
    run_experiment "${EVAL_TYPE}"
done
