#!/bin/bash

CORPUS="dtfit"
MODEL=$1 # text-curie-001, text-davinci-002, text-davinci-003

KEYFILE="key.txt"
RESULTDIR="results/exp2_word-comparison"
DATAFILE="datasets/exp2/${CORPUS}/corpus.csv"

mkdir -p $RESULTDIR

# Helper function
run_experiment () {
    # Capture relevant variables
    local EVAL_TYPE=$1
    local OPTION_ORDER=$2

    # Define variable-dependent file/folder names
    OUTFILE="${RESULTDIR}/${CORPUS}_${MODEL}_${EVAL_TYPE}_${OPTION_ORDER}.json"

    # Run the evaluation script
    echo "Running Experiment 2 (word comparison): model = ${MODEL}; eval_type = ${EVAL_TYPE}; option_order = ${OPTION_ORDER}" >&2
    python run_exp2_word-comparison.py \
        --model $MODEL \
        --model_type "openai" \
        --key $KEYFILE \
        --option_order ${OPTION_ORDER} \
        --eval_type ${EVAL_TYPE} \
        --data_file $DATAFILE --out_file ${OUTFILE}
}

# FIRST, run the "direct" model. Option order doesn't matter, so set it to goodFirst.
run_experiment "direct" "goodFirst"

# NEXT, run the other models, crossing prompt method with option order.
for OPTION_ORDER in "goodFirst" "badFirst"; do 
    for EVAL_TYPE in "metaQuestionSimple" "metaInstruct" "metaQuestionComplex"; do
        run_experiment "${EVAL_TYPE}" "${OPTION_ORDER}"
    done
done
