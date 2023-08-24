#!/bin/bash

CORPUS=$1 # "news_zh" 
MODEL=$2 # text-curie-001, text-davinci-002, text-davinci-003

KEYFILE="key.txt"
RESULTDIR="results/zh/exp1_word-prediction"
DATAFILE="datasets/exp1/${CORPUS}/corpus.csv"

mkdir -p $RESULTDIR

for EVAL_TYPE in "direct" "metaQuestionSimple" "metaInstruct" "metaQuestionComplex"; do
    OUTFILE="${RESULTDIR}/${CORPUS}_${MODEL}_${EVAL_TYPE}.json"

    echo "Running Experiment 1 (word prediction): model = ${MODEL}; eval_type = ${EVAL_TYPE}"
    python run_exp1_word-prediction.py \
        --model $MODEL \
        --lang "zh" \
        --model_type "openai" \
        --key $KEYFILE \
        --eval_type ${EVAL_TYPE} \
        --data_file $DATAFILE --out_file ${OUTFILE}
done
