#!/bin/bash
# custom config
TRAINER=LoCoOp

DATA=$1
DATASET=$2
CFG=$3  # config file
CTP=$4  # class token position (end or middle)
NCTX=$5  # number of context tokens
SHOTS=$6  # number of shots (1, 2, 4, 8, 16)
CSC=$7  # class-specific context (False or True)
lambda=$8
topk=$9
current_time=$(date "+%Y-%m-%d-%H-%M-%S")

for SEED in 1 2 3
do
    DIR=output/${TRAINER}_${DATASET}/${CFG}_${SHOTS}shots_nctx${NCTX}_csc${CSC}_ctp${CTP}_${current_time}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        echo $PWD
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/LoCoOp/${CFG}.yaml \
        --output-dir ${DIR} \
        --lambda_value ${lambda} \
        --topk ${topk} \
        TRAINER.LOCOOP.N_CTX ${NCTX} \
        TRAINER.LOCOOP.CSC ${CSC} \
        TRAINER.LOCOOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
    bash scripts/locoop/eval.sh data imagenet ${CFG} 1 ${DIR}
done