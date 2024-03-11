#! /bin/bash

module load anaconda3/2022.05
module load cuda/11.8

# TODO to include conda installations
# for general script

source activate <conda_env>
cd <path_to_here>/diaparser-diaparser

python -m diaparser.cmds.biaffine_dependency train -b -d 0      -p=<path_to_here>/diaparser-sbert/exp/sbert-combined-sbatch/model --train=<path_to_here>/combined/train.conllu        --dev=<path_to_here>/combined/dev.conllu        --test=<path_to_here>/combined/test.conllu -f=bert --bert=<path_to_here>/src/utils/LatinBERT/latin_bert/ --punct --sbert --sbertpath=<path_to_here>/latin_sbert
