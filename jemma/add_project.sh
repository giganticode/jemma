#!/usr/bin/env bash
python3 ./jemma_contrib_prep_meta.py --project_path $1 
python3 ./jemma_contrib_prep_repr.py --project_path $1 
python3 ./jemma_contrib_prep_prop.py --project_path $1 
python3 ./jemma_contrib_prep_calg.py --project_path $1 

