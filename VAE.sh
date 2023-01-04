#!/bin/bash
eval "$('/zhome/5f/4/147447/miniconda3/bin/conda' 'shell.bash' 'hook')"
conda activate VAE_isoform

python VAE_iso_expression.py --train_file materials/archs4_gene_expression_transposed.tsv --data_file materials/gtex_gene_expression_transposed.tsv > output/VAE_transformed_100epochs.tsv
