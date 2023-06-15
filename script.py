import os

os.system("snakemake -c4 -f create_all_signals")
os.system("snakemake -c4 -f output/timeslides_fm/")
os.system("snakemake -c4 -f output/trained/final_metric_params.npy")
os.system("snakemake -c4 -f output/far_bins.npy")
os.system("snakemake -c4 -f output/plots")
