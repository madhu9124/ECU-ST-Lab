# ECU-ST-Lab

Steps to Prioritize MRs for LLMs:

1) Run run_metrics.py file in the root folder of your project. The code will run the metrics and generate results folder for individual metrics.
2) Run Total_diversity_score_calculation.py file and it will give the diversity score.
3) Follow step1 and step2 for all the MRs and calculate the diversity score.
4) Rank the MRs based on diversity score and use the ranked MRs when testing your LLM.
