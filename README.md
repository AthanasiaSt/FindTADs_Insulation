# TAD Detection – Master's Project

This is a Python script made for a Master's assignment.  
It works on Hi-C interaction data, builds contact matrices, and tries to find
Topologically Associating Domains (TADs) using insulation scores and some basic signal processing.

It’s mainly for learning purposes and not meant to be a perfect or optimized TAD caller 🙂

---

## What the code does

- Reads Hi-C interaction data  
- Builds a symmetric contact frequency matrix  
- Splits the matrix per chromosome  
- Plots contact maps (heatmaps + sparse plots)  
- Calculates insulation scores for each bin  
- Smooths the insulation profile  
- Uses a delta vector method to find possible TAD boundaries  
- Estimates boundary strength  
- Runs simple t-tests around boundaries  
- Saves all results into a CSV file  

---

## Input

The script expects this file: interactions_HindIII_fdr0.01_intra.txt

(Tab-separated Hi-C intra-chromosomal interaction data)



