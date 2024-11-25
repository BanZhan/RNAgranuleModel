import numpy as np
import pandas as pd
import os
import math
import warnings
import scipy.stats as stats
from scipy import io
from datetime import datetime

import subprocess
import random
import re

warnings.filterwarnings('ignore')


######################################################################################################################
def fasta_prep_IDR(seq):
    # Initialize an empty string to store the FASTA format sequences
    fasta = ''
    
    # Iterate over each row in the DataFrame
    for i in range(len(seq)):
        # Get the sequence string
        sequence = seq[i]
        
        # Create the FASTA header
        header = f'>{i}'
        
        # Add the sequence and header to the FASTA string
        fasta += f'{header}\n{sequence}\n'

    # Write the FASTA string to a file
    with open(os.getcwd()+'/iupred3/example.seq', 'w') as file:
        file.write(fasta)
    return

def collect_IDR_scores():
    # define the path to the iupred3.py file
    path_original = os.getcwd()
    iupred3_path = path_original + "/iupred3/iupred3.py"

    # change the working directory to the directory containing the iupred3.py file
    os.chdir(os.path.dirname(iupred3_path))
    # define the command to run the iupred3.py script
    cmd = "python iupred3.py example.seq long"

    # run the command and collect the output
    output = subprocess.check_output(cmd, shell=True)

    # decode the output from bytes to string
    output = output.decode("utf-8")

    # split the output into lines
    lines = output.split("\n")

    # iterate over the lines and collect the results
    results = []
    for line in lines:
        # skip empty lines and comment lines
        if not line.strip() or line.startswith("#"):
            continue

        # split the line into columns
        columns = line.split()

        # extract the amino acid and score
        amino_acid = columns[1]
        score = float(columns[2])

        # add the result to the list of results
        results.append((amino_acid, score))

    # save the binary IUPRED2 scores into a list
    IDR_class = [1 if score > 0.5 else 0 for amino_acid, score in results]
    
    os.chdir(path_original)
    return IDR_class