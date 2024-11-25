import pandas as pd

def sequences_to_fasta(df, name, file_name):
    # Initialize an empty string to store the FASTA format sequences
    fasta = ''

    #labels = df[name].values
    # Iterate over each row in the DataFrame
    for i, row in df.iterrows():
        # Get the sequence string
        sequence = row['Sequence']
        label = row[name]
        
        # Create the FASTA header
        header = f'>{label}'
        
        # Add the sequence and header to the FASTA string
        fasta += f'{header}\n{sequence}\n'

    # Write the FASTA string to a file
    with open(file_name, 'w') as file:
        file.write(fasta)
    file.close()

    return

'''
my_dataframe = pd.read_csv('uniprot_human_proteome.csv')
sequences_to_fasta(my_dataframe, 'Entry', 'uniprot_human_proteome.fasta')
'''