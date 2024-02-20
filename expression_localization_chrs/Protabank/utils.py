import matplotlib.pyplot as plt
import numpy as np

# weird matplotlib error
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

s = '''
A	Alanine	
C	Cysteine	
D	Aspartic acid	
E	Glutamic acid	
F	Phenylalanine	
G	Glycine	
H	Histidine	
I	Isoleucine	
K	Lysine	
L	Leucine	
M	Methionine	
N	Asparagine	
P	Proline	
Q	Glutamine	
R	Arginine	
S	Serine	
T	Threonine	
V	Valine	
W	Tryptophan	
Y	Tyrosine	
'''

def constructAminoAcidMap(s):
    
    s = s.split("\n")
    s = [x.split("\t")[:-1] for x in s]
    s = {x[0]: x[1] for x in s if x}
    return s

aa_map = constructAminoAcidMap(s)

def plotLosses(name, train_loss, val_loss):
    plt.plot(list(range(0, len(train_loss))), train_loss, label='train_loss')
    plt.plot(list(range(0, len(train_loss))), val_loss, label='val_loss')
    
    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('{}: Epoch vs MSE'.format(name))
    
    # Adding a legend to distinguish between the two lists
    plt.legend()
    
    # Display the plot
    plt.show()

# data: DF; target_column: String
def random_test_train_split(data, input_column='Sequence', target_column='Data', test_size=0.2, random_seed=10):
    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # Shuffle the indices
    indices = np.random.permutation(data.index)

    # Calculate the split index
    split_index = int(len(indices) * (1 - test_size))

    # Split the data and target
    train_indices, test_indices = indices[:split_index], indices[split_index:]
    train_data, test_data = data.loc[train_indices], data.loc[test_indices]
    train_target, test_target = train_data.pop(target_column), test_data.pop(target_column)

    return list(train_data[input_column]), list(test_data[input_column]), list(train_target), list(test_target)