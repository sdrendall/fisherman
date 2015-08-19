from pylab import figure, imshow, ginput, show, close, xlim, ylim, subplots
from itertools import imap
from fisherman import data_io

def refine_training_set(training_set):
    """
    Displays each training example in the given training set.  The user clicks
     on the centroid of the displayed training example (read cell), and the new centroid
     is set in training_set.labels
    """
    
    for i, (centroid, label) in enumerate(training_set.labels):
        print centroid
        print label
        (start_row, end_row, start_col, end_col) = training_set.get_example_boundries(centroid)
        figure()
        xlim(start_col, end_col)
        ylim(start_row, end_row)
        imshow(training_set.image)

        new_centroid = map(round, imap(reversed, ginput(1)).next())
        training_set.labels[i] = (new_centroid, label)
        
        close()
