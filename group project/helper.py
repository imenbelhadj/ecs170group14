import matplotlib.pyplot as plt
from IPython import display

plt.ion() # making code iterative

"""
    open another window for plotting scores and mean score 
    to see AI learning
"""
def plot(scores, mean_scores): 
    
    display.clear_output(wait=True) # update the result
    display.display(plt.gcf()) # create new figure if there's no figure
    plt.clf() # updating figure
    
    plt.title('Training...') # name the figure
    plt.xlabel('Number of Games') # name x label
    plt.ylabel('Score') # name y level

    # plotting x and y value
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0) # setting y limit

    # add the latest score and mean score on the figure
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    # display figure
    plt.show(block=False)
    plt.pause(.1)
