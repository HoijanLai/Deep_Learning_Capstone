import numpy as np
def sample_display(X, y, plt, random_idc = None):
    # plt is matplotlib.pyplot object 
    import numpy as np
    plt.rcParams["figure.figsize"] = (15.0, 15.0)
    f, ax = plt.subplots(nrows = 1, ncols = 10)
    if random_idc is None:
        random_idc = np.random.randint(0, X.shape[0], 10)
    for i, j in enumerate(random_idc):
        ax[i].axis('off')
        ax[i].set_title(y[j], loc = 'center')
        ax[i].imshow(X[j], cmap = 'gray')
        
        
        
# peek the distribution of the data
def peek_distribution(y, plt, title = None, num_labels = 10):
    y_stat = [len(np.where(y == float(i))[0]) for i in range(num_labels)]
    ind = np.arange(10)
    width = .7
    plt.bar(ind, y_stat, width = width)
    plt.ylabel('Number')
    plt.xlabel('label')
    plt.title(title)
    plt.xticks(ind + .5*width, ind)
    plt.show
