def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss for each combination of w0 and w1.
    # ***************************************************
    w=np.array([w0,w1])
    for i in range(len(w0)):
        for j in range (len(w1)):
            losses[i][j]=compute_loss(y,tx,np.array([w0[i],w1[j]]))
    return losses
    raise NotImplementedError

    
def grid_search_general(y, tx, w):
    """Algorithm for grid search."""
    losses = np.zeros(w.shape)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss for each combination of w0 and w1.
    # ***************************************************
    w=np.array([w0,w1])
    for i in range(len(w0)):
        for j in range (len(w1)):
            losses[i][j]=compute_loss(y,tx,np.array([w0[i],w1[j]]))
    return losses
    raise NotImplementedError