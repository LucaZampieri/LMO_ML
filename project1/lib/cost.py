def compute_loss(y, tx, w, fct='mse'):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y-tx.dot(w)
    if (fct=='mse'):
        return 1./(2.*y.shape[0])*e.T@e
    elif fct=='mae':
        return 1./(2*y.shape[0])*np.sum(np.abs(e))
    else:
        raise NotImplementedError