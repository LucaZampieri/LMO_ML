def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    e = y-tx.dot(w)
    return -1./y.shape[0]*tx.T@e
    raise NotImplementedError


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad=0
        loss=0
        for minibatch_y, minibatch_tx in batch_iter(y=y, tx=tx, batch_size=batch_size):
            grad += compute_stoch_gradient(minibatch_y,minibatch_tx,w)/batch_size
            loss += compute_loss(minibatch_y,minibatch_tx,w)/batch_size
        w = w-gamma*grad
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws