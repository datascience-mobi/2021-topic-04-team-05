C =

def loss_function (x,w,y):
    #calculate hinge loss
    N = x.shape[0]  #number of rows in x = number of samples
    seperation = 1 - y * (np.dot(x,w))  #calculates distance of x to hyperplane
    seperation = [0 if i < 0 else i for i in seperation] #all negative seperation values are replaced by 0
    hinge_loss = C * (np.sum(seperation) / N)  # average loss because the whole Y is taken & it encompasses several samples

    # calculate loss
    loss = 1 / 2 * np.dot(w, w) + hinge_loss  # np.dot (W,W) ist gleich Betrag von w^2
    return loss

#calculate gradient of the loss function which then has to be minimized
def lagrange (x,w,y): #ggf. x_sample, y_sample
    if type(y) =! type(np.array):
        y = np.array([y])
        x = np.array([x])