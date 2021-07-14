sklearn.preprocessing.StandardScaler






pca_list = []
for image in image_dataframe:
    image = StandardScaler().fit_transform(image) #fit_transform ist oft für machine laearning
    #Function: StandardScaler Funktion standariert die Features dadurch dass sie den Durchschnitt (Mean) entfernt und zum Unit Variance skaliert.
    # z = (x - u) / s, wo u = Mean of Training Samples und s = Standard Deviation
    #scales the features have to be zero mean and standard deviation one -> properties of standard normal distribution. It doesnt change the shape of the distribution of the features
    #Mean und Standard Deviation werden gespeichert für spätere Nutzung mit der Funktion 'Transform'
    #standard scaler ist halt ein preprocessing vor PCA
    #fit_transform(X[, y]) mean to fit to data und transform it, X ist ein Array oder Sparse/Scattered Matrix of shape (n_samples, n_features)
    pca = skdecomp.PCA(variance) #Linear Dimensionality Reduction mithilfe Singular Value Decomposition (SVD) aus den Dateien um es in niedrigerer Dimensional Space zu konvertieren.
    #Bevor SVD appliziert wird, ist Die Eingabedatei schon zentriert aber nicht skaliert für jedes Features.
    pca.fit(image) #fit the model with the image array > calculate loading scores and the variation each principal component accounts for
    components = pca.transform(image) #applying dimensionality reduction to image array -> generating coordinates for a PCA Graph based on the loading scores and the scaled data
    projected = pca.inverse_transform(components) #transform data back into its original space -> In other words, return an input X_original whose transform would be X.
    #parameter:X_original array-like, shape (n_samples, n_features)
    #return: X_original array-like, shape (n_samples, n_features)

    #You can only expect that the result has the same dimension if the number of components you specify is the same as the dimensionality of the input data.
    #that means the variance should be n
    if projected is not None:
        pca_list.append(projected)
return pca_list
















