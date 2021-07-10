sklearn.preprocessing.StandardScaler






pca_list = []
for image in image_dataframe:
    image = StandardScaler().fit_transform(image)
    #Function: StandardScaler Funktion standariert die Features dadurch dass sie den Durchschnitt (Mean) entfernt und zum Unit Variance skaliert.
    # z = (x - u) / s, wo u = Mean of Training Samples und s = Standard Deviation
    #Mean und Standard Deviation werden gespeichert für spätere Nutzung mit der Funktion 'Transform'
    #fit_transform(X[, y]) mean to fit to data und transform it, X ist ein Array oder Sparse/Scattered Matrix of shape (n_samples, n_features)
    pca = skdecomp.PCA(variance) #Linear Dimensionality Reduction mithilfe Singular Value Decomposition (SVD) aus den Dateien um es in niedrigerer Dimensional Space zu konvertieren.
    #Bevor SVD appliziert wird, ist Die Eingabedatei schon zentriert aber nicht skaliert für jedes Features.
    pca.fit(image)
    components = pca.transform(image)
    projected = pca.inverse_transform(components)
    if projected is not None:
        pca_list.append(projected)
return pca_list
















