from sklearn.ensemble import RandomForestClassifier


# Turn the following into a function
def get_train_valid(train, valid, predictors, target):
    train_X = train[predictors]
    train_Y = train[target].values
    valid_X = valid[predictors]
    valid_Y = valid[target].values
    return train_X, train_Y, valid_X, valid_Y


def get_rf_model(
    train_X, train_Y, valid_X, n_jobs, random_state, criterion, n_estimators
):
    clf = RandomForestClassifier(
        n_jobs=n_jobs,
        random_state=random_state,
        criterion=criterion,
        n_estimators=n_estimators,
        verbose=False,
    )

    clf.fit(train_X, train_Y)

    preds_tr = clf.predict(train_X)

    preds = clf.predict(valid_X)
    return clf, preds_tr, preds
