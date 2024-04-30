from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import mutual_info_classif

def anova_feature_selection(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected_features = selector.get_support(indices=True)
    return X.iloc[:, selected_features]

def mutual_information_feature_selection(X, y, k=10):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X, y)
    selected_features = selector.get_support(indices=True)
    return X.iloc[:, selected_features]