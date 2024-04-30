import pandas as pd
from rdflib import URIRef, Literal
from urllib.parse import quote_plus

def get_feature_importance(data, ontology, target_column):
    feature_importance = {}
    target_uri = URIRef(f"http://example.com/ontology/{quote_plus(target_column)}")
    
    for column in data.columns:
        if column != target_column:
            column_uri = URIRef(f"http://example.com/ontology/{quote_plus(column)}")
            correlation = data[column].corr(data[target_column])
            categorical_count = len(data[column].dropna().unique())
            
            subclasses = [subclass for subclass in ontology.subjects(predicate=URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf"), object=column_uri)]
            is_numerical = any(subclass == URIRef("http://example.com/ontology/NumericalFeature") for subclass in subclasses)
            is_categorical = any(subclass == URIRef("http://example.com/ontology/CategoricalFeature") for subclass in subclasses)
            
            feature_importance[column] = {
                'correlation': correlation,
                'is_numerical': is_numerical,
                'is_categorical': is_categorical,
                'categorical_count': categorical_count if is_categorical else None
            }
    
    return feature_importance

def identify_uninformative_features(feature_importance, correlation_threshold, categorical_threshold):
    uninformative_features = []
    for feature, info in feature_importance.items():
        correlation = info['correlation']
        is_numerical = info['is_numerical']
        is_categorical = info['is_categorical']
        categorical_count = info['categorical_count']

        if is_numerical and abs(correlation) < correlation_threshold:
            uninformative_features.append(feature)
        elif is_categorical and categorical_count <= categorical_threshold:
            uninformative_features.append(feature)

    return uninformative_features