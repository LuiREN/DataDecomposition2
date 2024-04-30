from rdflib import Graph, Namespace, Literal, URIRef
from urllib.parse import quote_plus

def create_ontology(data):
    # Создание графа онтологии
    graph = Graph()

    # Определение пространства имен
    namespace = Namespace("http://example.com/ontology/")

    # Создание классов и свойств онтологии на основе типов данных
    for column, dtype in data.dtypes.items():
        if dtype == 'float64' or dtype == 'int64':
            class_uri = URIRef(namespace + quote_plus(column))
            graph.add((class_uri, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), URIRef("http://www.w3.org/2000/01/rdf-schema#Class")))
            graph.add((class_uri, URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf"), URIRef("http://example.com/ontology/NumericalFeature")))
        else:
            class_uri = URIRef(namespace + quote_plus(column))
            graph.add((class_uri, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), URIRef("http://www.w3.org/2000/01/rdf-schema#Class")))
            graph.add((class_uri, URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf"), URIRef("http://example.com/ontology/CategoricalFeature")))

    # Связывание данных с онтологией
    for index, row in data.iterrows():
        instance_uri = URIRef(namespace + f"instance_{index}")
        for column, value in row.items():
            property_uri = URIRef(namespace + quote_plus(column))
            graph.add((instance_uri, property_uri, Literal(value)))

    return graph

def create_semantic_db(data):
    ontology = create_ontology(data)
    return ontology.serialize(format='turtle')