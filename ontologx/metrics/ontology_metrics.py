import functools

from ontologx.store import GraphDocument, Node, Relationship

type _Triple = tuple[str, str, str]  # Triple in the form (subject, predicate, object)


def _triples(graph: GraphDocument) -> list[_Triple]:
    """Get the triples from a graph.

    Args:
        graph (GraphDocument): The graph to extract triples from.

    Returns:
        list[tuple[str,str,str]]: A list of triples in the form (subject, predicate, object).

    """
    triples = []
    for node in graph.nodes:
        for prop, value in node.properties.items():
            if prop == "uri" or prop.startswith("schema"):
                continue
            triples.append((node.type, prop, value.lower() if isinstance(value, str) else value))

    triples.extend([(rel.source.type, rel.type, rel.target.type) for rel in graph.relationships])

    return triples


def _entity_match(entity1: Node, entity2: Node) -> bool:
    """Check if two entities are equal.

    Two entities are considered equal if their types and values match.

    Args:
        entity1 (Node): First entity.
        entity2 (Node): Second entity.

    Returns:
        bool: True if entities are equal, False otherwise.

    """
    props1 = {
        k: (v.lower() if isinstance(v, str) else v)
        for k, v in entity1.properties.items()
        if (k != "uri" and not k.startswith("schema"))  # Exclude uri and schema properties
    }
    props2 = {
        k: (v.lower() if isinstance(v, str) else v)
        for k, v in entity2.properties.items()
        if (k != "uri" and not k.startswith("schema"))  # Exclude uri and schema properties
    }

    return entity1.type == entity2.type and props1 == props2


def _relationship_match(rel1: Relationship, rel2: Relationship) -> bool:
    """Check if two entity relationships are equal.

    Two entity relationships are considered equal if their entities and relationship type match,
    using the `_entity_match` function.

    Args:
        rel1 (Relationship): First entity relationship.
        rel2 (Relationship): Second entity relationship.

    Returns:
        bool: True if entity relationships are equal, False otherwise.

    """
    return (
        rel1.type == rel2.type and _entity_match(rel1.source, rel2.source) and _entity_match(rel1.target, rel2.target)
    )


class OntologyGraphMetrics:
    """Metrics for evaluating ontology graphs.

    This class computes precision, recall, F1 score, entity linking accuracy,
    relationship linking accuracy, using a given predicted and true GraphDocument.
    """

    __tp = 0  # True positives, the number of triples found that are in the true set
    __fp = 0  # False positives, the number of triples found that are not in the true set
    __fn = 0  # False negatives, the number of triples not found that are in the true set

    @functools.cached_property
    def precision(self) -> float:
        """Calculate precision based on true positives and false positives."""
        return self.__tp / (self.__tp + self.__fp) if (self.__tp + self.__fp) > 0 else 0

    @functools.cached_property
    def recall(self) -> float:
        """Calculate recall based on true positives and false negatives."""
        return self.__tp / (self.__tp + self.__fn) if (self.__tp + self.__fn) > 0 else 0

    @functools.cached_property
    def f1(self) -> float:
        """Calculate F1 score based on precision and recall."""
        return (
            2 * (self.precision * self.recall) / (self.precision + self.recall)
            if (self.precision + self.recall) > 0
            else 0
        )

    @functools.cached_property
    def __entities_correct(self) -> list[Node]:
        """Get the list of correctly matched entities."""
        entities_correct = []
        entities_pred = self.__pred.nodes.copy()
        entities_true = self.__true.nodes.copy()
        for entity_pred in entities_pred:
            for entity_true in entities_true:
                if _entity_match(entity_pred, entity_true):
                    entities_correct.append(entity_pred)
                    entities_true.remove(entity_true)
                    break

        return entities_correct

    @functools.cached_property
    def entity_linking_accuracy(self) -> float:
        """Calculate entity linking accuracy based on correct entities."""
        return len(self.__entities_correct) / len(self.__true.nodes)

    @functools.cached_property
    def relationship_linking_accuracy(self) -> float:
        """Calculate relationship linking accuracy based on correct relationships."""
        # Only consider relatioships among correct entities
        candidate_rels = [
            r
            for r in self.__pred.relationships
            if r.source in self.__entities_correct and r.target in self.__entities_correct
        ]

        rels_correct = 0
        rels_true = self.__true.relationships.copy()
        for rel_pred in candidate_rels:
            for rel_true in rels_true:
                if _relationship_match(rel_pred, rel_true):
                    rels_correct += 1
                    rels_true.remove(rel_true)
                    break
        return rels_correct / len(candidate_rels) if candidate_rels else 0

    def __init__(self, pred: GraphDocument, true: GraphDocument):
        """Initialize the evaluator with predictions and true values.

        Args:
            pred (GraphDocument): List of predicted GraphDocuments.
            true (GraphDocument): List of true GraphDocuments.
            store (Store): The store used to manage GraphDocuments.

        """
        self.__pred = pred
        self.__true = true

        triples_pred = _triples(self.__pred)
        triples_true = _triples(self.__true)

        for triple_pred in triples_pred:
            if triple_pred not in triples_true:
                self.__fp += 1
                continue

            self.__tp += 1
            triples_true.remove(triple_pred)

        self.__fn += len(triples_true)
