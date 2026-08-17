from sklearn.tree import DecisionTreeClassifier


def create_model():
    """Return the Decision Tree classifier used in the assignment."""
    return DecisionTreeClassifier(max_depth=5, random_state=42)
