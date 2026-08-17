from sklearn.ensemble import RandomForestClassifier


def create_model():
    """Return the Random Forest classifier used in the assignment."""
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
