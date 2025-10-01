# train.py
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

def main():
    print("🔑 Loading dataset (requires Hugging Face login)…")
    ds = load_dataset("Gamortsey/url_defanged", split="train")

    raw_urls = ds["url_defanged"]
    raw_labels = ds["label"]

    # Clean dataset: keep only valid rows
    urls, labels = [], []
    for u, l in zip(raw_urls, raw_labels):
        if u is None or str(u).strip() == "":
            continue
        if l is None:
            continue
        try:
            l = int(l)
        except Exception:
            continue
        urls.append(str(u))
        labels.append(l)

    print(f"✅ Cleaned dataset: {len(urls)} samples remain")

    # Vectorize URLs (character n-grams)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=5000)
    X = vectorizer.fit_transform(urls)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Train classifier
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n📊 Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

    # Save model + vectorizer
    joblib.dump(clf, "url_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("💾 Saved model as url_model.pkl and vectorizer.pkl")

if __name__ == "__main__":
    main()
