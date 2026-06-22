import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ==============================================================================
# 1. Data Ingestion & Data Preparation
# ==============================================================================
# Read corporate incident dataset and drop records missing critical features or labels
df = pd.read_csv("it_support_tickets.csv")
df = df.dropna(subset=["ticket_text", "category"]).reset_index(drop=True)

X = df["ticket_text"]
y = df["category"]

# ==============================================================================
# 2. Train-Test Partitioning
# ==============================================================================
# Stratify partitions to guarantee uniform category distributions across sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==============================================================================
# 3. Natural Language Processing (Feature Extraction)
# ==============================================================================
# Parse raw textual tokens, strip standard English stopwords, and clip vocabulary limits
vectorizer = CountVectorizer(stop_words="english", max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==============================================================================
# 4. Model Training & Evaluation Inferences
# ==============================================================================
# Fit Multinomial Naive Bayes model and perform inference on the validation segment
clf = MultinomialNB(alpha=1.0)
clf.fit(X_train_vec, y_train)
y_pred = clf.predict(X_test)

# Display structural validation summaries to stdout
print("=" * 60)
print("          SYSTEM CLASSIFICATION PERFORMANCE REPORT          ")
print("=" * 60)
print(classification_report(y_test, y_pred))
print("=" * 60)


# ==============================================================================
# 5. Operational Business Priority Engine
# ==============================================================================
def rule_priority_engine(category, impact):
    """
    Applies operational business rules based on the classification model output
    and original incident impact metrics to route tickets effectively.
    """
    if category in ["Network/VPN"] or impact == "High":
        return "Critical/High"
    elif category in ["Administrative Rights", "Access Request"]:
        return "Medium"
    else:
        return "Low"


# Restructure indices to extract matching impact indicators for priority mapping
eval_results = pd.DataFrame(
    {
        "Predicted_Category": y_pred,
        "True_Impact": df.loc[y_test.index, "impact"].values,
    }
)
eval_results["Final_Priority"] = eval_results.apply(
    lambda r: rule_priority_engine(r["Predicted_Category"], r["True_Impact"]),
    axis=1,
)

# ==============================================================================
# 6. Diagnostic Matrix Visualization
# ==============================================================================
# Compute multi-class classification matrix matching actual labels against predictions
labels = sorted(list(set(y_test)))
cm = confusion_matrix(y_test, y_pred, labels=labels)

# Initialize canvas settings and grid aesthetics
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 8))

# Map numerical counts into visual heatmap matrix
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0.8,
    linecolor="#e3e3e3",
    cbar_kws={"label": "Number of Classified Support Tickets"},
    ax=ax,
)

# Typography settings
ax.set_title(
    "IT Support Ticket Classification: Confusion Matrix Diagnostic",
    fontsize=14,
    fontweight="bold",
    pad=20,
    color="#1c1c1c",
)
ax.set_xlabel(
    "Predicted Operational Category",
    fontsize=11,
    fontweight="semibold",
    labelpad=12,
    color="#333333",
)
ax.set_ylabel(
    "True Operational Category",
    fontsize=11,
    fontweight="semibold",
    labelpad=12,
    color="#333333",
)

# Adjust label text orientation to avoid label truncation
plt.xticks(rotation=35, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.tight_layout()
plt.show()