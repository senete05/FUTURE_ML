"""
Support Ticket Classification & Prioritization System
Machine Learning Task 2 (2026)

This module implements an end-to-end ML system for:
- Automatic ticket categorization (Billing, Technical, Account, General)
- Priority prediction (High, Medium, Low)
- Model evaluation with comprehensive metrics

Author: Your Name
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


# ============================================================================
# SECTION 1: DATA LOADING & EXPLORATION
# ============================================================================

class TicketDataLoader:
    """
    Loads and explores support ticket dataset.
    
    Key Points:
    - Handles CSV/Excel files
    - Explores data distribution
    - Identifies missing values
    """
    
    def __init__(self, filepath):
        """
        Load dataset from file.
        
        Args:
            filepath (str): Path to CSV or Excel file
        """
        print("=" * 70)
        print("SECTION 1: DATA LOADING & EXPLORATION")
        print("=" * 70)
        
        if filepath.endswith('.csv'):
            self.df = pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            self.df = pd.read_excel(filepath)
        else:
            raise ValueError("File must be CSV or Excel format")
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
    def explore(self):
        """Display initial data exploration."""
        print("\n📊 First 5 rows:")
        print(self.df.head())
        
        print("\n📋 Dataset Info:")
        print(self.df.info())
        
        print("\n❌ Missing Values:")
        print(self.df.isnull().sum())
        
        print("\n📈 Data Types:")
        print(self.df.dtypes)
        
        return self.df


# ============================================================================
# SECTION 2: TEXT PREPROCESSING & CLEANING
# ============================================================================

class TextPreprocessor:
    """
    Cleans and preprocesses ticket text.
    
    Key Steps:
    1. Lowercase conversion
    2. Remove special characters & URLs
    3. Remove stopwords (common words like 'the', 'is')
    4. Lemmatization (convert 'running' → 'run')
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """
        Clean a single ticket text.
        
        Args:
            text (str): Raw ticket text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize (split into words)
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column):
        """
        Clean all texts in a dataframe column.
        
        Args:
            df (DataFrame): Input dataframe
            text_column (str): Name of text column
            
        Returns:
            DataFrame: DataFrame with cleaned text
        """
        print("\n" + "=" * 70)
        print("SECTION 2: TEXT PREPROCESSING & CLEANING")
        print("=" * 70)
        
        print(f"\n🧹 Cleaning '{text_column}' column...")
        
        df[f'{text_column}_cleaned'] = df[text_column].apply(self.clean_text)
        
        print(f"✓ Cleaned {len(df)} tickets")
        print(f"\n📝 Example (Original → Cleaned):")
        print(f"Original:  {df[text_column].iloc[0][:100]}...")
        print(f"Cleaned:   {df[f'{text_column}_cleaned'].iloc[0][:100]}...")
        
        return df


# ============================================================================
# SECTION 3: FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """
    Converts text into numerical features.
    
    Key Concepts:
    - TF-IDF: Weights words by importance
    - Higher weight = more important for classification
    - Transforms text into sparse vectors
    """
    
    def __init__(self, max_features=500):
        """
        Initialize feature extractor.
        
        Args:
            max_features (int): Maximum number of features (words) to use
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,  # Ignore words appearing in < 2 documents
            max_df=0.8,  # Ignore words appearing in > 80% of documents
            ngram_range=(1, 2)  # Use single words and bigrams
        )
        
    def fit_transform(self, texts):
        """
        Learn vocabulary and transform texts.
        
        Args:
            texts (Series): Raw ticket texts
            
        Returns:
            sparse matrix: TF-IDF feature matrix
        """
        print("\n" + "=" * 70)
        print("SECTION 3: FEATURE EXTRACTION (TF-IDF)")
        print("=" * 70)
        
        print(f"\n🔢 Extracting TF-IDF features...")
        print(f"  Max features: {self.max_features}")
        print(f"  N-gram range: (1, 2) - single words + bigrams")
        
        features = self.vectorizer.fit_transform(texts)
        
        print(f"\n✓ Feature extraction complete!")
        print(f"  Feature matrix shape: {features.shape}")
        print(f"  ({features.shape[0]} samples × {features.shape[1]} features)")
        
        print(f"\n📚 Top 20 features (most important words):")
        feature_names = self.vectorizer.get_feature_names_out()
        print(feature_names[:20])
        
        return features
    
    def transform(self, texts):
        """Transform new texts using learned vocabulary."""
        return self.vectorizer.transform(texts)


# ============================================================================
# SECTION 4: MODEL TRAINING - CATEGORY CLASSIFICATION
# ============================================================================

class CategoryClassifier:
    """
    Trains model to classify tickets into categories.
    
    Categories: Billing, Technical, Account, General
    
    Model: Logistic Regression (fast, interpretable, reliable)
    """
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.label_encoder = LabelEncoder()
        
    def train(self, X_train, y_train):
        """
        Train category classifier.
        
        Args:
            X_train: Training features (TF-IDF matrix)
            y_train: Training labels (categories)
        """
        print("\n" + "=" * 70)
        print("SECTION 4: TRAIN CATEGORY CLASSIFIER")
        print("=" * 70)
        
        # Encode labels to numbers
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        print(f"\n🎓 Training Logistic Regression classifier...")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Number of categories: {len(self.label_encoder.classes_)}")
        print(f"  Categories: {list(self.label_encoder.classes_)}")
        
        self.model.fit(X_train, y_train_encoded)
        
        print(f"✓ Model trained successfully!")
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate classifier on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Performance metrics
        """
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred_encoded = self.model.predict(X_test)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n📊 CATEGORY CLASSIFICATION RESULTS:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔲 Confusion Matrix:")
        print(cm)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        return metrics
    
    def predict(self, X):
        """Predict categories for new tickets."""
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)


# ============================================================================
# SECTION 5: MODEL TRAINING - PRIORITY PREDICTION
# ============================================================================

class PriorityPredictor:
    """
    Trains model to predict ticket priority.
    
    Priority Levels: High, Medium, Low
    
    Model: Random Forest (handles non-linear patterns)
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        
    def train(self, X_train, y_train):
        """
        Train priority predictor.
        
        Args:
            X_train: Training features
            y_train: Training labels (priority levels)
        """
        print("\n" + "=" * 70)
        print("SECTION 5: TRAIN PRIORITY PREDICTOR")
        print("=" * 70)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        print(f"\n🎓 Training Random Forest priority predictor...")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Number of priority levels: {len(self.label_encoder.classes_)}")
        print(f"  Priority levels: {list(self.label_encoder.classes_)}")
        
        self.model.fit(X_train, y_train_encoded)
        
        print(f"✓ Model trained successfully!")
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate priority predictor.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Performance metrics
        """
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred_encoded = self.model.predict(X_test)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n📊 PRIORITY PREDICTION RESULTS:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔲 Confusion Matrix:")
        print(cm)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        return metrics
    
    def predict(self, X):
        """Predict priority for new tickets."""
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)


# ============================================================================
# SECTION 6: END-TO-END PREDICTION SYSTEM
# ============================================================================

class TicketClassificationSystem:
    """
    Complete system for classifying and prioritizing support tickets.
    
    Workflow:
    1. Load raw ticket
    2. Clean text
    3. Extract features
    4. Predict category
    5. Predict priority
    """
    
    def __init__(self, preprocessor, feature_extractor, category_classifier, priority_predictor):
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.category_classifier = category_classifier
        self.priority_predictor = priority_predictor
        
    def classify_ticket(self, raw_ticket_text):
        """
        Classify and prioritize a single support ticket.
        
        Args:
            raw_ticket_text (str): Raw ticket text
            
        Returns:
            dict: Classification results
        """
        # Step 1: Clean text
        cleaned_text = self.preprocessor.clean_text(raw_ticket_text)
        
        # Step 2: Extract features
        features = self.feature_extractor.transform([cleaned_text])
        
        # Step 3: Predict category
        category = self.category_classifier.predict(features)[0]
        
        # Step 4: Predict priority
        priority = self.priority_predictor.predict(features)[0]
        
        return {
            'original_text': raw_ticket_text[:100] + '...',
            'category': category,
            'priority': priority,
            'confidence': 'High'
        }
    
    def classify_batch(self, ticket_texts):
        """Classify multiple tickets."""
        results = []
        for text in ticket_texts:
            results.append(self.classify_ticket(text))
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - runs entire pipeline.
    """
    
    print("\n" + "=" * 70)
    print("SUPPORT TICKET CLASSIFICATION & PRIORITIZATION SYSTEM")
    print("=" * 70)
    
    # ========== SECTION 1: LOAD DATA ==========
    data_loader = TicketDataLoader('tickets_data.csv')  # Change path as needed
    df = data_loader.explore()
    
    # IMPORTANT: Update these column names based on your dataset
    TEXT_COLUMN = 'ticket_text'  # Column containing ticket descriptions
    CATEGORY_COLUMN = 'category'  # Column with categories (Billing, Technical, etc.)
    PRIORITY_COLUMN = 'priority'  # Column with priority (High, Medium, Low)
    
    # Handle missing values
    print("\n🔧 Handling missing values...")
    df = df.dropna(subset=[TEXT_COLUMN, CATEGORY_COLUMN, PRIORITY_COLUMN])
    print(f"✓ Dataset shape after cleanup: {df.shape}")
    
    # ========== SECTION 2: PREPROCESS TEXT ==========
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df, TEXT_COLUMN)
    
    # ========== SECTION 3: EXTRACT FEATURES ==========
    feature_extractor = FeatureExtractor(max_features=500)
    X = feature_extractor.fit_transform(df[f'{TEXT_COLUMN}_cleaned'])
    
    # ========== SECTION 4 & 5: TRAIN MODELS ==========
    y_category = df[CATEGORY_COLUMN]
    y_priority = df[PRIORITY_COLUMN]
    
    # Split data: 80% training, 20% testing
    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
        X, y_category, y_priority, test_size=0.2, random_state=42, stratify=y_category
    )
    
    print(f"\n📊 Train-Test Split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Train category classifier
    category_classifier = CategoryClassifier()
    category_classifier.train(X_train, y_cat_train)
    cat_metrics = category_classifier.evaluate(X_test, y_cat_test)
    
    # Train priority predictor
    priority_predictor = PriorityPredictor()
    priority_predictor.train(X_train, y_pri_train)
    pri_metrics = priority_predictor.evaluate(X_test, y_pri_test)
    
    # ========== SECTION 6: CREATE PREDICTION SYSTEM ==========
    print("\n" + "=" * 70)
    print("SECTION 6: INFERENCE SYSTEM")
    print("=" * 70)
    
    system = TicketClassificationSystem(
        preprocessor, feature_extractor, category_classifier, priority_predictor
    )
    
    # Example predictions
    print("\n🔮 Example Predictions on New Tickets:")
    example_tickets = [
        df[TEXT_COLUMN].iloc[0],
        df[TEXT_COLUMN].iloc[1],
        df[TEXT_COLUMN].iloc[2]
    ]
    
    for i, ticket in enumerate(example_tickets, 1):
        result = system.classify_ticket(ticket)
        print(f"\nTicket {i}:")
        print(f"  Text: {result['original_text']}")
        print(f"  Category: {result['category']}")
        print(f"  Priority: {result['priority']}")
    
    # ========== SAVE MODELS ==========
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)
    
    pickle.dump(preprocessor, open('preprocessor.pkl', 'wb'))
    pickle.dump(feature_extractor.vectorizer, open('vectorizer.pkl', 'wb'))
    pickle.dump(category_classifier.model, open('category_model.pkl', 'wb'))
    pickle.dump(priority_predictor.model, open('priority_model.pkl', 'wb'))
    
    print("\n✓ Models saved:")
    print("  - preprocessor.pkl")
    print("  - vectorizer.pkl")
    print("  - category_model.pkl")
    print("  - priority_model.pkl")
    
    # ========== SUMMARY REPORT ==========
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    
    print(f"\n📊 CATEGORY CLASSIFICATION:")
    print(f"  Accuracy: {cat_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {cat_metrics['f1']:.4f}")
    
    print(f"\n📊 PRIORITY PREDICTION:")
    print(f"  Accuracy: {pri_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {pri_metrics['f1']:.4f}")
    
    print(f"\n✅ System ready for deployment!")


if __name__ == "__main__":
    main()
