"""
Decision Tree and Naive Bayes Model Visualization

This program visualizes the decision tree and Naive Bayes models shown in the slides,
and compares them.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn import tree
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

# Custom decision tree visualization
def visualize_custom_decision_tree():
    """Visualize a decision tree for boat rental decisions"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')  # Hide axes
    
    # Node positions
    nodes = {
        'root': (5, 8),         # Weather node
        'sunny_child': (3, 6),   # Condition node (Sunny)
        'cloudy_child': (7, 6),  # Condition node (Cloudy)
        'sunny_good': (2, 4),    # Result node (Good)
        'sunny_bad': (4, 4),    # Result node (Bad)
        'cloudy_good': (6, 4),  # Result node (Good)
        'cloudy_bad': (8, 4)    # Result node (Bad)
    }
    
    # Node colors
    node_colors = {
        'feature': '#F5F5F5',  # Light gray (feature nodes)
        'good': '#4CAF50',     # Green (good)
        'bad': '#F44336'       # Red (bad)
    }
    
    # Root node (Weather)
    circle = plt.Circle(nodes['root'], 0.8, color=node_colors['feature'], ec='black')
    ax.add_patch(circle)
    ax.text(nodes['root'][0], nodes['root'][1], "Weather", ha='center', va='center', fontsize=14)
    
    # Sunny subtree
    circle = plt.Circle(nodes['sunny_child'], 0.8, color=node_colors['feature'], ec='black')
    ax.add_patch(circle)
    ax.text(nodes['sunny_child'][0], nodes['sunny_child'][1], "Condition", ha='center', va='center', fontsize=14)
    
    # Cloudy subtree
    circle = plt.Circle(nodes['cloudy_child'], 0.8, color=node_colors['feature'], ec='black')
    ax.add_patch(circle)
    ax.text(nodes['cloudy_child'][0], nodes['cloudy_child'][1], "Condition", ha='center', va='center', fontsize=14)
    
    # Sunny & Good
    rect = plt.Rectangle((nodes['sunny_good'][0]-0.8, nodes['sunny_good'][1]-0.5), 1.6, 1,
                        color=node_colors['good'], ec='black')
    ax.add_patch(rect)
    ax.text(nodes['sunny_good'][0], nodes['sunny_good'][1], "Rent Boat", ha='center', va='center', fontsize=14)
    
    # Sunny & Bad
    rect = plt.Rectangle((nodes['sunny_bad'][0]-0.8, nodes['sunny_bad'][1]-0.5), 1.6, 1,
                        color=node_colors['bad'], ec='black')
    ax.add_patch(rect)
    ax.text(nodes['sunny_bad'][0], nodes['sunny_bad'][1], "Don't Rent\nBoat", ha='center', va='center', fontsize=14)
    
    # Cloudy & Good
    rect = plt.Rectangle((nodes['cloudy_good'][0]-0.8, nodes['cloudy_good'][1]-0.5), 1.6, 1,
                        color=node_colors['good'], ec='black')
    ax.add_patch(rect)
    ax.text(nodes['cloudy_good'][0], nodes['cloudy_good'][1], "Rent Boat", ha='center', va='center', fontsize=14)
    
    # Cloudy & Bad
    rect = plt.Rectangle((nodes['cloudy_bad'][0]-0.8, nodes['cloudy_bad'][1]-0.5), 1.6, 1,
                        color=node_colors['bad'], ec='black')
    ax.add_patch(rect)
    ax.text(nodes['cloudy_bad'][0], nodes['cloudy_bad'][1], "Don't Rent\nBoat", ha='center', va='center', fontsize=14)
    
    # Draw connections
    # Root -> Sunny
    ax.plot([nodes['root'][0], nodes['sunny_child'][0]],
           [nodes['root'][1]-0.8, nodes['sunny_child'][1]+0.8], 'k-')
    ax.text((nodes['root'][0] + nodes['sunny_child'][0])/2 - 0.5,
           (nodes['root'][1] + nodes['sunny_child'][1])/2, "Sunny", fontsize=12)
    
    # Root -> Cloudy
    ax.plot([nodes['root'][0], nodes['cloudy_child'][0]],
           [nodes['root'][1]-0.8, nodes['cloudy_child'][1]+0.8], 'k-')
    ax.text((nodes['root'][0] + nodes['cloudy_child'][0])/2 + 0.5,
           (nodes['root'][1] + nodes['cloudy_child'][1])/2, "Cloudy", fontsize=12)
    
    # Sunny -> Sunny & Good
    ax.plot([nodes['sunny_child'][0], nodes['sunny_good'][0]],
           [nodes['sunny_child'][1]-0.8, nodes['sunny_good'][1]+0.5], 'k-')
    ax.text((nodes['sunny_child'][0] + nodes['sunny_good'][0])/2 - 0.5,
           (nodes['sunny_child'][1] + nodes['sunny_good'][1])/2, "Good", fontsize=12)
    
    # Sunny -> Sunny & Bad
    ax.plot([nodes['sunny_child'][0], nodes['sunny_bad'][0]],
           [nodes['sunny_child'][1]-0.8, nodes['sunny_bad'][1]+0.5], 'k-')
    ax.text((nodes['sunny_child'][0] + nodes['sunny_bad'][0])/2 + 0.5,
           (nodes['sunny_child'][1] + nodes['sunny_bad'][1])/2, "Bad", fontsize=12)
    
    # Cloudy -> Cloudy & Good
    ax.plot([nodes['cloudy_child'][0], nodes['cloudy_good'][0]],
           [nodes['cloudy_child'][1]-0.8, nodes['cloudy_good'][1]+0.5], 'k-')
    ax.text((nodes['cloudy_child'][0] + nodes['cloudy_good'][0])/2 - 0.5,
           (nodes['cloudy_child'][1] + nodes['cloudy_good'][1])/2, "Good", fontsize=12)
    
    # Cloudy -> Cloudy & Bad
    ax.plot([nodes['cloudy_child'][0], nodes['cloudy_bad'][0]],
           [nodes['cloudy_child'][1]-0.8, nodes['cloudy_bad'][1]+0.5], 'k-')
    ax.text((nodes['cloudy_child'][0] + nodes['cloudy_bad'][0])/2 + 0.5,
           (nodes['cloudy_child'][1] + nodes['cloudy_bad'][1])/2, "Bad", fontsize=12)
    
    plt.title('Decision Tree for Boat Rental Decisions', fontsize=16)
    plt.tight_layout()
    plt.savefig('custom_decision_tree_en.png', dpi=300)
    plt.show()

# Create sample data
def create_sample_data():
    """Create sample data for decision tree and Naive Bayes models"""
    # Weather: Sunny(1), Cloudy(2)
    # Condition: Good(1), Bad(2)
    # Injury: Yes(1), No(2)
    # Result(Boat rental): Yes(1), No(0)
    
    # Data based on the slide example
    data = {
        'Weather': ['Sunny', 'Sunny', 'Sunny', 'Sunny', 'Cloudy', 'Cloudy', 'Cloudy', 'Cloudy', 'Cloudy', 'Cloudy'],
        'Condition': ['Good', 'Good', 'Bad', 'Bad', 'Good', 'Good', 'Good', 'Bad', 'Bad', 'Bad'],
        'Injury': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes'],
        'Boat_Rental': ['Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }
    
    df = pd.DataFrame(data)
    return df

# Train and visualize decision tree using scikit-learn
def train_and_visualize_decision_tree(df):
    """Train and visualize a decision tree using scikit-learn"""
    # Split features and target
    X = df[['Weather', 'Condition', 'Injury']]
    y = df['Boat_Rental']
    
    # Convert categorical features to numbers
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)
    
    # Create and train decision tree model
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf.fit(X_encoded, y)
    
    # Visualize decision tree
    plt.figure(figsize=(12, 8))
    tree.plot_tree(clf, 
                  feature_names=['Weather', 'Condition', 'Injury'],
                  class_names=['No', 'Yes'],
                  filled=True,
                  fontsize=10)
    plt.title('scikit-learn Decision Tree', fontsize=16)
    plt.tight_layout()
    plt.savefig('sklearn_decision_tree_en.png', dpi=300)
    plt.show()
    
    return clf, encoder

# Train Naive Bayes model
def train_naive_bayes(df):
    """Train a Naive Bayes model"""
    # Split features and target
    X = df[['Weather', 'Condition', 'Injury']]
    y = df['Boat_Rental'].map({'Yes': 1, 'No': 0})
    
    # Convert categorical features to numbers
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)
    
    # Create and train Naive Bayes model
    clf = CategoricalNB()
    clf.fit(X_encoded, y)
    
    return clf, encoder

# Visualize Naive Bayes predictions
def visualize_naive_bayes_predictions(clf, encoder, test_cases):
    """Visualize predictions from the Naive Bayes model"""
    plt.figure(figsize=(10, 6))
    
    # Encode test cases
    test_cases_encoded = encoder.transform(test_cases[['Weather', 'Condition', 'Injury']])
    
    # Make predictions and calculate probabilities
    predictions = clf.predict(test_cases_encoded)
    probabilities = clf.predict_proba(test_cases_encoded)
    
    # Calculate results
    results = []
    for i, (_, row) in enumerate(test_cases.iterrows()):
        prob_yes = probabilities[i][1]
        prob_no = probabilities[i][0]
        odds = prob_yes / prob_no if prob_no > 0 else float('inf')
        prediction = 'Yes' if predictions[i] == 1 else 'No'
        
        results.append({
            'Weather': row['Weather'],
            'Condition': row['Condition'],
            'Injury': row['Injury'],
            'P(Yes)': f'{prob_yes:.3f}',
            'P(No)': f'{prob_no:.3f}',
            'Odds': f'{odds:.3f}',
            'Prediction': prediction
        })
    
    # Create result DataFrame and display
    results_df = pd.DataFrame(results)
    print("Naive Bayes Prediction Results:")
    print(results_df)
    
    # Visualize as a table
    fig, ax = plt.subplots(figsize=(12, len(results_df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=results_df.values,
                    colLabels=results_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color cells based on prediction
    for i in range(len(results_df)):
        table[(i+1, 6)].set_facecolor('#4CAF50' if results_df.iloc[i]['Prediction'] == 'Yes' else '#F44336')
        table[(i+1, 6)].set_text_props(color='white')
    
    plt.title('Naive Bayes Prediction Results', fontsize=16)
    plt.tight_layout()
    plt.savefig('naive_bayes_predictions_en.png', dpi=300)
    plt.show()
    
    return results_df

# Compare decision tree and Naive Bayes models
def compare_models(tree_clf, nb_clf, encoder, test_cases):
    """Compare predictions from decision tree and Naive Bayes models"""
    # Encode test cases
    test_cases_encoded = encoder.transform(test_cases[['Weather', 'Condition', 'Injury']])
    
    # Decision tree predictions
    tree_predictions = tree_clf.predict(test_cases_encoded)
    tree_pred_labels = ['Yes' if p else 'No' for p in tree_predictions]
    
    # Naive Bayes predictions
    nb_predictions = nb_clf.predict(test_cases_encoded)
    nb_probabilities = nb_clf.predict_proba(test_cases_encoded)
    nb_pred_labels = ['Yes' if p == 1 else 'No' for p in nb_predictions]
    
    # Calculate results
    results = []
    for i, (_, row) in enumerate(test_cases.iterrows()):
        prob_yes = nb_probabilities[i][1]
        prob_no = nb_probabilities[i][0]
        odds = prob_yes / prob_no if prob_no > 0 else float('inf')
        
        results.append({
            'Weather': row['Weather'],
            'Condition': row['Condition'],
            'Injury': row['Injury'],
            'Decision Tree': tree_pred_labels[i],
            'Naive Bayes': nb_pred_labels[i],
            'NB Odds': f'{odds:.3f}'
        })
    
    # Create result DataFrame and display
    results_df = pd.DataFrame(results)
    print("Model Comparison Results:")
    print(results_df)
    
    # Visualize as a table
    fig, ax = plt.subplots(figsize=(12, len(results_df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=results_df.values,
                    colLabels=results_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.1, 0.1, 0.1, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color cells based on predictions
    for i in range(len(results_df)):
        table[(i+1, 3)].set_facecolor('#4CAF50' if results_df.iloc[i]['Decision Tree'] == 'Yes' else '#F44336')
        table[(i+1, 3)].set_text_props(color='white')
        table[(i+1, 4)].set_facecolor('#4CAF50' if results_df.iloc[i]['Naive Bayes'] == 'Yes' else '#F44336')
        table[(i+1, 4)].set_text_props(color='white')
    
    plt.title('Decision Tree and Naive Bayes Model Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('model_comparison_en.png', dpi=300)
    plt.show()
    
    return results_df

def main():
    print("Decision Tree and Naive Bayes Model Visualization and Comparison")
    print("-" * 60)
    
    # Custom decision tree visualization
    print("\n1. Boat Rental Decision Tree Visualization")
    visualize_custom_decision_tree()
    
    # Create sample data
    print("\n2. Sample Data Creation")
    df = create_sample_data()
    print(df)
    
    # Train and visualize scikit-learn decision tree
    print("\n3. scikit-learn Decision Tree Training and Visualization")
    tree_clf, encoder = train_and_visualize_decision_tree(df)
    
    # Train Naive Bayes model
    print("\n4. Naive Bayes Model Training")
    nb_clf, _ = train_naive_bayes(df)
    
    # Create test cases
    print("\n5. Test Case Creation")
    test_cases = pd.DataFrame([
        {'Weather': 'Sunny', 'Condition': 'Good', 'Injury': 'No'},  # Slide example 1
        {'Weather': 'Cloudy', 'Condition': 'Good', 'Injury': 'No'},  # Slide example 2
    ])
    print(test_cases)
    
    # Visualize Naive Bayes predictions
    print("\n6. Naive Bayes Prediction Visualization")
    nb_results = visualize_naive_bayes_predictions(nb_clf, encoder, test_cases)
    
    # Compare models
    print("\n7. Decision Tree and Naive Bayes Model Comparison")
    compare_results = compare_models(tree_clf, nb_clf, encoder, test_cases)
    
    print("\nProgram Completed")

if __name__ == "__main__":
    main() 