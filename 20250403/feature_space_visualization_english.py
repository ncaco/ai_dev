"""
Visualization of Logical Models' Feature Space

This program visually represents the feature space as shown in the slides
using matplotlib.

Slide content:
- Different areas of the feature space can have different classification results
- Showing inconsistent and incomplete regions in the feature space
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def create_feature_space_visualization():
    """
    Visualize the feature space using matplotlib.
    
    Draw the feature space similar to the slide and display 
    classification results and email counts in each region.
    """
    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Axis setup
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Peter = 0', 'Peter = 1'])
    ax.set_yticklabels(['Lottery = 0', 'Lottery = 1'])
    ax.set_xlabel('Peter')
    ax.set_ylabel('Lottery')
    
    # Grid setup
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Color definitions
    colors = {
        'spam': '#FF6B6B',  # Red (spam)
        'ham': '#4ECDC4',   # Teal (ham)
        'contradiction': '#FFD166',  # Yellow (contradiction)
        'no_prediction': '#CCCCCC'   # Gray (no prediction)
    }
    
    # Color regions based on classification results
    # (x, y, width, height, color)
    # Lottery=1, Peter=0: spam
    ax.add_patch(plt.Rectangle((0-0.5, 1-0.5), 1, 1, color=colors['spam'], alpha=0.7))
    # Lottery=1, Peter=1: contradiction (spam takes precedence here)
    ax.add_patch(plt.Rectangle((1-0.5, 1-0.5), 1, 1, color=colors['contradiction'], alpha=0.7))
    # Lottery=0, Peter=1: ham
    ax.add_patch(plt.Rectangle((1-0.5, 0-0.5), 1, 1, color=colors['ham'], alpha=0.7))
    # Lottery=0, Peter=0: no prediction
    ax.add_patch(plt.Rectangle((0-0.5, 0-0.5), 1, 1, color=colors['no_prediction'], alpha=0.7))
    
    # Add text for each region
    # (x, y, text)
    # Lottery=1, Peter=0: spam
    ax.text(0, 1, "spam: 20\nham: 5", ha='center', va='center', fontsize=12)
    # Lottery=1, Peter=1: contradiction
    ax.text(1, 1, "spam: 20\nham: 5\n(contradiction)", ha='center', va='center', fontsize=12)
    # Lottery=0, Peter=1: ham
    ax.text(1, 0, "spam: 10\nham: 5", ha='center', va='center', fontsize=12)
    # Lottery=0, Peter=0: no prediction
    ax.text(0, 0, "spam: 20\nham: 40\n(no prediction)", ha='center', va='center', fontsize=12)
    
    # Add title
    ax.set_title('Feature Space Visualization', fontsize=16)
    
    # Add legend
    legend_elements = [
        Patch(facecolor=colors['spam'], edgecolor='black', alpha=0.7, label='Spam'),
        Patch(facecolor=colors['ham'], edgecolor='black', alpha=0.7, label='Ham'),
        Patch(facecolor=colors['contradiction'], edgecolor='black', alpha=0.7, label='Contradiction'),
        Patch(facecolor=colors['no_prediction'], edgecolor='black', alpha=0.7, label='No Prediction')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add rule explanation
    rule_text = (
        "Rule 1: if Lottery = 1 then Y = spam\n"
        "Rule 2: if Peter = 1 then Y = ham\n\n"
        "- Inconsistency: Lottery=1 & Peter=1\n"
        "- Incompleteness: Lottery=0 & Peter=0"
    )
    plt.figtext(0.15, 0.02, rule_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('feature_space_visualization_en.png', dpi=300)
    print("Visualization image saved as 'feature_space_visualization_en.png'")
    plt.show()

def visualize_decision_tree():
    """
    Visualize the decision tree from the slide using matplotlib.
    """
    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')  # Hide axes
    
    # Node positions
    nodes = {
        'root': (5, 8),         # Viagra node
        'left_child': (3, 5),   # Lottery node (Viagra=0)
        'right_child': (7, 5),  # Spam node (Viagra=1)
        'left_left': (2, 2),    # Ham node (Viagra=0, Lottery=0)
        'left_right': (4, 2)    # Spam node (Viagra=0, Lottery=1)
    }
    
    # Node colors
    node_colors = {
        'feature': '#FFFFFF',  # White (feature nodes)
        'spam': '#FF6B6B',     # Red (spam nodes)
        'ham': '#4ECDC4'       # Teal (ham nodes)
    }
    
    # Draw nodes
    # Root node (Viagra)
    circle = plt.Circle(nodes['root'], 0.8, color=node_colors['feature'], ec='black')
    ax.add_patch(circle)
    ax.text(nodes['root'][0], nodes['root'][1], "'Viagra'", ha='center', va='center', fontsize=12)
    
    # Left child node (Lottery, Viagra=0)
    circle = plt.Circle(nodes['left_child'], 0.8, color=node_colors['feature'], ec='black')
    ax.add_patch(circle)
    ax.text(nodes['left_child'][0], nodes['left_child'][1], "'lottery'", ha='center', va='center', fontsize=12)
    
    # Right child node (Spam, Viagra=1)
    rectangle = plt.Rectangle((nodes['right_child'][0]-1, nodes['right_child'][1]-0.5), 2, 1, 
                             color=node_colors['spam'], ec='black')
    ax.add_patch(rectangle)
    ax.text(nodes['right_child'][0], nodes['right_child'][1], "spam: 20\nham: 5", ha='center', va='center', fontsize=12)
    
    # Left-left node (Ham, Viagra=0, Lottery=0)
    rectangle = plt.Rectangle((nodes['left_left'][0]-1, nodes['left_left'][1]-0.5), 2, 1, 
                             color=node_colors['ham'], ec='black')
    ax.add_patch(rectangle)
    ax.text(nodes['left_left'][0], nodes['left_left'][1], "spam: 20\nham: 40", ha='center', va='center', fontsize=12)
    
    # Left-right node (Spam, Viagra=0, Lottery=1)
    rectangle = plt.Rectangle((nodes['left_right'][0]-1, nodes['left_right'][1]-0.5), 2, 1, 
                             color=node_colors['spam'], ec='black')
    ax.add_patch(rectangle)
    ax.text(nodes['left_right'][0], nodes['left_right'][1], "spam: 10\nham: 5", ha='center', va='center', fontsize=12)
    
    # Draw edges
    # Root → Left child (Viagra=0)
    ax.plot([nodes['root'][0], nodes['left_child'][0]], 
            [nodes['root'][1]-0.8, nodes['left_child'][1]+0.8], 'k-')
    ax.text((nodes['root'][0] + nodes['left_child'][0])/2 - 0.5, 
            (nodes['root'][1] + nodes['left_child'][1])/2, "=0", fontsize=12)
    
    # Root → Right child (Viagra=1)
    ax.plot([nodes['root'][0], nodes['right_child'][0]], 
            [nodes['root'][1]-0.8, nodes['right_child'][1]+0.5], 'k-')
    ax.text((nodes['root'][0] + nodes['right_child'][0])/2 + 0.5, 
            (nodes['root'][1] + nodes['right_child'][1])/2, "=1", fontsize=12)
    
    # Left child → Left-left (Lottery=0)
    ax.plot([nodes['left_child'][0], nodes['left_left'][0]], 
            [nodes['left_child'][1]-0.8, nodes['left_left'][1]+0.5], 'k-')
    ax.text((nodes['left_child'][0] + nodes['left_left'][0])/2 - 0.5, 
            (nodes['left_child'][1] + nodes['left_left'][1])/2, "=0", fontsize=12)
    
    # Left child → Left-right (Lottery=1)
    ax.plot([nodes['left_child'][0], nodes['left_right'][0]], 
            [nodes['left_child'][1]-0.8, nodes['left_right'][1]+0.5], 'k-')
    ax.text((nodes['left_child'][0] + nodes['left_right'][0])/2 + 0.5, 
            (nodes['left_child'][1] + nodes['left_right'][1])/2, "=1", fontsize=12)
    
    # Add title
    ax.set_title('Decision Tree Visualization', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('decision_tree_visualization_en.png', dpi=300)
    print("Decision tree image saved as 'decision_tree_visualization_en.png'")
    plt.show()

if __name__ == "__main__":
    print("Starting feature space visualization program...")
    
    try:
        import matplotlib
        print("Visualizing feature space with graphics.")
        create_feature_space_visualization()
        visualize_decision_tree()
    except ImportError:
        print("matplotlib library is not installed.")
        print("Install it with the command: pip install matplotlib")
    except Exception as e:
        print(f"An error occurred during visualization: {e}")
        
    print("Program terminated.") 