import cv2
import numpy as np
import os
from feature_matcher import process_images
import matplotlib.pyplot as plt

def run_demo():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)
    output_dir = os.path.join(project_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    template_path = os.path.join(project_dir, "images", "template.png")
    target_path = os.path.join(project_dir, "images", "target.jpeg")
    matches_path = os.path.join(output_dir, "matches.jpg")
    
    print("Starting feature matching process...")
    num_matches = process_images(template_path, target_path, matches_path, debug=True)
    print(f"Found {num_matches} verified matches")
    
    result_img = cv2.imread(matches_path)
    if result_img is not None:
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Feature Matching Results ({num_matches} verified matches)")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    run_demo() 