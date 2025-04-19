import cv2
import numpy as np
from sklearn.neighbors import KDTree
import time
from concurrent.futures import ThreadPoolExecutor
import os
import matplotlib.pyplot as plt

class EnhancedFeatureMatcher:
    def __init__(self, debug=False):
        self.debug = debug
        
        self.orb = cv2.ORB_create(
            nfeatures=3000,      
            scaleFactor=1.05,    
            nlevels=16,          
            edgeThreshold=10,    
            patchSize=31,        
            fastThreshold=20     
        )
        
        self.sift = cv2.SIFT_create(
            nfeatures=2000,          
            nOctaveLayers=6,         
            contrastThreshold=0.01,  
            edgeThreshold=15,        
            sigma=1.3               
        )
        
        self.FLANN_INDEX_KDTREE = 1
        self.flann_params = dict(
            algorithm=self.FLANN_INDEX_KDTREE,
            trees=8  
        )
        self.flann = cv2.FlannBasedMatcher(self.flann_params, {})
        
        self.min_matches = 4      
        self.distance_threshold = 0.75  
        
    def debug_plot(self, image, keypoints, title):
        """Debug helper to visualize keypoints"""
        if self.debug:
            img_with_kp = cv2.drawKeypoints(
                image, 
                keypoints, 
                None, 
                color=(0, 255, 0)
            )
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
            plt.show()

    def extract_features(self, image):
        """Extract features using both ORB and SIFT"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        orb_keypoints, orb_descriptors = self.orb.detectAndCompute(gray, None)
        if self.debug:
            print(f"ORB features detected: {len(orb_keypoints)}")
            self.debug_plot(image, orb_keypoints, "ORB Keypoints")
        
        sift_keypoints, sift_descriptors = self.sift.detectAndCompute(gray, None)
        if self.debug:
            print(f"SIFT features detected: {len(sift_keypoints)}")
            self.debug_plot(image, sift_keypoints, "SIFT Keypoints")
        
        return {
            'orb': (orb_keypoints, orb_descriptors),
            'sift': (sift_keypoints, sift_descriptors)
        }
    
    def build_feature_index(self, descriptors):
        """Build KD-Tree index for fast nearest neighbor search"""
        if descriptors is not None:
            if self.debug:
                print(f"Building KD-Tree index for {descriptors.shape[0]} features")
            return KDTree(descriptors)
        return None
    
    def branch_and_bound_search(self, query_desc, index, bounds):
        """Implement Branch and Bound search for feature matching"""
        if index is None or query_desc is None:
            return []
            
        distances, indices = index.query(
            query_desc, 
            k=2,
            return_distance=True
        )
        
        good_matches = []
        
        for i, (dist1, dist2) in enumerate(zip(distances[:, 0], distances[:, 1])):
            if dist1 < self.distance_threshold * dist2:
                if dist1 < bounds[i]:
                    good_matches.append((i, indices[i][0], dist1))
                    bounds[i] = min(bounds[i], dist1)
                    
        return good_matches
    
    def spatial_verification(self, kp1, kp2, matches):
        """Verify matches using spatial relationships"""
        if len(matches) < self.min_matches:
            return []
            
        src_pts = np.float32([kp1[m[0]].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m[1]].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if self.debug:
            print(f"Spatial verification: {np.sum(mask)} inliers out of {len(matches)} matches")
        
        return [m for m, msk in zip(matches, mask) if msk]
    
    def match_images(self, template_image, target_image):
        """Main function to match features between two images"""
        print("Starting feature matching process...")
        start_time = time.time()
        
        print("Extracting features...")
        template_features = self.extract_features(template_image)
        target_features = self.extract_features(target_image)
        
        all_matches = []
        
        if template_features['orb'][1] is not None and target_features['orb'][1] is not None:
            orb_index = self.build_feature_index(target_features['orb'][1])
            orb_bounds = np.full(len(template_features['orb'][1]), np.inf)
            
            print("Matching ORB features...")
            orb_matches = self.branch_and_bound_search(
                template_features['orb'][1],
                orb_index,
                orb_bounds
            )
            all_matches.extend([
                (i, j, d) for i, j, d in orb_matches
            ])
        
        if template_features['sift'][1] is not None and target_features['sift'][1] is not None:
            sift_index = self.build_feature_index(target_features['sift'][1])
            sift_bounds = np.full(len(template_features['sift'][1]), np.inf)
            
            print("Matching SIFT features...")
            sift_matches = self.branch_and_bound_search(
                template_features['sift'][1],
                sift_index,
                sift_bounds
            )
            orb_len = len(template_features['orb'][0]) if template_features['orb'][0] is not None else 0
            all_matches.extend([
                (i + orb_len, j + len(target_features['orb'][0]), d) 
                for i, j, d in sift_matches
            ])
        
        template_kp = template_features['orb'][0] + template_features['sift'][0]
        target_kp = target_features['orb'][0] + target_features['sift'][0]
        
        print("Performing spatial verification...")
        verified_matches = self.spatial_verification(
            template_kp,
            target_kp,
            all_matches
        )
        
        end_time = time.time()
        print(f"Matching completed in {end_time - start_time:.2f} seconds")
        print(f"Found {len(verified_matches)} verified matches")
        
        return verified_matches, template_kp, target_kp
    
    def visualize_matches(self, template_image, target_image, matches, template_kp, target_kp):
        """Visualize the matched features between images"""
        cv_matches = [cv2.DMatch(m[0], m[1], m[2]) for m in matches]
        
        match_img = cv2.drawMatches(
            template_image,
            template_kp,
            target_image,
            target_kp,
            cv_matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS + 
                  cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        if self.debug:
            plt.figure(figsize=(15, 10))
            plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Feature Matching Results ({len(matches)} verified matches)")
            plt.axis('off')
            plt.show()
        
        return match_img

def process_images(template_path, target_path, output_path, debug=False):
    """Process a pair of images and save the results"""
    matcher = EnhancedFeatureMatcher(debug=debug)
    
    template_image = cv2.imread(template_path)
    target_image = cv2.imread(target_path)
    
    if template_image is None or target_image is None:
        raise ValueError("Could not read one or both images")
    
    matches, template_kp, target_kp = matcher.match_images(
        template_image,
        target_image
    )
    
    if matches:
        result_image = matcher.visualize_matches(
            template_image,
            target_image,
            matches,
            template_kp,
            target_kp
        )
        
        cv2.imwrite(output_path, result_image)
        print(f"Results saved to '{output_path}'")
        
        return len(matches)
    else:
        print("No good matches found")
        return 0 