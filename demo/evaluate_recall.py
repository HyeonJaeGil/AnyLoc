import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import cv2

def load_descriptors(folder):
    """Load descriptors from .npy files in a folder."""
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    descriptors = [np.load(os.path.join(folder, f)) for f in files]
    return np.vstack(descriptors), files

def load_poses(pose_file):
    """Load pose information from a txt file."""
    poses = np.loadtxt(pose_file).reshape(-1, 4, 4)
    return poses

def compute_similarity_matrix(desc1, desc2):
    """Compute similarity matrix using cosine similarity."""
    similarity_matrix = 1 - cdist(desc1, desc2, metric='cosine')
    return similarity_matrix

def compute_recall_at_k(similarity_matrix, k):
    """Compute recall@K based on similarity rankings."""
    indices = np.argsort(-similarity_matrix, axis=0)[:k, :]
    return indices

def evaluate_recall(indices, poses1, poses2, threshold):
    """Evaluate recall@K using pose ground truth."""
    recalls = []
    for i in range(indices.shape[1]):  # For each query
        retrieved_indices = indices[:, i]
        gt_pose = poses2[i]
        retrieved_poses = poses1[retrieved_indices]
        errors = np.linalg.norm(retrieved_poses[:, :3, 3] - gt_pose[:3, 3], axis=1)
        recalls.append(any(errors < threshold))
    return np.mean(recalls)

def visualize_results(folder1, folder2, image_folder1, image_folder2, files1, files2, indices, k):
    """Visualize retrieval results for each query."""
    for i, file in enumerate(files2):
        query_img = cv2.imread(os.path.join(image_folder2, file.replace('.npy', '')))
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        retrieved_imgs = [cv2.imread(os.path.join(image_folder1, files1[idx].replace('.npy', ''))) for idx in indices[:, i]]
        retrieved_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in retrieved_imgs]
        combined_img = np.hstack([query_img] + retrieved_imgs)
        plt.figure(figsize=(15, 5))
        plt.imshow(combined_img)
        plt.axis('off')
        plt.title(f'Query: {file}, Top {k} Retrieved')
        plt.show()

def main(folder1, folder2, image_folder1, image_folder2, pose1=None, pose2=None, k=5, threshold=1.0, visualize=False):
    # Check existence
    assert os.path.exists(folder1), f"{folder1} does not exist."
    assert os.path.exists(folder2), f"{folder2} does not exist."
    assert os.path.exists(image_folder1), f"{image_folder1} does not exist."
    assert os.path.exists(image_folder2), f"{image_folder2} does not exist."
    
    if pose1 and pose2:
        assert os.path.exists(pose1), f"{pose1} does not exist."
        assert os.path.exists(pose2), f"{pose2} does not exist."
    
    # Load descriptors
    desc1, files1 = load_descriptors(folder1)
    desc2, files2 = load_descriptors(folder2)
    
    # Load poses if given
    poses1 = load_poses(pose1) if pose1 else None
    poses2 = load_poses(pose2) if pose2 else None
    
    if pose1 and pose2:
        assert len(files1) == len(poses1), "Mismatch between descriptors and poses in folder1"
        assert len(files2) == len(poses2), "Mismatch between descriptors and poses in folder2"
    
    # Check image count matches descriptor count
    img_files1 = sorted([f for f in os.listdir(image_folder1) if f.endswith('.jpg')])
    img_files2 = sorted([f for f in os.listdir(image_folder2) if f.endswith('.jpg')])
    assert len(img_files1) == len(files1), "Mismatch between images and descriptors in folder1"
    assert len(img_files2) == len(files2), "Mismatch between images and descriptors in folder2"
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(desc1, desc2)
    
    # Compute recall@K indices
    indices = compute_recall_at_k(similarity_matrix, k)
    
    result = {files2[i]: [files1[idx] for idx in indices[:, i]] for i in range(len(files2))}
    
    if poses1 is not None and poses2 is not None:
        recall_score = evaluate_recall(indices, poses1, poses2, threshold)
        result['recall@K'] = recall_score
        print(f'Recall@{k}: {recall_score:.4f}')
    
    # Visualization
    if visualize:
        visualize_results(folder1, folder2, image_folder1, image_folder2, files1, files2, indices, k)
    
    return result