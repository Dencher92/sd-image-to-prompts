from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import os

# Configurable constants
DATASET_ROOT = '/mnt/home/data/laion/laion2b-en-vit-h-14-embeddings'
OUTPUT_ROOT = '/mnt/home/data/laion/laion2b-en-vit-h-14-embeddings_clustering'
N_JOBS = 8
BATCH_SIZE = 5000


# First-level clustering parameters
n_clusters_first_level = 10_000

# Second-level clustering parameters
n_clusters_second_level = 50_000

# Define the function for first-level clustering on a single file
def first_level_clustering(file):
    print(f"Starting first-level clustering for {file}")
    embeddings = np.load(file)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters_first_level, random_state=0, batch_size=BATCH_SIZE)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    np.save(file.replace(DATASET_ROOT, OUTPUT_ROOT).replace('text_emb', 'centroids'), centroids)
    print(f"Finished first-level clustering for {file}")

# Get a list of all npy files
npy_files = [os.path.join(f'{DATASET_ROOT}/text_emb/', file) for file in os.listdir(f'{DATASET_ROOT}/text_emb/') if file.endswith('.npy')]

# Perform first-level clustering in parallel
Parallel(n_jobs=N_JOBS)(delayed(first_level_clustering)(file) for file in npy_files)

# Get a list of all centroid files
centroid_files = [os.path.join(f'{OUTPUT_ROOT}/centroids/', file) for file in os.listdir(f'{OUTPUT_ROOT}/centroids/') if file.endswith('.npy')]

# Load all centroids
print('Loading all centroids')
all_centroids = np.concatenate([np.load(file) for file in centroid_files])

# Perform MiniBatchKMeans clustering on all centroids
print("Starting second-level clustering")
kmeans = MiniBatchKMeans(n_clusters=n_clusters_second_level, random_state=0, batch_size=BATCH_SIZE)
kmeans.fit(all_centroids)
print("Finished second-level clustering")

# Save the final centroids
np.save(f'{OUTPUT_ROOT}/final_centroids.npy', kmeans.cluster_centers_)

# Define the function for assigning embeddings to final clusters
def assign_to_clusters(file, final_centroids):
    print(f"Starting assignment to final clusters for {file}")
    embeddings = np.load(file)
    first_level_centroids = np.load(file.replace(DATASET_ROOT, OUTPUT_ROOT).replace('text_emb', 'centroids'))
    first_level_assignments = cdist(embeddings, first_level_centroids).argmin(axis=1)
    final_assignments = cdist(first_level_centroids, final_centroids).argmin(axis=1)
    final_cluster_assignments = final_assignments[first_level_assignments]
    np.save(file.replace(DATASET_ROOT, OUTPUT_ROOT).replace('text_emb', 'final_cluster_assignments'), final_cluster_assignments)
    print(f"Finished assignment to final clusters for {file}")

# Load the final centroids
final_centroids = np.load(f'{OUTPUT_ROOT}/final_centroids.npy')

# Assign embeddings to final clusters in parallel
Parallel(n_jobs=N_JOBS)(delayed(assign_to_clusters)(file, final_centroids) for file in npy_files)

# Define the function for selecting representative embeddings
def select_representatives(file, final_centroids):
    print(f"Starting selection of representative embeddings for {file}")
    embeddings = np.load(file)
    final_cluster_assignments = np.load(file.replace(DATASET_ROOT, OUTPUT_ROOT).replace('text_emb', 'final_cluster_assignments'))
    closest, _ = pairwise_distances_argmin_min(final_centroids, embeddings)
    representative_embeddings = embeddings[closest]
    np.save(file.replace(DATASET_ROOT, OUTPUT_ROOT).replace('text_emb', 'representative_embeddings'), representative_embeddings)
    print(f"Finished selection of representative embeddings for {file}")

# Select representative embeddings in parallel
Parallel(n_jobs=-N_JOBS)(delayed(select_representatives)(file, final_centroids) for file in npy_files)
