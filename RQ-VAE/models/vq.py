import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm
import random
import wandb


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim, mu = 0.25,
                 beta = 1, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=0.01, sk_iters=100):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.mu = mu
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        # centers = kmeans(
        #     data,
        #     self.n_e,
        #     self.kmeans_iters,
        # )
        centers, _ = self.constrained_km(data, 256)
        self.embedding.weight.data.copy_(centers)
        self.initted = True
    
    def constrained_km(self, data, n_clusters=10):
        from k_means_constrained import KMeansConstrained 
        x = data.cpu().detach().numpy()
        
        # Ensure n_clusters is not larger than data size
        n_clusters = min(n_clusters, len(x) - 1)  # Subtract 1 to ensure we have enough points for initialization
        
        # # Calculate size constraints based on dataset size
        # if len(x) > 10000:
        #     # For large datasets, use more flexible size constraints
        #     size_min = max(2, len(x) // (n_clusters * 4))  # More clusters, smaller min size
        #     size_max = len(x) // n_clusters + 100  # Allow larger clusters
        # else:
        # Original logic for smaller datasets
        size_min = min(len(data) // (n_clusters * 2), 50) # 50 for the very first time, 10 the latter
        size_min = max(2, size_min)  # Ensure size_min is at least 2
        size_max = size_min * 4

        # Ensure the constraint is satisfied
        if size_max * n_clusters < len(x):
            # If constraint is violated, increase size_max
            size_max = (len(x) // n_clusters) + 100
        
        print(f"Clustering {len(x)} items into {n_clusters} clusters with size_min={size_min}, size_max={size_max}")
        print(f"Constraint check: {size_max} * {n_clusters} = {size_max * n_clusters} >= {len(x)} = {size_max * n_clusters >= len(x)}")

        clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_max, max_iter=10, n_init=10,
                                n_jobs=10, verbose=False)
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()
        value_counts = {}
        return t_centers, t_labels


    def diversity_loss(self, x_q, indices, indices_cluster, indices_list):
        emb = self.embedding.weight
        temp = 1

        pos_list = [indices_list[i] for i in indices_cluster]
        pos_sample = []
        for idx, pos in enumerate(pos_list):
            random_element = random.choice(pos)

            while random_element == indices[idx]:
                random_element = random.choice(pos)
            pos_sample.append(random_element)

        y_true = torch.tensor(pos_sample, device=x_q.device)

        # sim = F.cosine_similarity(x_q, emb, dim=-1)
        sim = torch.matmul(x_q, emb.t())

        # sampled_ids = torch.multinomial(best_scores, num_samples=1)
        sim_self = torch.zeros_like(sim)
        for idx, row in enumerate(sim_self):
            sim_self[idx, indices[idx]] = 1e12
        sim = sim - sim_self
        sim = sim / temp
        loss = F.cross_entropy(sim, y_true)

        return loss

    def diversity_loss_main_entry(self, x, x_q, indices, labels):

        indices_cluster = [labels[idx.item()] for idx in indices]
        target_numbers = list(range(10)) 
        indices_list = {}
        for target_number in target_numbers:
            indices_list[target_number] = [index for index, num in enumerate(labels) if num == target_number]

        diversity_loss = self.diversity_loss(x_q, indices, indices_cluster, indices_list)

        return diversity_loss
                    
    
    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances
    
    def vq_init(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim)

        if not self.initted:
            self.init_emb(latent)

        _distance_flag = 'distance'    
        
        if _distance_flag == 'distance':
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, self.embedding.weight.t())
        else:    
        # Calculate Cosine Similarity 
            d = latent@self.embedding.weight.t()


        if not use_sk or self.sk_epsilon <= 0:
            if _distance_flag == 'distance':
                indices = torch.argmin(d, dim=-1)
            else:    
                indices = torch.argmax(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()

            Q = sinkhorn_algorithm(d,self.sk_epsilon,self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
                indices = torch.argmin(d, dim=-1)
            else:
                # Use Sinkhorn assignment matrix to get deterministic but different assignments
                sorted_indices = torch.argsort(Q, dim=-1, descending=True)
                
                # First, get the default assignments (best for each item)
                indices = sorted_indices[:, 0].clone()
                
                # Check if there are actual collisions in the assignments
                unique_assignments = set(indices.tolist())
                if len(unique_assignments) < len(indices):  # There are collisions
                    # For items that would get the same assignment, find the best available centroid
                    for i in range(1, len(indices)):
                        # Check if this item collides with any previous item
                        while indices[i] in indices[:i]:  # Keep trying until we find a unique assignment
                            # Find the next best available centroid for this item
                            for j in range(1, sorted_indices.shape[1]):  # Try second, third, fourth best, etc.
                                candidate = sorted_indices[i, j]
                                if candidate not in indices[:i]:  # This centroid is available
                                    indices[i] = candidate
                                    break
                            else:
                                # If we've tried all centroids and still have collision, 
                                # just use the next available one (this shouldn't happen in practice)
                                break
                else:
                    pass  # No collisions, keep original assignments

        x_q = self.embedding(indices).view(x.shape)

        return x_q
    
    def forward(self, x, label, idx, use_sk=True):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        _distance_flag = 'distance'    
        
        if _distance_flag == 'distance':
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, self.embedding.weight.t())
        else:    
        # Calculate Cosine Similarity 
            d = latent@self.embedding.weight.t()
        if not use_sk or self.sk_epsilon <= 0:
            if _distance_flag == 'distance':
                if idx != -1:
                    indices = torch.argmin(d, dim=-1)
                else:
                    temp = 1.0
                    prob_dist = F.softmax(-d/temp, dim=1)  
                    indices = torch.multinomial(prob_dist, 1).squeeze()
            else:    
                indices = torch.argmax(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()

            Q = sinkhorn_algorithm(d,self.sk_epsilon,self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")

            indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # Diversity
        if self.training and self.beta > 0:
            diversity_loss = self.diversity_loss_main_entry(x, x_q, indices, label)
        else:
            diversity_loss = 0

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())

        loss = codebook_loss + self.mu * commitment_loss + self.beta * diversity_loss


        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices

