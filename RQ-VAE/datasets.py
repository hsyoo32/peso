import numpy as np
import torch
import torch.utils.data as data
import json
import os


class EmbDataset(data.Dataset):

    def __init__(self, data_path, dataset_name=None, block_id=0, num_blocks=5, mode="train"):
        """
        Legacy block-aware embedding dataset.
        
        Args:
            data_path: Path to the embeddings file
            dataset_name: Name of the dataset (e.g., "Instruments")
            block_id: Which block to use for training (0 for base items)
            num_blocks: Total number of blocks
            mode: "train" for one-block training, "inject" for all-item training
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.block_id = block_id
        self.num_blocks = num_blocks
        self.mode = mode
        
        # Load all embeddings
        self.embeddings = np.load(data_path)
        self.dim = self.embeddings.shape[-1]
        print(f"Loaded embeddings: {self.embeddings.shape[0]} items x {self.dim} dimensions")
        
        # Load block information only for block-aware training modes.
        if mode != "none" and dataset_name:
            self._load_block_info()
        else:
            self.block_indices = list(range(len(self.embeddings)))
    
    def _load_block_info(self):
        """Load block-based item information"""
        # Load block data to get item to block mapping
        self.block_data = {}
        for block_id in range(self.num_blocks):
            block_file = os.path.join(os.path.dirname(self.data_path), 
                                    f"{self.dataset_name}.all.block_{block_id}.json")
            if os.path.exists(block_file):
                with open(block_file, 'r') as f:
                    self.block_data[block_id] = json.load(f)
            else:
                print(f"Warning: Block file {block_file} not found")
                self.block_data[block_id] = {}
        
        # Get indices for current block
        if self.mode == "train":
            # For training, only use items from the specified block's current_block
            self.block_indices = []
            unique_items = set()
            if self.block_id in self.block_data:
                block_items = self.block_data[self.block_id]
                for user_id, data in block_items.items():
                    # Only use current_block items
                    current_items = data.get("current_block", [])
                    for item_id in current_items:
                        try:
                            embedding_idx = int(item_id)
                            if embedding_idx < len(self.embeddings):
                                #self.block_indices.append(embedding_idx)
                                unique_items.add(embedding_idx)
                            else:
                                print(f"Warning: Item ID {item_id} exceeds embedding size {len(self.embeddings)}")
                        except ValueError:
                            print(f"Warning: Could not convert item_id {item_id} to embedding index")
            
            #self.dim = len(unique_items)
            self.block_indices = list(unique_items)
            #print('block_indices: ', self.block_indices)
            print('block_indices length: ', len(self.block_indices))
            print(f"Training on block {self.block_id}, {len(unique_items)} unique items")
        
        elif self.mode == "inject":
            # Use all items while still reading block metadata.
            self.block_indices = list(range(len(self.embeddings)))
            print(f"Using all items: {len(self.block_indices)} items")
        
        #self.all_indices = list(range(len(self.embeddings)))

    def __getitem__(self, index):
        # Use block-specific index
        actual_index = self.block_indices[index]
        emb = self.embeddings[actual_index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb, actual_index

    def __len__(self):
        return len(self.block_indices)
    
    def get_all_embeddings(self):
        """Get all embeddings."""
        return self.embeddings
    
    def get_block_indices(self):
        """Get the indices of items in the current block"""
        return self.block_indices
