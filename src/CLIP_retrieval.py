import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import cv2
import torch.nn.functional as F
import textwrap

class CLIPRetrieval:
    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model.to(config['device'])
        self.dataset = dataset
        self.device = config['device']
        self.model.eval()
        self.output_dir = config['output_dir']

        self.build_dictionnaries()              ## projected_image_embeddings, projected_text_embeddings, labels, image_paths
        self.compute_baseline_statistics()
        
    def build_dictionnaries(self):
        image_embeddings, text_embeddings, labels, image_paths = [], [], [], []
        
        with torch.no_grad():
            for image, text, path, label in self.dataset:
                # Project image embeddings to shared space
                image_embedding = image.unsqueeze(0).to(self.device)  # Add batch dimension
                image_projected = self.model.image_projection(image_embedding).squeeze(0)
                
                # Project text embeddings to shared space
                text_embedding = text.unsqueeze(0).to(self.device)  # Add batch dimension
                text_projected = self.model.text_projection(text_embedding).squeeze(0)
                
                image_embeddings.append(image_projected)
                text_embeddings.append(text_projected)
                image_paths.append(path)
                labels.append(label)
        
        self.image_embeddings = torch.stack(image_embeddings).to(self.device)
        self.text_embeddings = torch.stack(text_embeddings).to(self.device)
        self.labels = labels
        self.image_paths = image_paths
    
    def compute_baseline_statistics(self):
        print("Computing baseline similarity statistics...")
        
        # Sample size for efficiency
        n_samples = min(1000, len(self.text_embeddings))            ## Len of text_embeddings and image_embeddings = 4045
        
        # Compute text-to-text similarities
        text_indices = torch.randperm(len(self.text_embeddings))[:n_samples]
        text_samples = self.text_embeddings[text_indices]
        text_similarities = torch.matmul(text_samples, text_samples.T)
        
        # Compute image-to-image similarities
        image_indices = torch.randperm(len(self.image_embeddings))[:n_samples]
        image_samples = self.image_embeddings[image_indices]
        image_similarities = torch.matmul(image_samples, image_samples.T)
        
        # Store separate statistics for text and image
        self.text_stats = self.compute_stats(text_similarities)
        self.image_stats = self.compute_stats(image_similarities)
        
    def compute_stats(self, similarities):
        return {
            'mean': similarities.mean().item(),
            'std': similarities.std().item(),
            'min': similarities.min().item(),
            'max': similarities.max().item(),
            'percentiles': {
                '25': similarities.quantile(0.25).item(),
                '50': similarities.quantile(0.50).item(),
                '75': similarities.quantile(0.75).item(),
                '90': similarities.quantile(0.90).item(),
                '95': similarities.quantile(0.95).item(),
            }
        }


    def save_similarity_matrix(self, sample_size=10):
        """
        Create a visualization of cosine similarities between image and text embeddings,
        handling different embedding dimensions
        """
        # Sample a subset of embeddings for visualization
        indices = torch.randperm(len(self.image_embeddings))[:sample_size]
        
        # Get the sampled embeddings
        image_features = self.image_embeddings[indices]         ## (batch_size, 256)
        text_features = self.text_embeddings[indices]           ## (batch_size, 256)
        sampled_labels = [self.labels[i] for i in indices]
        sampled_paths = [self.image_paths[i] for i in indices]
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Project features to same dimension if needed
        if image_features.shape[1] != text_features.shape[1]:
            print("Features have different dimensions. Using projection...")
            # Option 1: Project to smaller dimension
            min_dim = min(image_features.shape[1], text_features.shape[1])
            if image_features.shape[1] > min_dim:
                projection_matrix = torch.randn(image_features.shape[1], min_dim).to(self.device)
                projection_matrix = projection_matrix / projection_matrix.norm(dim=0, keepdim=True)
                image_features = torch.matmul(image_features, projection_matrix)
            if text_features.shape[1] > min_dim:
                projection_matrix = torch.randn(text_features.shape[1], min_dim).to(self.device)
                projection_matrix = projection_matrix / projection_matrix.norm(dim=0, keepdim=True)
                text_features = torch.matmul(text_features, projection_matrix)
        
        # Calculate similarity matrix
        similarity = torch.matmul(text_features, image_features.T).cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(20, 14))
        
        # Plot similarity matrix
        im = plt.imshow(similarity, vmin=similarity.min(), vmax=similarity.max(), cmap='viridis')
        plt.colorbar(im, label='Cosine Similarity')
        
        # Add text labels
        for i in range(similarity.shape[0]):
            for j in range(similarity.shape[1]):
                plt.text(j, i, f"{similarity[i, j]:.2f}", 
                        ha="center", va="center", 
                        color='white' if similarity[i, j] < similarity.mean() else 'black',
                        size=10)
        
        # Add thumbnail images on x-axis
        for i, path in enumerate(sampled_paths):
            img = self.load_and_resize_image(path, target_size=(64, 64))
            plt.imshow(img, extent=(i - 0.5, i + 0.5, similarity.shape[0], similarity.shape[0] + 1), 
                    aspect='auto')
        
        # Customize axes
        plt.yticks(range(len(sampled_labels)), sampled_labels, fontsize=12)
        plt.xticks([])
        
        # Remove spines
        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)
        
        # Set plot limits and title
        plt.xlim([-0.5, sample_size - 0.5])
        plt.ylim([sample_size + 0.5, -1])
        plt.title("Cosine Similarity between Text and Image Features", size=20, pad=40)
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.output_dir, f'similarity_matrix_{timestamp}.png')
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()


    def find_similar(self, query, embeddings, modality, k=5):
        with torch.no_grad():
            
            modality_embeddings = embeddings.to(self.device)
            
            # Compute similarities with all images/texts
            similarities = torch.matmul(query, modality_embeddings.T).squeeze(0)
            
            # Get top k matches
            top_k_similarities, top_k_indices = torch.topk(similarities, k)
            
            # Add evaluation for each similarity score
            normalized_scores = [self.normalize_similarity(sim.item(), modality) for sim in top_k_similarities]
            evaluations = [self.evaluate_similarity(sim.item(), modality) for sim in top_k_similarities]
            
            return {
                'indices': top_k_indices.cpu().numpy(),
                'similarities': top_k_similarities.cpu().numpy(),
                'normalized_scores': normalized_scores,
                'evaluations': evaluations,
                'labels': [self.labels[idx] for idx in top_k_indices],
                'paths': [self.image_paths[idx] for idx in top_k_indices]
            }
        
    def normalize_similarity(self, similarity, modality='text'):
        stats = self.text_stats if modality == 'text' else self.image_stats
        normalized = (similarity - stats['min']) / (stats['max'] - stats['min'])
        # print(f"Highest similarity: {stats['max']}, Lowest similarity: {stats['min']}")
        return normalized * 100
    
    def evaluate_similarity(self, similarity, modality='text'):
        stats = self.text_stats if modality == 'text' else self.image_stats
        
        if similarity >= stats['percentiles']['95']:
            return "Excellent match (top 5%)"
        elif similarity >= stats['percentiles']['90']:
            return "Very good match (top 10%)"
        elif similarity >= stats['percentiles']['75']:
            return "Good match (top 25%)"
        elif similarity >= stats['percentiles']['50']:
            return "Moderate match"
        else:
            return "Weak match"


    def load_and_resize_image(self, image_path, target_size=(224, 224)):
        """Load image and resize it for visualization."""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        return img

    def create_retrieval_plot(self, query_image_path, query_label, similar_results, query_type):
        """Create and save a plot with query image and retrieved results."""
        k = len(similar_results['indices'])
        
        # Create a single row plot for query and similar images with increased height
        fig, axes = plt.subplots(1, k + 1, figsize=(4 * (k + 1), 7))  # Increased height to 5
        fig.suptitle(f'{query_type} Retrieval Results', fontsize=16, y=0.95)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Plot query image
        query_img = self.load_and_resize_image(query_image_path)
        axes[0].imshow(query_img)
        axes[0].axis('off')
        axes[0].set_title('Query Image', fontsize=12)
        wrapped_query_label = textwrap.fill(" ".join(query_label) if isinstance(query_label, list) else query_label, width=50)
        axes[0].text(0.5, -0.15, wrapped_query_label, ha='center', va='top', transform=axes[0].transAxes, fontsize=10)

        # Plot retrieved images
        for i, (idx, sim, norm_score, label) in enumerate(zip(
            similar_results['indices'],
            similar_results['similarities'],
            similar_results['normalized_scores'],
            similar_results['labels']
        )):
            retrieved_img = self.load_and_resize_image(similar_results['paths'][i])
            axes[i + 1].imshow(retrieved_img)
            axes[i + 1].axis('off')
            
            # Set similarity score above the image
            axes[i + 1].set_title(f'Similarity: {norm_score:.2f}%', fontsize=14)
            
            # Wrap the label text to ensure it doesn't exceed image width
            wrapped_label = textwrap.fill(" ".join(label) if isinstance(label, list) else label, width=50)
            axes[i + 1].text(0.5, -0.1, wrapped_label, ha='center', va='top', transform=axes[i + 1].transAxes, fontsize=10)

        # Adjust layout to give more space for titles and labels
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, wspace=0.1)
        filename = f'retrieval_{query_type}_{timestamp}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        
        return filepath

     
    def retrieve_similar_content(self, k=5):
        image_tensor, text_tensor, sample_path, sample_label = self.dataset[100]
        # self.save_similarity_matrix(sample_size=15)

        # print("\nImage-to-Image Baseline Statistics:")
        # print(f"Average similarity: {self.image_stats['mean']:.3f}")
        # print(f"90th percentile: {self.image_stats['percentiles']['90']:.3f}")
        # print(f"95th percentile: {self.image_stats['percentiles']['95']:.3f}")
        
        # print("\nText-to-Text Baseline Statistics:")
        # print(f"Average similarity: {self.text_stats['mean']:.3f}")
        # print(f"90th percentile: {self.text_stats['percentiles']['90']:.3f}")
        # print(f"95th percentile: {self.text_stats['percentiles']['95']:.3f}\n")

        print("\n-----------IMAGE-TO-TEXT RETRIEVAL-----------")
        query_embedding = image_tensor.to(self.device)
        query_embedding = self.model.image_projection(query_embedding).squeeze(0)
        similar_images = self.find_similar(query_embedding, self.image_embeddings, modality='image', k=k)
        
        print(f"Original image label is '{sample_label}'")
        print("Top similar items are:")
        for i, (idx, sim, norm_score, eval_result, path, label) in enumerate(zip(
            similar_images['indices'], 
            similar_images['similarities'], 
            similar_images['normalized_scores'],
            similar_images['evaluations'],
            similar_images['paths'], 
            similar_images['labels']
        )):
            print(f"{i+1}. Image {idx} with normalized sim {norm_score:.2f}% - {eval_result}.\n   Label: {label}")

        # Create and save image-to-image plot
        img2img_plot = self.create_retrieval_plot(sample_path, sample_label, similar_images, 'Image2Text')
        print(f"Image-to-Text retrieval plot saved to: {img2img_plot}")


        print("\n-----------TEXT-TO-IMAGE RETRIEVAL-----------")
        # text = "a group of people dancing in a party"
        # text_tensor = self.dataset.text_encoder(text).unsqueeze(0)
        query_embedding = text_tensor.to(self.device)
        query_embedding = self.model.text_projection(query_embedding).squeeze(0)
        similar_texts = self.find_similar(query_embedding, self.text_embeddings, modality='text', k=k)
        
        print(f"Original image label is '{sample_label}'")
        print("Top similar items are:")
        for i, (idx, sim, norm_score, eval_result, path, label) in enumerate(zip(
            similar_texts['indices'], 
            similar_texts['similarities'], 
            similar_texts['normalized_scores'],
            similar_texts['evaluations'],
            similar_texts['paths'], 
            similar_texts['labels']
        )):
            print(f"{i+1}. Image {idx} with normalized sim {norm_score:.2f}% - {eval_result}.\n   Label: {label}")

        # Create and save image-to-image plot
        text2img_plot = self.create_retrieval_plot(sample_path, sample_label, similar_texts, 'Text2Image')
        print(f"Text-to-Image retrieval plot saved to: {text2img_plot}")

        return { 'similar_images': similar_images, 'matching_text': similar_texts}