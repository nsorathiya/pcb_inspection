import os
import torch
import matplotlib.pyplot as plt

def visualize_activation_grid(tensor, title, max_channels=16):
    # tensor shape: (1, C, H, W)
    tensor = tensor.squeeze(0)  # remove batch dim -> (C, H, W)
    channels = tensor.shape[0]
    n_cols = 4
    n_rows = (min(channels, max_channels) + n_cols - 1) // n_cols

    plt.figure(figsize=(n_cols * 3, n_rows * 3))
    for i in range(min(channels, max_channels)):
        plt.subplot(n_rows, n_cols, i + 1)
        channel_img = tensor[i].cpu().numpy()
        plt.imshow(channel_img, cmap='viridis')
        plt.axis('off')
        plt.title(f"Channel {i}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    base_path = "neuron_data"
    for fname in sorted(os.listdir(base_path)):
        if fname.endswith(".pt"):
            layer_path = os.path.join(base_path, fname)
            tensor = torch.load(layer_path)
            visualize_activation_grid(tensor, title=fname)

if __name__ == "__main__":
    main()
