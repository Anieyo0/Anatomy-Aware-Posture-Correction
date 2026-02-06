import matplotlib.pyplot as plt
import os

def plot_training_logs(
    epoch_total_losses,
    epoch_primary_losses,
    epoch_smooth_losses,
    epoch_mse_component_losses,
    epoch_anatomy_component_losses,
    epoch_edge_plain_component_losses,
    epoch_edge_anatomy_component_losses,
    save_path=None
):
    plt.figure(figsize=(12, 6))

    # subplot 1: 총합 loss
    plt.subplot(1, 2, 1)
    plt.plot(epoch_total_losses, label='Total Loss')
    plt.plot(epoch_primary_losses, label='Total Primary Loss')
    plt.plot(epoch_smooth_losses, label='Total Smoothness Loss')
    plt.title('Total Loss, Primary, and Grad Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Sum')
    plt.legend()
    plt.grid(True)

    # subplot 2: weighted components per epoch
    plt.subplot(1, 2, 2)
    plt.plot(epoch_mse_component_losses, label='Global Intensity MSE Loss')
    plt.plot(epoch_anatomy_component_losses, label='Anatomical Intensity MSE Loss')
    plt.plot(epoch_edge_plain_component_losses, label='Global Edge Loss')
    plt.plot(epoch_edge_anatomy_component_losses, label='Anatomical Edge Loss')
    plt.title('Weighted Loss Components per Epoch (Sum)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Sum')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")

    plt.show()
