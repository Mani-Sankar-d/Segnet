import matplotlib.pyplot as plt


def visualize_prediction(image, pred_mask, label_mask=None):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(image.permute(1,2,0).cpu()) 
    plt.title("Input Image")
    
    plt.subplot(1,3,2)
    plt.imshow(pred_mask, cmap='nipy_spectral')
    plt.title("Predicted Mask")
    
    if label_mask is not None:
        plt.subplot(1,3,3)
        plt.imshow(label_mask, cmap='nipy_spectral')
        plt.title("Ground Truth Mask")
    plt.show()