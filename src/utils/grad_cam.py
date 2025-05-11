
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch

class GradCAMpp3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.clone().detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].clone().detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()

        input_tensor = input_tensor.unsqueeze(0)  # (1, C, D, H, W)
        input_tensor.requires_grad = True

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        loss = output[:, target_class]
        self.model.zero_grad()
        loss.backward()

        gradients = self.gradients  # (B, C, D, H, W)
        activations = self.activations  # (B, C, D, H, W)

        grad_squared = gradients ** 2
        grad_cubed = gradients ** 3

        # Compute weights
        alpha = grad_squared / (2 * grad_squared + activations * grad_cubed + 1e-7)
        weights = (alpha * F.relu(gradients)).sum(dim=[2,3,4], keepdim=True)

        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1).squeeze(0)
        cam = F.relu(cam)

        # Normalize
        cam -= cam.min()
        cam /= cam.max()

        return cam.detach().cpu()
    
class EigenGradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, _inp, output):
        # output shape: (B, C, D, H, W)
        self.activations = output.detach()

    def generate(self, x):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x.unsqueeze(0))

        # (C, D, H, W)
        acts = self.activations.squeeze(0)
        c, d, h, w = acts.shape
        flat = acts.view(c, -1)                         # (C, D*H*W)

        # covariance & eigen‑decomp (torch >=1.8 uses torch.linalg)
        cov = flat @ flat.t() / flat.size(1)            # (C, C)
        eigvals, eigvecs = torch.linalg.eigh(cov)       # symmetric → eigh
        principal = eigvecs[:, -1]                      # largest eigen‑vector

        cam = (principal.unsqueeze(1) * flat).sum(0).view(d, h, w)
        cam = F.relu(cam)
        cam -= cam.min(); cam /= cam.max() + 1e-8
        return cam.cpu()

def show_multi_slice_cam(scan, cam, slice_idxs=None, projection='sagittal', save_path=None):
    scan = scan.squeeze(0)

    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=scan.shape,
        mode='trilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)

    if slice_idxs is None:
        center = scan.shape[0] // 2
        slice_idxs = [center-20, center-10, center, center+10, center+20]

    fig, axes = plt.subplots(2, len(slice_idxs), figsize=(4*len(slice_idxs), 8))

    for i, idx in enumerate(slice_idxs):
        if projection == 'sagittal':
            img_slice = scan[idx, :, :]
            cam_slice = cam[idx, :, :]
        elif projection == 'coronal':
            img_slice = scan[:, idx, :]
            cam_slice = cam[:, idx, :]
        elif projection == 'axial':
            img_slice = scan[:, :, idx]
            cam_slice = cam[:, :, idx]
        else:
            raise ValueError(f"Unknown projection: {projection}")

        axes[0,i].imshow(img_slice.cpu(), cmap='gray')
        axes[0,i].set_title(f'Original Slice {idx}')
        axes[0,i].axis('off')

        axes[1,i].imshow(img_slice.cpu(), cmap='gray')
        axes[1,i].imshow(cam_slice.cpu(), cmap='jet', alpha=0.5)
        axes[1,i].set_title(f'GradCAM Slice {idx}')
        axes[1,i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"GradCAM saved to {save_path}")
    plt.close()       
     