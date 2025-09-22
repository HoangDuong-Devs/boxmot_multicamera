from boxmot.appearance.backends.base_backend import BaseModelBackend
from boxmot.appearance.reid.registry import ReIDModelRegistry


class PyTorchBackend(BaseModelBackend):

    def __init__(self, weights, device, half):
        super().__init__(weights, device, half)
        self.nhwc = False
        self.half = half

    def load_model(self, w):
        # Load a PyTorch model
        if w and w.is_file():
            ReIDModelRegistry.load_pretrained_weights(self.model, w)
        self.model.to(self.device).eval()
        self.model.half() if self.half else self.model.float()
        
        # Get feature dimension based on model type
        if hasattr(self.model, "model_name") and "ViT" in self.model.model_name:
            # CLIP ViT model: in_planes (768) + in_planes_proj (512)
            self.feature_dim = self.model.in_planes + self.model.in_planes_proj
        elif hasattr(self.model, "model_name") and "RN50" in self.model.model_name: 
            # CLIP ResNet model: in_planes (2048) + in_planes_proj (1024)
            self.feature_dim = self.model.in_planes + self.model.in_planes_proj
        else:
            # Default OSNet and other models
            self.feature_dim = getattr(self.model, "feature_dim", 512)

    def forward(self, im_batch):
        features = self.model(im_batch)
        return features
