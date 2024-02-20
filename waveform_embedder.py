import torch
from tqdm import tqdm
from model import CombinedModel

class WaveformEmbedder:
    def __init__(self, model_path, window_size, num_channels, num_transformer_blocks, device='cpu'):
        self.model = self._load_model(model_path, window_size, num_channels, num_transformer_blocks, device)
        self.device = device
        self.hook_output = None

    def _load_model(self, model_path, window_size, patch_size, num_transformer_blocks, device):
        model = CombinedModel(window_size, patch_size, num_transformer_blocks).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        return model

    def _register_hook(self):
        def hook(module, input, output):
            self.hook_output = output.detach()
        return self.model.transformer_decoder1.register_forward_hook(hook)

    def embed(self, data):
        handle = self._register_hook()
        self.model.eval()  # Ensure the model is in evaluation mode

        with torch.no_grad():
            if not torch.is_tensor(data):
                data = torch.from_numpy(data).float()
            data = data.to(self.device)  # Add batch dimension and move to device
            self.model(data)
            embedding = self.hook_output.cpu()

        handle.remove()
        return embedding

    def extract_embedding(self, dataset, index):
        handle = self._register_hook()
        self.model.eval()  # Ensure the model is in evaluation mode

        with torch.no_grad():
            data, _ = dataset[index]  # Direct indexing from the dataset
            if not torch.is_tensor(data):
                data = torch.from_numpy(data).float()
            data = data.to(self.device)  # Add batch dimension and move to device
            self.model(data)
            embedding = self.hook_output.cpu()

        handle.remove()
        return embedding

def extract_embeddings(dataset, extractor):
    embeddings = []
    labels = []
    for i in range(len(dataset)):
        embedding = extractor.extract_embedding(dataset, i)
        _, label = dataset[i]
        embeddings.append(embedding.numpy()) 
        labels.append(label)
    labels = np.array(labels)
    return embeddings, labels
