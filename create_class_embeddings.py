import torch
from transformers import CLIPModel, CLIPTokenizer
from tqdm import tqdm
from datasets.label_list import (
    imagenet_templates,
    shapenetcore_label_descriptions,
    modelnet_label_descriptions,
    scanobjectnn_label_descriptions,
)


def extract_label_features(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [
                template.format(classname) for template in templates
            ]  # format templates
            inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
            class_template_features = model.get_text_features(**inputs)  # extract features
            class_template_features = class_template_features / class_template_features.norm(
                dim=-1, keepdim=True
            )
            # class_embedding = text_embeddings.mean(dim=0)
            # class_embedding = class_embedding / class_embedding.norm()
            zeroshot_weights.append(class_template_features)
        zeroshot_weights = torch.stack(zeroshot_weights).to(device)
    return zeroshot_weights


# Load the CLIP model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model.eval()



modelnet_label_features = extract_label_features(
    modelnet_label_descriptions, imagenet_templates
)
shapenet_label_features = extract_label_features(
    shapenetcore_label_descriptions, imagenet_templates
)
scanobjectnn_label_features = extract_label_features(
    scanobjectnn_label_descriptions, imagenet_templates
)

torch.save(
    modelnet_label_features.cpu(),
    "label_features/modelnet_label_features.pt",
)
torch.save(
    shapenet_label_features.cpu(),
    "label_features/shapenetcore_label_features.pt",
)
torch.save(
    scanobjectnn_label_features.cpu(),
    "label_features/scanobject_label_features.pt",
)