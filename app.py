import streamlit as st
import torch
import torchvision.transforms as T
from torchvision.models import resnet34
from huggingface_hub import hf_hub_download
from PIL import Image

st.title("🐾 Pet Breed Classifier")
st.write("Upload a photo of a pet to identify its breed.")

@st.cache_resource
def load_model():
    # Download the weight file from your HF repository
    model_path = hf_hub_download(
        repo_id="abder-rahmen11/pet-models",
        filename="resnet34_pet_breeds.pth"
    )
    
    # Load the checkpoint dictionary
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Extract data using the keys from your saved dictionary
    state_dict = checkpoint["model_state_dict"]
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    
    # Reconstruct the model
    model = resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, class_names

model, class_names = load_model()

# Define image preprocessing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    
    top_prob, top_catid = torch.topk(probs, 3)
    
    st.subheader("Top Predictions:")
    for i in range(top_prob.size(0)):
        st.write(f"**{class_names[top_catid[i]]}**: {top_prob[i].item()*100:.2f}%")
