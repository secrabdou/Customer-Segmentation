import streamlit as st
import torch
import torchvision.transforms as T
from torchvision.models import resnet34
from huggingface_hub import hf_hub_download
from PIL import Image
import requests
from io import BytesIO

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

with st.expander("ℹ️ See the 37 breeds this model can recognize"):
    st.warning("Note: This model only knows the specific breeds listed below. If you upload a breed not on this list (like a Husky or Golden Retriever), the model will try to guess the closest visual match.")
    
    # Sort and display the classes neatly
    sorted_breeds = sorted(class_names)
    formatted_list = ", ".join(sorted_breeds)
    st.write(formatted_list)

# Define image preprocessing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.write("### Test the Model")

# Sample Images Dropdown
sample_images = {
    "Select a sample image...": None,
    "Sample 1: Abyssinian Cat": "https://upload.wikimedia.org/wikipedia/commons/9/9b/Gustav_chocolate.jpg",
    "Sample 2: Beagle Dog": "https://upload.wikimedia.org/wikipedia/commons/5/55/Beagle_600.jpg"
}

selected_sample = st.selectbox("Try a ready-made photo:", list(sample_images.keys()))

st.write("--- OR ---")

# Copy-Paste friendly uploader
uploaded_file = st.file_uploader("Upload or Copy/Paste an image here...", type=["jpg", "jpeg", "png", "webp"])

image = None

# Determine which image to process (Uploaded file takes priority over sample)
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif sample_images[selected_sample] is not None:
    # Fetch the sample image from the internet
    response = requests.get(sample_images[selected_sample])
    image = Image.open(BytesIO(response.content)).convert("RGB")

# Run inference whenever an image is loaded (uploaded OR selected sample)
if image is not None:
    st.image(image, caption="Selected Image", use_column_width=True)
    
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    
    top_prob, top_catid = torch.topk(probs, 3)
    
    st.subheader("Top Predictions:")
    
    # Confidence threshold check
    CONFIDENCE_THRESHOLD = 0.40  # 40%
    highest_confidence = top_prob[0].item()
    
    if highest_confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Low confidence! This image might not contain a recognized pet, or the pet is too far away. Try cropping the photo closer to the animal's face.")
    
    # Display the results
    for i in range(top_prob.size(0)):
        st.write(f"**{class_names[top_catid[i]]}**: {top_prob[i].item()*100:.2f}%")
