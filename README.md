## Leaf Classifier 🌿

Leaf Classifier is a small end-to-end machine learning project where I fine-tuned a **ResNet18** model (transfer learning in **PyTorch**) to classify leaf shapes into **smooth (lisse)** vs **toothed/serrated (dentée)**.  
I then wrapped the model inside a **Streamlit web app** so anyone can upload a leaf image and get a prediction with confidence and per-class probabilities.

---

## What this project shows

- Fine-tuning a pretrained CNN (**ResNet18**) for a custom classification task  
- Consistent preprocessing (resize + normalisation) to match training  
- Real-time inference in a simple, clean UI using **Streamlit**  
- Probabilities + confidence visualisation to make predictions interpretable  
