import tkinter as tk
from tkinter import filedialog, Label, Button, Text, Scrollbar, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import google.generativeai as genai

# ---------------------------
# 🔑 Gemini API Setup
# ---------------------------
genai.configure(api_key="AIzaSyDyS2vVU9R3n3qx8_g6IX_wgqkQfYoSyLg")

# ---------------------------
# Load trained model
# ---------------------------
model = tf.keras.models.load_model("plant_disease_mobilenetv2.keras")

# ---------------------------
# Class names
# ---------------------------
class_names = [
    'Apple__alternaria_leaf_spot', 'Apple_black_rot', 'Apple_brown_spot', 'Apple__gray_spot',
    'Apple__healthy', 'Apple_rust', 'Apple__scab',
    'Bell_pepper__bacterial_spot', 'Bell_pepper__healthy',
    'Blueberry___healthy',
    'Cassava__bacterial_blight', 'Cassava_brown_streak_disease', 'Cassava__green_mottle',
    'Cassava__healthy', 'Cassava__mosaic_disease',
    'Cherry__healthy', 'Cherry__powdery_mildew',
    'Coffee__healthy', 'Coffee_red_spider_mite', 'Coffee__rust',
    'Corn__common_rust', 'Corn_gray_leaf_spot', 'Corn_healthy', 'Corn__northern_leaf_blight',
    'Grape__black_measles', 'Grape_black_rot', 'Grape_healthy', 'Grape__Leaf_blight',
    'Orange___citrus_greening',
    'Peach__bacterial_spot', 'Peach__healthy',
    'Potato__bacterial_wilt', 'Potato_early_blight', 'Potato_healthy', 'Potato__late_blight',
    'Potato__leafroll_virus', 'Potato_mosaic_virus', 'Potato_nematode', 'Potato__pests',
    'Potato___phytophthora',
    'Raspberry___healthy',
    'Rice__bacterial_blight', 'Rice_blast', 'Rice_brown_spot', 'Rice__tungro',
    'Rose__healthy', 'Rose_rust', 'Rose__slug_sawfly',
    'Soybean___healthy',
    'Squash___powdery_mildew',
    'Strawberry__healthy', 'Strawberry__leaf_scorch',
    'Sugarcane__healthy', 'Sugarcane_mosaic', 'Sugarcane_red_rot', 'Sugarcane__rust',
    'Sugarcane___yellow_leaf',
    'Tomato__bacterial_spot', 'Tomato_early_blight', 'Tomato_healthy', 'Tomato__late_blight',
    'Tomato__leaf_curl', 'Tomato_leaf_mold', 'Tomato_mosaic_virus', 'Tomato__septoria_leaf_spot',
    'Tomato__spider_mites', 'Tomato__target_spot',
    'Watermelon__anthracnose','Watermelondowny_mildew','Watermelonhealthy','Watermelon_mosaic_virus'
]

# ---------------------------
# 💬 Gemini Fertilizer Suggestion
# ---------------------------
def get_fertilizer_suggestion(disease_name):
    try:
        prompt = (
            f"You are an agriculture expert. Suggest fertilizers and organic treatments for {disease_name} in plants. "
            "Include 3-5 practical recommendations farmers can follow. give only the names of fertilzers and its reasion in just one line, Keep it clean, no ** symbols"
        )
        model_gemini = genai.GenerativeModel("gemini-2.0-flash")

        print(f"[DEBUG] Sending prompt to Gemini: {prompt}")  # Debug info
        response = model_gemini.generate_content(prompt)
        print("[DEBUG] Gemini API response received.")

        if hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return "⚠ No suggestion text returned from Gemini."
    except Exception as e:
        print(f"[ERROR] Gemini API failed: {e}")
        return f"⚠ Error fetching suggestion: {str(e)}"

# ---------------------------
# 🌿 GUI Setup
# ---------------------------
root = tk.Tk()
root.title("🌿 Plant Disease Detection & Fertilizer Suggestion")
root.geometry("720x850")
root.configure(bg="#e8f5e9")

Label(root, text="🌱 Plant Disease Detection System", font=("Arial", 22, "bold"), bg="#a5d6a7").pack(pady=20)

img_label = Label(root, bg="#e8f5e9")
img_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 16), bg="#e8f5e9", fg="darkgreen")
result_label.pack(pady=10)

Label(root, text="💡 Fertilizer & Treatment Suggestions", font=("Arial", 16, "bold"), bg="#e8f5e9", fg="brown").pack(pady=10)

fertilizer_text = Text(root, height=12, width=80, wrap="word", font=("Arial", 12))
fertilizer_text.pack(pady=10)

scrollbar = Scrollbar(root, command=fertilizer_text.yview)
fertilizer_text.config(yscrollcommand=scrollbar.set)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

img_path = None

# ---------------------------
# 📸 Image Handling
# ---------------------------
def open_image():
    global img_path
    img_path = filedialog.askopenfilename(
        title="Select a Leaf Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if img_path:
        img = Image.open(img_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk
        result_label.config(text="")
        fertilizer_text.delete("1.0", tk.END)

# ---------------------------
# 🔍 Predict Disease
# ---------------------------
def predict_disease():
    if not img_path:
        messagebox.showwarning("No Image", "Please select an image first.")
        return

    try:
        img = Image.open(img_path).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        pred_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        disease_name = class_names[pred_class]
        result_label.config(
            text=f"Prediction: {disease_name}\nConfidence: {confidence*100:.2f}%",
            fg="blue"
        )

        fertilizer_text.delete("1.0", tk.END)
        fertilizer_text.insert(tk.END, "Fetching fertilizer suggestions...\n")
        root.update_idletasks()

        suggestion = get_fertilizer_suggestion(disease_name)
        fertilizer_text.delete("1.0", tk.END)
        fertilizer_text.insert(tk.END, suggestion)

    except Exception as e:
        messagebox.showerror("Error", str(e))
        print(f"[ERROR] Prediction failed: {e}")

# ---------------------------
# 🧭 Buttons
# ---------------------------
Button(root, text="📂 Select Image", command=open_image, font=("Arial", 14), bg="#81c784", width=20).pack(pady=10)
Button(root, text="🔍 Predict Disease", command=predict_disease, font=("Arial", 14), bg="#66bb6a", width=20).pack(pady=10)

root.mainloop()
