import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, RobertaTokenizer, RobertaForSequenceClassification
from easyocr import Reader

# Mappa, ahol a teszt képek találhatóak
input_folder = "test_pictures"

# EasyOCR inicializálása
reader = Reader(['en'], gpu=True)

# BLIP modell és processor betöltése
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Roberta modell és tokenizer betöltése
tokenizer = RobertaTokenizer.from_pretrained('./roberta_meme_modelv3-50%_6epoch')
roberta_model = RobertaForSequenceClassification.from_pretrained('./roberta_meme_modelv3-50%_6epoch')

# Feldolgozás
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        
        try:
            print(f"\nFeldolgozás: {filename}")
            
            # 1. Kép szövegének kinyerése EasyOCR-rel
            ocr_text = ""
            ocr_result = reader.readtext(image_path, detail=0)  # Csak a szövegeket kinyerjük
            if ocr_result:
                ocr_text = " ".join(ocr_result).strip()
            print(f"Kinyert szöveg (OCR): {ocr_text}")

            # 2. Kép leírásának generálása BLIP segítségével
            image = Image.open(image_path)
            inputs = blip_processor(images=image, return_tensors="pt")
            out = blip_model.generate(**inputs)
            description = blip_processor.decode(out[0], skip_special_tokens=True).strip()
            print(f"Generált leírás (BLIP): {description}")

            # 3. Roberta modellbe való betöltés és kategorizálás
            input_text = description + " " + ocr_text
            inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = roberta_model(**inputs)
                logits = outputs.logits
            
            # Softmax alkalmazása a valószínűségekre
            probabilities = torch.nn.functional.softmax(logits/2.5, dim=-1)
            negative_prob = probabilities[0][0].item()
            positive_prob = probabilities[0][1].item()
            sentiment = "Positive" if positive_prob > negative_prob else "Negative"

            # 4. Eredmények kiírása
            print(f"Eredmény (Sentiment): {sentiment}")
            print(f"Negatív valószínűség: {negative_prob:.4f}")
            print(f"Pozitív valószínűség: {positive_prob:.4f}")
            print("-" * 50)
        except Exception as e:
            print(f"Hiba történt a {filename} képnél: {e}")
