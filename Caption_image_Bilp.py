
import os
import csv
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# A mappák nevei
folders = ["D:/Személyes dolgok/MESTERI/Moul II/Sematica si pragmatica limbajului natural/Prompt/negative"]

# BLIP modell és processor betöltése
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# CSV fájlba történő mentés
output_file = "image_descriptions-negativeBlip.csv"

# Nyisd meg a CSV fájlt írásra, ha nem létezik, akkor létrehozza
with open(output_file, mode="w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # Fejléc a CSV fájlhoz
    writer.writerow(["Image Name", "Description"])

    # Képeket keresünk a mappákban
    for folder in folders:
        folder_path = folder
        # Ellenőrizzük, hogy a mappa létezik
        if not os.path.isdir(folder_path):
            print(f"A '{folder}' mappa nem található.")
            continue  # Ha a mappa nem létezik, lépünk tovább
        
        # Végigmegyünk a mappa fájljain
        for filename in os.listdir(folder_path):
            # Csak a megfelelő formátumú fájlokat dolgozzuk fel
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, filename)
                
                try:
                    # Kép betöltése
                    image = Image.open(image_path)

                    # Kép előkészítése a modell számára
                    inputs = processor(images=image, return_tensors="pt")

                    # Leírás generálása
                    out = model.generate(**inputs)
                    caption = processor.decode(out[0], skip_special_tokens=True)

                    # Kép neve és a generált leírás mentése a CSV fájlba
                    writer.writerow([filename, caption])
                    print(f"Leírás mentve: {filename}")
                except Exception as e:
                    print(f"Hiba történt a {filename} képnél: {e}")
                    
print(f"A képek leírásai mentve lettek a '{output_file}' fájlba.")
