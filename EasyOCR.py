import os
import csv
from easyocr import Reader

# EasyOCR inicializálása
# A nyelvi kódokat itt adhatod meg, például ['en'] az angolhoz, vagy ['hu'] a magyarhoz
reader = Reader(['en'], gpu=True)  # Állítsd True-ra a GPU gyorsításért, ha elérhető

# Mappa elérési útja
input_folder = os.path.normpath("negative")

output_csv = "image_descriptions-negativeEasyOCR.csv"

# Ellenőrizzük, hogy a CSV fájl létezik-e
if not os.path.exists(output_csv):
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["", "", "File Name", "Extracted Text"])  # Fejléc létrehozása üres oszlopokkal

# Fájlok feldolgozása és szövegek kiírása CSV-be
with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # A mappa összes képfájl feldolgozása
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        
        # Csak képfájlokat dolgoz fel (pl. .jpg, .png)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            try:
                # Ellenőrzés, hogy a fájl megnyitható-e
                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"Nem található a fájl: {file_path}")
                
                # Szöveg kinyerése EasyOCR-rel
                result = reader.readtext(file_path, detail=0)  # detail=0 csak a szövegeket adja vissza
                if not result:
                    raise ValueError(f"A fájlból nem nyerhető ki szöveg: {file_name}")

                text = " ".join(result)  # Több szövegrészlet egyesítése
                
                # Kép nevét és kinyert szövegét CSV-be mentjük
                writer.writerow(["", "", file_name, text.strip()])
                print(f"Kép: {file_name}")
                print("Kinyert szöveg:")
                print(text.strip())
                print("-" * 40)  # Elválasztó vonal a következő képhez
            except Exception as e:
                print(f"Hiba történt a(z) {file_name} fájl feldolgozásakor: {e}")