Fecebook meme classification

The goal of the project is to filter out negative, offensive, or anger-inducing memes that appear on social media. The model is trained on a dataset containing 4,000 images — 2,000 labeled as positive and 2,000 as negative.

The system is built using three main components:

1.EasyOCR

2.BLIP

3.RO-BERTa

EasyOCR extracts the text from all 4,000 images. In parallel, BLIP generates a visual description for each image. After this, both the extracted text (from EasyOCR) and the generated captions (from BLIP) are passed as input to the RO-BERTa language model.

RO-BERTa then determines the category (positive or negative) and gives a confidence score. This combined EasyOCR + BLIP + RO-BERTa architecture achieves very high accuracy — up to 97–98%.

This high performance is mainly due to the fact that RO-BERTa, as a language model, brings background knowledge. It understands the meaning of the extracted text and is able to connect it with the visual description of the image.
