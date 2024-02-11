from flask import Flask, render_template, request
import boto3
import fitz
import io
from PIL import Image, ImageDraw
import os
import re
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

app = Flask(__name__)

pdf_folder="PDF_data"
app.config['UPLOAD_FOLDER'] = 'PDF_data'
output_folder = "Temp"
txt_folder="txt_data"

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


#plagiarism checker
def preprocess_text(text):
    # Convert text to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def calculate_cosine_similarity(text1, text2):
    # Tokenize and preprocess the texts
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    # Create TF-IDF vectors for the documents
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    # Calculate the cosine similarity between the two vectors
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity

def check_plagiarism(file1_path, file2_path, similarity_threshold=0.7):
    # Read the content of the two files
    with open(file1_path, 'r', encoding='latin-1') as file1:
        document1 = file1.read()
    with open(file2_path, 'r', encoding='latin-1') as file2:
        document2 = file2.read()
    # Calculate the cosine similarity between the documents
    similarity = calculate_cosine_similarity(document1, document2)
    if similarity >= similarity_threshold:
        return f"\nPlagiarism detected with Similarity score: {similarity:.2f}\n"
    else:
        return f"\nNo plagiarism detected with Similarity score: {similarity:.2f}\n"

#preprocessing pdfs
def pdf_to_images(pdf_path):
    images = []
    pdf_document = fitz.open(pdf_path)
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        img_list = page.get_images(full=True)
        for img_info in img_list:
            base_image = pdf_document.extract_image(img_info[0])
            image_data = base_image["image"]
            image = Image.open(io.BytesIO(image_data))
            images.append(image)
    pdf_document.close()
    return images

def convert_to_grayscale(image):
    return image.convert("L")

def main(pdf_path, output_folder):
    images = pdf_to_images(pdf_path)
    for i, image in enumerate(images):
        grayscale_image = convert_to_grayscale(image)
        output_path = f"{output_folder}/image_{i + 1}.jpg"
        grayscale_image.save(output_path)
    return images

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']

        if file1 and file2:
            f1_path=os.path.join(app.config['UPLOAD_FOLDER'], 'file1.pdf')
            f2_path=os.path.join(app.config['UPLOAD_FOLDER'], 'file2.pdf')
            file1.save(f1_path)
            file2.save(f2_path)

            # text1 = extract_text_from_pdf(f1_path,"file1")
            # text2 = extract_text_from_pdf(f2_path,"file2")

            # testing
            file_path = os.path.join(txt_folder, "file1.txt")
            with open(file_path, 'r') as file:
                # Read all lines from the file and store them in a list
                text1= file.readlines()
            file_path = os.path.join(txt_folder, "file2.txt")
            with open(file_path, 'r') as file:
                # Read all lines from the file and store them in a list
                text2= file.readlines()

            fname_txt1=os.path.join(txt_folder,"file1.txt")
            fname_txt2=os.path.join(txt_folder,"file2.txt")
            result = check_plagiarism(fname_txt1, fname_txt2)
            
            # remove uploaded PDFs, not necessary

            # files = os.listdir(pdf_folder)
            # for file in files:
            #     file_path = os.path.join(pdf_folder , file)
            #     # Check if the path is a file (not a directory)
            #     if os.path.isfile(file_path):
            #         # Delete the file
            #         os.remove(file_path)

            #supposed to be text1 and text2, not None
            return render_template('index.html', text1=text1, text2=text2,result=result) #text1, text2

    return render_template('index.html', text1="", text2="",result="")

if __name__ == '__main__':
    app.run(debug=True)
