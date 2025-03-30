from flask import Flask, request, render_template, send_file
import os
import pandas as pd
from werkzeug.utils import secure_filename
import zipfile
import shutil
from Process_PDF import process_pdfs_in_zip, marcar_sospechosos_compuesto

app = Flask(__name__)

# Rutas absolutas basadas en la ubicación del archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_CSV = os.path.join(BASE_DIR, 'resultados.csv')
PDF_TEMP = os.path.join(BASE_DIR, 'pdfs_temp')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        zip_file = request.files['zipfile']
        if zip_file:
            # Guardar ZIP subido
            filename = secure_filename(zip_file.filename)
            zip_path = os.path.join(UPLOAD_FOLDER, filename)

            # Reemplazar si ya existe
            if os.path.exists(zip_path):
                os.remove(zip_path)

            zip_file.save(zip_path)

            # Procesar PDFs y calcular métricas
            df = process_pdfs_in_zip(zip_path, extract_to=PDF_TEMP)
            df = marcar_sospechosos_compuesto(df)

            # Guardar CSV
            if os.path.exists(RESULT_CSV):
                os.remove(RESULT_CSV)
            df.to_csv(RESULT_CSV, index=False)

            # Limpiar subida ZIP
            os.remove(zip_path)

            return render_template(
                'index.html',
                tables=[df.to_html(classes='table table-striped table-hover', index=False)],
                download_link='/download'
            )

    return render_template('index.html')

@app.route('/download')
def download():
    return send_file(RESULT_CSV, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
