import os
import sys
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

# Proje kök dizinini Python yoluna ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from mip_setup import read_inputs, InputsSetup
    from mip_solve import mathematical_model_solve
except ImportError as e:
    print(f"Import hatası: {e}")
    raise

app = Flask(__name__)
app.secret_key = 'gizli_anahtar'

# Config ayarları
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Klasörleri oluştur
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Dosya seçilmedi', 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('Dosya seçilmedi', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Önceki dosyaları temizle
            for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f'Dosya silinemedi: {file_path}. Hata: {e}')

            file.save(filepath)

            try:
                inputs_dict = read_inputs(filepath)
                mip_inputs = InputsSetup(inputs_dict)
                mathematical_model_solve(mip_inputs)
                flash('Planlama başarıyla oluşturuldu!', 'success')
                return redirect(url_for('results'))
            except Exception as e:
                flash(f'Hata oluştu: {str(e)}', 'error')
                return redirect(request.url)

    return render_template('upload.html')


@app.route('/results')
def results():
    output_file = os.path.join(app.config['OUTPUT_FOLDER'], 'results.xlsx')

    if not os.path.exists(output_file):
        flash('Önce bir planlama oluşturun', 'error')
        return redirect(url_for('upload_file'))

    try:
        # Basit bir özet oluştur
        summary = {
            'status': 'success',
            'message': 'Planlama tamamlandı'
        }
        return render_template('results.html', summary=summary)
    except Exception as e:
        flash(f'Sonuçlar okunurken hata: {str(e)}', 'error')
        return redirect(url_for('upload_file'))


@app.route('/download')
def download():
    return send_from_directory(
        directory=app.config['OUTPUT_FOLDER'],
        path='results.xlsx',
        as_attachment=True
    )


if __name__ == '__main__':
    app.run(debug=True)