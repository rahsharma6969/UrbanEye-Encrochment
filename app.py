import io
import os
from flask import Flask, request, jsonify, render_template, send_file
import requests
from scripts.run_pipeline import main as run_detection
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from data.locations import PRESET_LOCATIONS
from pathlib import Path
import datetime
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    lat_min = data.get('lat_min')
    lon_min = data.get('lon_min')
    lat_max = data.get('lat_max')
    lon_max = data.get('lon_max')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    try:
        # Create args list for run_detection
        args = [
            # 'dummy',  # First arg is normally script name
            str(lat_min), 
            str(lon_min),
            str(lat_max), 
            str(lon_max),
            start_date,
            end_date
        ]
        
        # Run analysis with args list
        run_detection(args)
        
        # Return results
        return jsonify({
            'success': True,
            'image_url': '/results/change_map.png',
            'report_url': '/results/report.txt'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/results/<path:filename>')
def get_result(filename):
    # Use absolute path and ensure outputs directory exists
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    return send_file(output_dir / filename)

@app.route('/locations')
def get_locations():
    return jsonify(PRESET_LOCATIONS)

@app.route('/download-report', methods=['POST'])
def download_report():
    try:
        data = request.get_json()
        aoi = data.get("aoi", "Unknown AOI")
        start_date = data.get("start_date", "")
        end_date = data.get("end_date", "")
        stats = data.get("statistics", {})
        image_url = data.get("image_url")  # should be like http://127.0.0.1:5000/results/change_map.png

        change_percentage = stats.get("change_percentage")
        building_density = stats.get("building_density_per_km2")
        area_km2 = stats.get("area_km2")

        # Create PDF in memory
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        # Header
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(1 * inch, height - 1 * inch, "UrbanEye Detection Report")

        pdf.setFont("Helvetica", 11)
        pdf.drawString(
            1 * inch,
            height - 1.3 * inch,
            f"Date Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        pdf.line(1 * inch, height - 1.4 * inch, width - 1 * inch, height - 1.4 * inch)

        # AOI & Date info
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(1 * inch, height - 1.8 * inch, "Area of Interest:")
        pdf.setFont("Helvetica", 11)
        pdf.drawString(1.5 * inch, height - 2.0 * inch, aoi)
        pdf.drawString(1 * inch, height - 2.3 * inch, f"Date Range: {start_date} → {end_date}")

        # Statistics Section
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(1 * inch, height - 2.8 * inch, "Detection Metrics:")
        pdf.setFont("Helvetica", 11)
        y = height - 3.1 * inch
        if change_percentage is not None:
            pdf.drawString(1.5 * inch, y, f"Change Area: {change_percentage:.2f}%")
            y -= 0.25 * inch
        if building_density is not None:
            pdf.drawString(1.5 * inch, y, f"Building Density: {building_density:.2f} /km²")
            y -= 0.25 * inch
        if area_km2 is not None:
            pdf.drawString(1.5 * inch, y, f"Total Area: {area_km2:.2f} km²")
            y -= 0.25 * inch

        # Image Section
        if image_url:
            pdf.setFont("Helvetica-Bold", 13)
            pdf.drawString(1 * inch, y - 0.3 * inch, "Detection Map:")
            y -= 0.5 * inch

            try:
                RESULTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")

                # Get just the filename (e.g. change_map.png)
                image_filename = os.path.basename(image_url)
                image_path = os.path.join(RESULTS_DIR, image_filename)

                print("🖼️ Trying to load image from:", image_path)

                if os.path.exists(image_path):
                    with open(image_path, "rb") as f:
                        img_data = ImageReader(io.BytesIO(f.read()))

                    img_width = 5.5 * inch
                    img_height = 3.5 * inch
                    pdf.drawImage(img_data, 1 * inch, y - img_height, width=img_width, height=img_height)
                    y -= img_height + 0.3 * inch
                else:
                    print("⚠️ Image not found at:", image_path)
                    pdf.setFont("Helvetica-Oblique", 10)
                    pdf.drawString(1.5 * inch, y, "(Map image not found at path)")

            except Exception as e:
                print("⚠️ Error loading image:", e)
                pdf.setFont("Helvetica-Oblique", 10)
                pdf.drawString(1.5 * inch, y, f"(Error loading image: {e})")


        # Footer
        pdf.setFont("Helvetica-Oblique", 10)
        pdf.drawString(1 * inch, 0.75 * inch, "© 2025 UrbanEye | AI-based Urban Change Detection System")

        pdf.showPage()
        pdf.save()

        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name="UrbanEye_Report.pdf", mimetype="application/pdf")

    except Exception as e:
        print("❌ Error in download-report:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/test-pdf")
def test_pdf():
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(100, 750, "✅ PDF generation working!")
    pdf.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="test.pdf", mimetype="application/pdf")

if __name__ == '__main__':
    app.run(debug=True)