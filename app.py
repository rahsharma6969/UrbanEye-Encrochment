from flask import Flask, request, jsonify, render_template, send_file
from scripts.run_pipeline import main as run_detection
from data.locations import PRESET_LOCATIONS
import os
from pathlib import Path

app = Flask(__name__)

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
            'dummy',  # First arg is normally script name
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

if __name__ == '__main__':
    app.run(debug=True)