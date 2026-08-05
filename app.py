# app.py

import json
from pathlib import Path

from flask import Flask, render_template, request, jsonify

from sentiment_model import ElectionAnalyzer



app = Flask(__name__)

analyzer = ElectionAnalyzer()

PARTY_VALIDATION_PATH = Path(__file__).resolve().parent / 'party_validation.json'



@app.route('/')

def index():

    return render_template('index.html')



@app.route('/analyze', methods=['POST'])

def analyze():

    try:

        data = request.json

        candidate = data.get('candidate')

        source = data.get('source')



        if not candidate:

            return jsonify({"error": "Candidate name is required"}), 400



        result = analyzer.analyze_candidate(candidate, source)

        return jsonify(result)



    except Exception as e:

        return jsonify({"error": str(e)}), 500



@app.route('/api/parties')

def parties():

    if not PARTY_VALIDATION_PATH.exists():

        return jsonify({"error": "party_validation.json not found -- run validate_parties.py first"}), 404

    with open(PARTY_VALIDATION_PATH, encoding='utf-8') as f:

        data = json.load(f)

    return jsonify(data)



if __name__ == '__main__':

    app.run(debug=True, port=5001)
