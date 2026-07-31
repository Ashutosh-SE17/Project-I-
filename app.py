# app.py

from flask import Flask, render_template, request, jsonify

from sentiment_model import ElectionAnalyzer



app = Flask(__name__)

analyzer = ElectionAnalyzer()



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



if __name__ == '__main__':

    app.run(debug=True, port=5001)
