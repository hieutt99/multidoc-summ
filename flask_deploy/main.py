from flask import Flask
from flask_ngrok import run_with_ngrok
from predictor import build_predictor
from views import SummaryView


predictor = build_predictor()


app = Flask(__name__)
app.logger.setLevel('INFO')
run_with_ngrok(app)   
  
@app.route("/test/", methods=['GET'])
def home():
    return '<h1>test string</h1>'


app.add_url_rule('/api/v1/summarization/', SummaryView.as_view('summary_view', predictor))
    
app.run()