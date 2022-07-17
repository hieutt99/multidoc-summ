import json
from flask import request, jsonify, Blueprint, abort
from flask.views import MethodViews


class SummaryView(MethodView):
    
    def __init__(self, predictor, *args, **kwargs):
        self.predictor = predictor
        MethodView.__init__(self, *args, **kwargs)

    def post(self, ):
        src_docs = request.form.get("src_docs")
        return jsonify({
            "summary":"summary"
        })