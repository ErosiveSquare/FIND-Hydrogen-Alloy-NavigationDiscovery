from flask import Flask, request, jsonify, render_template
from functools import wraps
from demo import predict_result
import time

app = Flask(__name__)

def log_request(f):
    """日志装饰器"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        app.logger.info(f"Incoming request: {request.method} {request.path}")
        if request.is_json:
            app.logger.debug(f"Request JSON: {request.json}")
        return f(*args, **kwargs)

    return decorated_function


@app.route('/predict2', methods=['POST'])
@log_request
def predict_single():
    """
    处理单个合金的预测请求
    请求格式: {"alloy": "Ti0.5Y0.3Na0.2", "temperature": 298}
    """
    data = request.get_json()

    # 验证输入
    if not data or 'alloy' not in data or 'temperature' not in data:
        return jsonify({"error": "Missing required fields (alloy, temperature)"}), 400

    try:
        alloy = str(data['alloy'])
        temperature = float(data['temperature'])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input format: {str(e)}"}), 400

    # 调用预测函数
    result = predict_result(alloy, temperature)

    # 添加元数据
    response = {
        "alloy": alloy,
        "temperature": temperature,
        "prediction": result,
        "timestamp": time.time()
    }

    return jsonify(response)


@app.route('/')
def index():
    # 渲染前端页面
    return render_template('LinePre.html')


@app.route('/batch_predict', methods=['POST'])
@log_request
def batch_predict():
    """
    处理批量预测请求
    请求格式: {
        "alloys": ["Ti0.5Y0.3Na0.2", "Ti0.6Y0.2Na0.2"],
        "temperature": 298
    }
    返回格式: {
        "temperature": 298,
        "results": [
            {
                "alloy": "Ti0Cr0.5Na0.2V0.3",
                "absorption": {
                    "platform_pressure": 1.033,
                    "enthalpy": -27.658,
                    "entropy": -70.57,
                    "capacity": 3.135
                },
                "desorption": {...}
            },
            ...
        ]
    }
    """
    data = request.get_json()

    # 验证输入
    if not data or 'alloys' not in data or 'temperature' not in data:
        return jsonify({"error": "Missing required fields (alloys, temperature)"}), 400

    try:
        alloys = list(data['alloys'])
        temperature = float(data['temperature'])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input format: {str(e)}"}), 400

    # 批量预测
    results = []
    for alloy in alloys:
        try:
            prediction = predict_result(str(alloy), temperature)
            results.append({
                "alloy": alloy,
                "absorption": prediction['absorption'],
                "desorption": prediction['desorption']
            })
        except Exception as e:
            app.logger.error(f"Prediction failed for {alloy}: {str(e)}")
            results.append({
                "alloy": alloy,
                "error": str(e)
            })

    return jsonify({
        "temperature": temperature,
        "count": len(results),
        "success_count": len([r for r in results if 'absorption' in r]),
        "results": results
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)