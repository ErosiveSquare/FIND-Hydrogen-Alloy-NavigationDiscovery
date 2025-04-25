from flask import Flask, jsonify, request, send_from_directory, render_template, redirect
import jwt
import sqlite3
from functools import wraps
from flask_cors import CORS
from datetime import datetime, timedelta
import time
from statsmodels.graphics.tukeyplot import results
from werkzeug.exceptions import BadRequest
from werkzeug.security import generate_password_hash, check_password_hash
from new_forward import predict_properties,format_output_to_json

from modpack import TPC_to_formula
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) # 允许所有域跨域访问
app.config['SECRET_KEY'] = '123'
app.config['DATABASE'] = 'users.db'
"""Ti16Cr22Zr5V54Fe2Mn1 298"""

"""
T = 329.2256
P = 0.5826
C = 1.5990

"""

# 初始化数据库
def init_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    # c.execute('''CREATE TABLE IF NOT EXISTS users
    #              (id INTEGER PRIMARY KEY AUTOINCREMENT,
    #               username TEXT UNIQUE NOT NULL,
    #               password TEXT NOT NULL)''')
    # 新增预测记录表
    print("建立数据库")
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     user_id TEXT,
                     alloy TEXT NOT NULL,
                     temperature REAL NOT NULL,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    print("建立好了")
    conn.close()


# 数据库查询助手
def query_db(query, args=(), one=False):
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query, args)
    result = cur.fetchall()
    conn.commit()
    conn.close()
    return (result[0] if result else None) if one else result


# 注册路由
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    # 添加调试输出
    print("[Register] 注册用户:", username)
    print("[Register] 原始密码:", password)

    if not username or not password:
        return jsonify({'message': '需要用户名和密码'}), 400

    existing_user = query_db('SELECT * FROM users WHERE username = ?', [username], one=True)
    if existing_user:
        return jsonify({'message': '用户名已存在'}), 400

    hashed_pw = generate_password_hash(password)
    print("[Register] 生成的哈希:", hashed_pw)
    query_db('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))

    return jsonify({'message': '注册成功'}), 201

@app.route('/line')
def line():
    # 渲染前端页面
    return render_template('LinePre.html')

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


@app.route('/')
def index():
    # 渲染前端页面
    return render_template('login.html')
# 登录路由
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = query_db('SELECT * FROM users WHERE username = ?', [username], one=True)
    if not user or not check_password_hash(user['password'], password):
        return jsonify({'message': '无效的用户名或密码'}), 401

    token = jwt.encode({
        'sub': str(user['id']),  # 将用户ID转换为字符串
        'exp': datetime.utcnow() + timedelta(hours=1)
    }, app.config['SECRET_KEY'], algorithm='HS256')

    return jsonify({'token': token}), 200




def predict_result(alloy_composition, temperature):
    try:

        result = predict_properties(alloy_composition, temperature)
        results = format_output_to_json(result)
        result1 = results["results"][0]
        print(result1)

        return result1
    except Exception as e:
        return {"error": str(e)}

@app.route('/register2')
def register2():
    return render_template('registe_424.html')

@app.route('/reverse_predict', methods=['POST'])
def reverse_predict():
    try:
        # 1. 获取并验证请求数据
        data = request.get_json()
        print("backward界面前端返回：")
        print(data)
        if not data:
            raise BadRequest("请求必须包含JSON数据")

        # 2. 检查必需字段
        required_fields = ['pressure', 'capacity', 'temperature']
        for field in required_fields:
            if field not in data:
                raise BadRequest(f"缺少必需字段: {field}")

        # 3. 类型验证
        try:
            pressure = float(data['pressure'])
            capacity = float(data['capacity'])
            temperature = int(data['temperature'])
        except ValueError as e:
            raise BadRequest(f"参数类型错误: {str(e)}")

        print("TPC->formula:")
        result = TPC_to_formula(temperature,pressure,capacity)
        print(result)

        # 6. 返回标准格式的响应
        return jsonify({
            "predicted_alloy": result,
        }), 200

    except BadRequest as e:
        return jsonify({
            "status": "error",
            "message": str(e.description)
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"服务器内部错误: {str(e)}"
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    # 获取前端发送的 JSON 数据
    data = request.get_json()
    print("前端输入发送json：")
    print(data)
    alloy = data.get('alloy')
    temperature = data.get('temperature')

    # 获取并尝试解码token
    user_id = "unknown"
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        try:
            token = auth_header.split()[1]
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'],
                                 options={'verify_signature': False})
            user_id = payload.get('sub', 'unknown')
        except Exception as e:
            print(f"Token解码错误: {str(e)}")

    # 打印当前用户ID
    print(f"当前用户ID: {user_id}")

    # 记录到数据库
    try:
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        c.execute('INSERT INTO predictions (user_id, alloy, temperature) VALUES (?, ?, ?)',
                  (user_id, alloy, temperature))
        conn.commit()
    except sqlite3.Error as e:
        print(f"数据库写入失败: {str(e)}")
    finally:
        conn.close()

    print("开始传参：result = predict_result(alloy, temperature)")
    print(alloy)
    print(temperature)
    # 调用预测函数
    result = predict_result(alloy, temperature)
    print("预测结果后端json：")
    print(result)
    # 返回预测结果为 JSON 格式
    return jsonify(result)

@app.route('/advanced')
def advanced():
    return render_template('LinePre.html')

@app.route('/forward')
def forward():
    token = request.cookies.get('token') or request.args.get('token')

    if not token:
        return redirect('/')

    try:
        decoded = jwt.decode(
            token,
            app.config['SECRET_KEY'],
            algorithms=['HS256'],
            options={'require_sub': True}  # 强制要求sub声明存在
        )
        print(f"解码后的token内容: {decoded}")
        return render_template('forward.html')
    except jwt.ExpiredSignatureError:
        print("token已过期")
        return redirect('/')
    except jwt.InvalidTokenError as e:
        print(f"token无效: {str(e)}")
        return redirect('/')

@app.route('/forward_white')
def forward_white():
    token = request.cookies.get('token') or request.args.get('token')

    if not token:
        return redirect('/')

    try:
        decoded = jwt.decode(
            token,
            app.config['SECRET_KEY'],
            algorithms=['HS256'],
            options={'require_sub': True}  # 强制要求sub声明存在
        )
        print(f"解码后的token内容: {decoded}")
        return render_template('forward_white.html')
    except jwt.ExpiredSignatureError:
        print("token已过期")
        return redirect('/')
    except jwt.InvalidTokenError as e:
        print(f"token无效: {str(e)}")
        return redirect('/')


@app.route('/backward')
def backward():
    # 渲染前端页面
    return render_template('backward.html')

# 受保护的路由示例
@app.route('/dashboard')
def dashboard():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'message': '缺少访问令牌'}), 401

    try:
        data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        user_id = data['sub']
        user = query_db('SELECT * FROM users WHERE id = ?', [user_id], one=True)
        return jsonify({'message': f'欢迎回来，{user["username"]}!'})
    except jwt.ExpiredSignatureError:
        return jsonify({'message': '令牌已过期'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'message': '无效令牌'}), 401

if __name__ == '__main__':
    #init_db()  # 初始化数据库表
    app.run(host='0.0.0.0', port=5005, debug=False)
