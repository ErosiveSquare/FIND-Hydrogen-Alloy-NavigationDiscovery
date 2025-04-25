import pandas as pd
import joblib
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import CompositionError

# 模型路径配置
MODEL_DIR = r"/home/hongchang/FIND0413/joblib"

# 吸氢侧配置
HYDRO_MODEL_PATHS = {
    'XGBoost': f"{MODEL_DIR}/XGBoost.joblib",
}
HYDRO_SCALER_PATH = f"{MODEL_DIR}/scaler.joblib"
HYDRO_FEATURE_NAMES_PATH = f"{MODEL_DIR}/feature_names.joblib"

# 脱氢侧配置
DEHYDRO_MODEL_PATHS = {
    'XGBoost': f"{MODEL_DIR}/XGBoost2.joblib",
}
DEHYDRO_SCALER_PATH = f"{MODEL_DIR}/scaler2.joblib"
DEHYDRO_FEATURE_NAMES_PATH = f"{MODEL_DIR}/feature_names2.joblib"

# 加载配置数据
hydro_feature_names = joblib.load(HYDRO_FEATURE_NAMES_PATH)
hydro_scaler = joblib.load(HYDRO_SCALER_PATH)
hydro_models = {name: joblib.load(path) for name, path in HYDRO_MODEL_PATHS.items()}

dehydro_feature_names = joblib.load(DEHYDRO_FEATURE_NAMES_PATH)
dehydro_scaler = joblib.load(DEHYDRO_SCALER_PATH)
dehydro_models = {name: joblib.load(path) for name, path in DEHYDRO_MODEL_PATHS.items()}


def clean_feature_name(name):
    """统一处理特征名称的格式化"""
    if name == "temperature(K)":
        return name

    parts = name.replace("MagpieData ", "").split()
    if not parts:
        return name

    cleaned_parts = [parts[0].capitalize()] + parts[1:]
    return ' '.join(cleaned_parts)


# 生成清洗后的特征名称列表
cleaned_hydro_features = [clean_feature_name(name) for name in hydro_feature_names]
cleaned_dehydro_features = [clean_feature_name(name) for name in dehydro_feature_names]


def predict_properties(composition_str, temperature):
    try:
        # 创建初始 DataFrame
        df = pd.DataFrame({
            "Components": [composition_str],
            "temperature(K)": [temperature]
        })

        # 特征工程
        try:
            df = StrToComposition().featurize_dataframe(df, "Components")
        except (CompositionError, ValueError, KeyError) as e:
            return f"化学式解析失败: {str(e)}"

        ep_feat = ElementProperty.from_preset(preset_name="magpie", impute_nan=True)
        df = ep_feat.featurize_dataframe(df, col_id="composition")

        # 删除不需要的列
        columns_to_remove = ['number', 'Components', 'Components2', 'composition', 'Doi']
        df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors="ignore")

        # 额外特征清理
        additional_drop = [
            'MagpieData minimum NpValence', 'MagpieData mode NpValence',
            'MagpieData minimum NfValence', 'MagpieData mode NfValence',
            'MagpieData minimum NpUnfilled', 'MagpieData mode NpUnfilled',
            'MagpieData minimum NfUnfilled', 'MagpieData mode NfUnfilled',
            'MagpieData minimum GSbandgap', 'MagpieData maximum GSbandgap',
            'MagpieData range GSbandgap', 'MagpieData mean GSbandgap',
            'MagpieData avg_dev GSbandgap', 'MagpieData mode GSbandgap'
        ]
        df = df.drop(columns=additional_drop, errors='ignore')

        # 确保温度列存在
        df["temperature(K)"] = temperature

        # 统一处理特征名称
        df.rename(columns={col: clean_feature_name(col) for col in df.columns}, inplace=True)

        # 双预测流程
        def process_phase(phase_models, phase_scaler, phase_features):
            valid_features = [col for col in phase_features if col in df.columns]
            if len(valid_features) != len(phase_features):
                return None, f"特征维度不匹配，预期{len(phase_features)}，实际{len(valid_features)}"

            phase_df = df[valid_features]
            scaled = phase_scaler.transform(phase_df)

            predictions = {}
            for model_name, model in phase_models.items():
                try:
                    pred = model.predict(scaled).flatten()
                    # 取绝对值并保留三位小数
                    plateau_pressure = round(abs(pred[0]), 3)
                    predictions[model_name] = {
                        'plateau_pressure(MPa)': plateau_pressure,
                        'Enthalpy(KJ/mol)': round(pred[1], 3),
                        'Entropy(J·mol-1·K-1)': round(pred[2], 3),
                        'Max_Capacity(wt%)': round(pred[3], 3)
                    }
                except Exception as e:
                    predictions[model_name] = f"预测失败: {str(e)}"
            return predictions, None

        # 吸氢侧预测
        hydro_predictions, error = process_phase(hydro_models, hydro_scaler, cleaned_hydro_features)
        if error:
            return error

        # 脱氢侧预测
        dehydro_predictions, error = process_phase(dehydro_models, dehydro_scaler, cleaned_dehydro_features)
        if error:
            return error

        # 对比并选择较大的 Max_Capacity(wt%)
        for model_name in hydro_predictions:
            hydro_capacity = hydro_predictions[model_name]['Max_Capacity(wt%)']
            dehydro_capacity = dehydro_predictions[model_name]['Max_Capacity(wt%)']
            max_capacity = max(hydro_capacity, dehydro_capacity)
            hydro_predictions[model_name]['Max_Capacity(wt%)'] = max_capacity
            dehydro_predictions[model_name]['Max_Capacity(wt%)'] = max_capacity

        return {
            'Hydrogen Absorption': hydro_predictions,
            'Hydrogen Desorption': dehydro_predictions
        }

    except Exception as e:
        return f"整体预测失败: {str(e)}"


"""result = {
            "absorption": {
                "platform_pressure": 0.123,
                "enthalpy": 45.678,
                "entropy": 90.123,
                "capacity": 1.234
            },
            "desorption": {
                "platform_pressure": 0.789,
                "enthalpy": 67.890,
                "entropy": 159.123,
                "capacity": 3.456
            }
        }"""


def format_output_to_json(predictions):
    results = []
    models = set()
    # 收集所有模型名称
    for phase in ['Hydrogen Absorption', 'Hydrogen Desorption']:
        phase_data = predictions.get(phase, {})
        models.update(phase_data.keys())

    for model in models:
        # 初始化吸收和脱附数据
        absorption = {
            "platform_pressure": None,
            "enthalpy": None,
            "entropy": None,
            "capacity": None
        }
        desorption = {
            "platform_pressure": None,
            "enthalpy": None,
            "entropy": None,
            "capacity": None
        }

        # 处理吸收数据
        absorption_data = predictions.get('Hydrogen Absorption', {}).get(model, {})
        if absorption_data:
            absorption.update({
                "platform_pressure": round(float(absorption_data.get('plateau_pressure(MPa)')), 3),
                "enthalpy": round(float(absorption_data.get('Enthalpy(KJ/mol)')), 3),
                "entropy": round(float(absorption_data.get('Entropy(J·mol-1·K-1)')), 3),
                "capacity": round(float(absorption_data.get('Max_Capacity(wt%)')), 3)
            })

        # 处理脱附数据
        desorption_data = predictions.get('Hydrogen Desorption', {}).get(model, {})
        if desorption_data:
            desorption.update({
                "platform_pressure": round(float(desorption_data.get('plateau_pressure(MPa)')), 3),
                "enthalpy": round(float(desorption_data.get('Enthalpy(KJ/mol)')), 3),
                "entropy": round(float(desorption_data.get('Entropy(J·mol-1·K-1)')), 3),
                "capacity": round(float(desorption_data.get('Max_Capacity(wt%)')), 3)
            })

        # 构建结果对象
        result = {
            "absorption": absorption,
            "desorption": desorption
        }
        results.append(result)

    return {"results": results}



def format_output(predictions):
    """增强版格式化输出，支持双预测结果"""
    target_columns = ['plateau_pressure(MPa)', 'Enthalpy(KJ/mol)',
                      'Entropy(J·mol-1·K-1)', 'Max_Capacity(wt%)']

    results_table = []
    # 构建双预测结果
    for phase in ['Hydrogen Absorption', 'Hydrogen Desorption']:
        phase_data = predictions.get(phase, {})
        for model, values in phase_data.items():
            row_header = f"{model} ({phase.split()[-1]})"
            if isinstance(values, dict):
                row = [row_header] + [f"{values[col]:.3f}" for col in target_columns]
            else:
                row = [row_header, values, "", "", ""]
            results_table.append(row)

    # 构建表格格式
    col_widths = [35] + [18] * 4
    separator = "+" + "+".join(["-" * w for w in col_widths]) + "+"
    header = "|" + "|".join([f"{col:<{w}}" for w, col in zip(col_widths, ["Model"] + target_columns)]) + "|"

    body = []
    for row in results_table:
        body_line = "|" + "|".join([f"{str(cell):<{w}}" for w, cell in zip(col_widths, row)]) + "|"
        body.append(body_line)

    return "\n".join([separator, header, separator] + body + [separator])


if __name__ == '__main__':
    input_str = input("请输入合金成分（例如 MgNi3）: ").strip()

    while True:
        try:
            temperature = float(input("请输入温度(K): ").strip())
            break
        except ValueError:
            print("错误: 请输入有效的数值温度！")

    result = predict_properties(input_str, temperature)
    results = format_output_to_json(result)
    result1 = results["results"][0]

    if isinstance(result, dict):
        print("\n各模型预测结果对比：")
        print(format_output(result))
        print(result1)
    else:
        print(result)
