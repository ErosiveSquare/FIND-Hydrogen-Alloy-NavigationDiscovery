import pandas as pd
import joblib
import numpy as np
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import CompositionError
import os

os.environ["JOBLIB_MULTIPROCESSING"] = "0"

MODEL_DIR = r"/home/hongchang/FIND0413/models"

# 吸氢侧配置（仅保留XGBoost）
HYDRO_MODEL_PATHS = {
    'XGBoost': f"{MODEL_DIR}/XGBoost_pipeline14.22.04_Ab10.joblib"
}
HYDRO_METADATA_PATHS = {
    'XGBoost': f"{MODEL_DIR}/XGBoost_metadata14.22.04_Ab10.joblib"
}

# 脱氢侧配置（仅保留XGBoost）
DEHYDRO_MODEL_PATHS = {
    'XGBoost': f"{MODEL_DIR}/XGBoost_pipeline14.22.04_De10.joblib"
}
DEHYDRO_METADATA_PATHS = {
    'XGBoost': f"{MODEL_DIR}/XGBoost_metadata14.22.04_De10.joblib"
}


def clean_feature_name(name):
    """统一处理特征名称的格式化"""
    if name == "temperature(K)":
        return name
    parts = name.replace("MagpieData ", "").split()
    if not parts:
        return name
    return ' '.join([parts[0].capitalize()] + parts[1:])


def load_phase_resources(model_paths, metadata_paths):
    """改进的模型加载方法（保持原有校验逻辑）"""
    models = {}
    metadatas = {}

    for model_name in model_paths.keys():
        try:
            with open(model_paths[model_name], 'rb') as f:
                models[model_name] = joblib.load(f)

            metadata = joblib.load(metadata_paths[model_name])
            required_keys = ['feature_names', 'target_names']
            if not all(k in metadata for k in required_keys):
                raise ValueError(f"元数据缺少必要字段: {required_keys}")

            if 'ln_plateau_pressure(MPa)' not in metadata['target_names']:
                raise ValueError("目标变量中未找到对数压力列")

            metadatas[model_name] = metadata
            print(f"✅ 成功加载 {model_name} (目标变量: {metadata['target_names']})")

        except Exception as e:
            print(f"⚠️  {model_name} 加载失败: {str(e)[:100]}")
            models[model_name] = None
            metadatas[model_name] = None

    return models, metadatas


# 加载双模型资源
print("正在加载吸氢侧模型...")
hydro_models, hydro_metadatas = load_phase_resources(HYDRO_MODEL_PATHS, HYDRO_METADATA_PATHS)
print("\n正在加载脱氢侧模型...")
dehydro_models, dehydro_metadatas = load_phase_resources(DEHYDRO_MODEL_PATHS, DEHYDRO_METADATA_PATHS)


def process_single_prediction(pred_array, metadata):
    """处理单个模型的预测结果（保持原有逻辑）"""
    target_names = metadata['target_names']
    result = {}

    for idx, target in enumerate(target_names):
        raw_value = pred_array[0][idx]

        if target == 'ln_plateau_pressure(MPa)':
            result['plateau_pressure(MPa)'] = round(np.exp(raw_value), 3)
        else:
            result[target] = round(float(raw_value), 3)

    return result


def process_phase(df, phase_models, phase_metadatas):
    """单模型预测处理（简化多模型逻辑）"""
    model_name = 'XGBoost'
    model = phase_models.get(model_name)
    predictions = {}

    if model is None:
        predictions[model_name] = "模型未加载"
        return predictions

    try:
        metadata = phase_metadatas.get(model_name)
        if not metadata:
            raise ValueError("元数据不可用")

        required_features = [clean_feature_name(f) for f in metadata['feature_names']]
        missing_features = set(required_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"缺失特征: {list(missing_features)[:3]}...")

        X = df[required_features].astype(np.float32)
        pred = model.predict(X)

        if pred.shape != (1, 4):
            raise ValueError(f"预测维度错误: {pred.shape}")

        model_pred = process_single_prediction(pred, metadata)
        predictions[model_name] = model_pred

    except Exception as e:
        predictions[model_name] = f"预测失败: {str(e)[:100]}"

    return predictions


def post_process_results(hydro_pred, dehydro_pred):
    """新增结果后处理（实现三个业务逻辑）"""
    model_name = 'XGBoost'

    # 获取双阶段预测结果
    abs_data = hydro_pred.get(model_name, {})
    des_data = dehydro_pred.get(model_name, {})

    # 跳过错误结果处理
    if not isinstance(abs_data, dict) or not isinstance(des_data, dict):
        return hydro_pred, dehydro_pred

    # ① 容量取最大值
    max_cap = max(
        abs_data.get('Max_Capacity(wt%)', 0),
        des_data.get('Max_Capacity(wt%)', 0)
    )
    abs_data['Max_Capacity(wt%)'] = max_cap
    des_data['Max_Capacity(wt%)'] = max_cap

    # ② 压力取绝对值
    abs_pressure = abs(abs_data.get('plateau_pressure(MPa)', 0))
    des_pressure = abs(des_data.get('plateau_pressure(MPa)', 0))
    abs_data['plateau_pressure(MPa)'] = round(abs_pressure, 3)
    des_data['plateau_pressure(MPa)'] = round(des_pressure, 3)

    # ③ 调整压力关系
    if abs_pressure < des_pressure:
        adjusted_pressure = abs_pressure * 0.9
        des_data['plateau_pressure(MPa)'] = round(adjusted_pressure, 3)

    # 更新预测结果
    hydro_pred[model_name] = abs_data
    dehydro_pred[model_name] = des_data

    return hydro_pred, dehydro_pred


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



def predict_properties(composition_str, temperature):
    try:
        df = pd.DataFrame({
            "Components": [composition_str],
            "temperature(K)": [temperature]
        })

        df = StrToComposition().featurize_dataframe(df, "Components")
        ep_feat = ElementProperty.from_preset("magpie", impute_nan=True)
        df = ep_feat.featurize_dataframe(df, col_id="composition")

        df = df.drop(columns=[
            'number', 'Components', 'composition', 'Doi',
            'MagpieData minimum NpValence', 'MagpieData mode NpValence',
            'MagpieData minimum NfValence', 'MagpieData mode NfValence',
            'MagpieData minimum NpUnfilled', 'MagpieData mode NpUnfilled',
            'MagpieData minimum NfUnfilled', 'MagpieData mode NfUnfilled',
            'MagpieData minimum GSbandgap', 'MagpieData maximum GSbandgap',
            'MagpieData range GSbandgap', 'MagpieData mean GSbandgap',
            'MagpieData avg_dev GSbandgap', 'MagpieData mode GSbandgap'
        ], errors="ignore")

        df.rename(columns={col: clean_feature_name(col) for col in df.columns}, inplace=True)
        df["temperature(K)"] = temperature

        hydro_pred = process_phase(df.copy(), hydro_models, hydro_metadatas)
        dehydro_pred = process_phase(df.copy(), dehydro_models, dehydro_metadatas)

        # 应用新增的后处理逻辑
        hydro_pred, dehydro_pred = post_process_results(hydro_pred, dehydro_pred)

        return {
            'Hydrogen Absorption': hydro_pred,
            'Hydrogen Desorption': dehydro_pred
        }

    except CompositionError as ce:
        return f"化学式解析失败: {str(ce)}"
    except Exception as e:
        return f"系统错误: {str(e)}"


def format_output(predictions):
    """调整后的格式化输出（保持表格结构）"""
    target_columns = ['plateau_pressure(MPa)', 'Enthalpy(KJ/mol)',
                      'Entropy(J·mol-1·K-1)', 'Max_Capacity(wt%)']

    results_table = []
    for phase in ['Hydrogen Absorption', 'Hydrogen Desorption']:
        phase_data = predictions.get(phase, {})
        for model, values in phase_data.items():
            row_header = f"{model} ({phase.split()[-1]})"
            if isinstance(values, dict):
                row = [row_header] + [f"{values[col]:.3f}" for col in target_columns]
            else:
                row = [row_header, values, "", "", ""]
            results_table.append(row)

    col_widths = [35] + [18] * 4
    separator = "+" + "+".join(["-" * w for w in col_widths]) + "+"
    header = "|" + "|".join([f"{col:<{w}}" for w, col in zip(col_widths, ["Model"] + target_columns)]) + "|"
    body = ["|" + "|".join([f"{cell:<{w}}" for w, cell in zip(col_widths, row)]) + "|" for row in results_table]

    return "\n".join([separator, header, separator] + body + [separator])


if __name__ == '__main__':
    input_str = input("请输入合金成分（例如 MgNi3）: ").strip()
    temperature = float(input("请输入温度(K): ").strip())
    result = predict_properties(input_str, temperature)

    results = format_output_to_json(result)
    result1 = results["results"][0]

    if isinstance(result, dict):
        print("\nXGBoost模型预测结果：")
        print(format_output(result))
        print(result1)
    else:
        print(result)