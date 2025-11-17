import streamlit as st
import pandas as pd
import xgboost as xgb
from pathlib import Path
import joblib
import os

# 页面设置
st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🏥", layout="centered")

# 导入必要的库（移到Streamlit初始化之后）
try:
    import shap
    import matplotlib.pyplot as plt
    import numpy as np
    from io import BytesIO
    import base64
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    SHAP_AVAILABLE = True
    
    # 设置专业学术图表样式
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
except ImportError as e:
    SHAP_AVAILABLE = False
    st.error(f"缺少依赖库: {e}")

# ==================== 多语言支持 ====================
class Translation:
    def __init__(self):
        self.chinese = {
            "title": "🏥 糖尿病风险预测系统",
            "subtitle": "基于XGBoost与SHAP解释的智能预测",
            "patient_info": "请输入患者临床指标",
            "all_indicators": "请填写所有19个临床指标：",
            "predict_button": "预测糖尿病风险",
            "result_title": "📊 预测结果",
            "probability": "糖尿病概率",
            "risk_level": "风险等级",
            "samples": "评估样本",
            "medical_advice": "医学建议",
            "shap_analysis": "📈 SHAP特征重要性分析",
            "high_risk": "🔴 高风险",
            "medium_risk": "🟡 中等风险", 
            "low_risk": "🟢 低风险",
            "high_risk_suggestion": "建议立即就医并进行详细检查，包括糖化血红蛋白、口服葡萄糖耐量试验等",
            "medium_risk_suggestion": "建议定期监测血糖，改善饮食和运动习惯，3-6个月后复查",
            "low_risk_suggestion": "保持良好的生活习惯，每年进行常规体检",
            "shap_loading": "正在生成专业SHAP可视化...",
            "language": "语言",
            "urine_guide": "尿常规分级指南",
            "disclaimer_title": "免责声明",
            "disclaimer_content": """
**重要提示 - 请仔细阅读**

**1. 模型性质与局限性声明**

**非诊断工具，仅供辅助参考**：明确声明本模型及其预测结果不能替代专业医生的正式诊断。它仅作为一个辅助性的风险评估和决策支持工具。

**基于概率与统计**：说明模型的预测结果是基于群体数据和统计概率得出的，不具有确定性。它评估的是风险高低，而非给出是或否的绝对结论。

**存在不确定性**：明确指出所有预测都存在一定程度的错误率，包括假阳性（预测有病，实际无病）和假阴性（预测无病，实际有病）。

**2. 适用范围与数据基础声明**

**训练数据来源**：模型基于26,294个医疗样本数据进行训练，包含尿常规和血常规指标。模型在不同人群中的适用性可能有限。

**适用与不适用场景**：
- **适用**：用于高危人群的初步筛查、辅助医生进行鉴别诊断
- **不适用**：不适用于急诊生命决策、不适用于孕妇或特定罕见病患者

**3. 用户/医生责任与义务**

**必须结合临床判断**：强调医生必须将模型预测结果与患者的完整临床信息相结合。

**最终决策责任方**：明确声明最终的诊断和治疗方案责任完全在于主治医生和患者本人。

**禁止患者自行解读与决策**：强烈建议患者不要仅根据模型预测结果进行自我诊断。

**4. 开发者/提供方责任限制**

**按原样提供**：声明模型是按原样和现有提供的，不承诺其准确性、完整性、可靠性。

**不承担医疗责任**：明确免除因使用模型预测结果而导致的任何直接或间接责任。

**5. 数据隐私与安全**

**符合法规**：用户数据的处理将严格遵守相关法律法规。

**匿名化与脱敏**：承诺采取技术措施保护患者隐私。

**6. 知识产权**

声明模型相关的知识产权归开发者所有。
"""
        }
        
        self.english = {
            "title": "🏥 Diabetes Risk Prediction System",
            "subtitle": "Intelligent Prediction with XGBoost and SHAP Explanation",
            "patient_info": "Patient Clinical Indicators",
            "all_indicators": "Please enter all 19 clinical indicators:",
            "predict_button": "Predict Diabetes Risk",
            "result_title": "📊 Prediction Results",
            "probability": "Diabetes Probability",
            "risk_level": "Risk Level",
            "samples": "Evaluated Samples", 
            "medical_advice": "Medical Advice",
            "shap_analysis": "📈 SHAP Feature Importance Analysis",
            "high_risk": "🔴 High Risk",
            "medium_risk": "🟡 Medium Risk",
            "low_risk": "🟢 Low Risk",
            "high_risk_suggestion": "Recommend immediate medical consultation and detailed examination including HbA1c, OGTT, etc.",
            "medium_risk_suggestion": "Recommend regular blood glucose monitoring, improve diet and exercise habits, recheck in 3-6 months",
            "low_risk_suggestion": "Maintain healthy lifestyle habits, undergo routine annual physical examination",
            "shap_loading": "Generating professional SHAP visualization...",
            "language": "Language",
            "urine_guide": "Urinalysis Grading Guide",
            "disclaimer_title": "Disclaimer",
            "disclaimer_content": """
**Important Notice - Please Read Carefully**

**1. Nature and Limitations of the Model**

**Not a Diagnostic Tool, For Reference Only**: This model and its predictions are not a substitute for formal diagnosis by qualified healthcare professionals.

**Based on Probability and Statistics**: Predictions are derived from population data and statistical probabilities, not deterministic.

**Inherent Uncertainty**: All predictions carry error rates including False Positives and False Negatives.

**2. Scope of Application and Data Foundation**

**Training Data Source**: Model was trained on 26,294 medical samples. Applicability may be limited in different populations.

**Intended and Non-Intended Use Cases**:
- **Intended Use**: For preliminary screening, to assist physicians in differential diagnosis
- **Non-Intended Use**: Not for emergency decision making, not for pregnant women

**3. User/Physician Responsibilities and Obligations**

**Must Be Integrated with Clinical Judgment**: Healthcare professionals must integrate predictions with complete clinical information.

**Ultimate Decision-Making Responsibility**: Final diagnosis and treatment decisions rest solely with treating physician.

**Prohibition of Self-Interpretation**: Patients should not use predictions for self-diagnosis.

**4. Liability Limitations**

**As Is Provision**: Model provided as is with no warranties of accuracy or reliability.

**No Medical Liability**: Provider disclaims liability for consequences from model use.

**5. Data Privacy and Security**

**Regulatory Compliance**: Data handling strictly adheres to relevant laws.

**Anonymization**: Technical measures protect patient privacy.

**6. Intellectual Property**

All intellectual property rights belong to the developer.
"""
        }

# 初始化翻译
trans = Translation()

# 语言切换
if 'language' not in st.session_state:
    st.session_state.language = 'chinese'

def get_text(key):
    return trans.chinese[key] if st.session_state.language == 'chinese' else trans.english[key]

# 语言切换按钮
col_lang1, col_lang2, col_lang3 = st.columns([1, 2, 1])
with col_lang2:
    lang_option = st.radio(
        get_text("language"),
        ["中文", "English"],
        horizontal=True,
        index=0 if st.session_state.language == 'chinese' else 1
    )
    
    # 更新语言状态
    if lang_option == "中文":
        st.session_state.language = 'chinese'
    else:
        st.session_state.language = 'english'

# 标题
st.title(get_text("title"))
st.markdown(f"**{get_text('subtitle')}**")
st.markdown("---")

# ==================== 模型加载 ====================
@st.cache_resource
def load_model_and_explainer():
    try:
        st.info("正在加载数据并训练模型..." if st.session_state.language == 'chinese' else "Loading data and training model...")
        data_dir = Path(r'C:\Users\13003\Desktop\Fusion_XGBoost_SHAP_Output')
        train_features = pd.read_csv(data_dir / 'fusion_train_features.csv')
        train_labels = pd.read_csv(data_dir / 'fusion_train_labels.csv')['DiabetesLabel']
        
        # 训练XGBoost模型
        model = xgb.XGBClassifier(
            random_state=42, 
            eval_metric='logloss',
            n_estimators=100,
            max_depth=6
        )
        
        with st.spinner('正在训练模型...' if st.session_state.language == 'chinese' else 'Training model...'):
            model.fit(train_features, train_labels)
        
        # 创建SHAP解释器
        explainer = None
        if SHAP_AVAILABLE:
            with st.spinner('正在创建SHAP解释器...' if st.session_state.language == 'chinese' else 'Creating SHAP explainer...'):
                explainer = shap.TreeExplainer(model)
        
        success_msg = "模型训练完成！" if st.session_state.language == 'chinese' else "Model training completed!"
        if explainer:
            success_msg += " SHAP解释器就绪！" if st.session_state.language == 'chinese' else " SHAP explainer ready!"
        st.success(success_msg)
        return model, explainer, train_features.columns.tolist()
        
    except Exception as e:
        st.error(f"错误: {str(e)}" if st.session_state.language == 'chinese' else f"Error: {str(e)}")
        return None, None, None

model, explainer, feature_names = load_model_and_explainer()

if model is None:
    st.stop()

# ==================== 简洁版SHAP可视化（无小圆点版本） ====================
def create_clean_shap_plot(input_data, prediction_prob):
    """创建简洁版SHAP可视化（无小圆点）"""
    if not SHAP_AVAILABLE or explainer is None:
        return None
        
    try:
        with st.spinner(get_text("shap_loading")):
            # 计算SHAP值
            shap_values = explainer.shap_values(input_data)
            
            # 定义颜色方案
            category_colors = {
                'urine': '#2E86AB',
                'blood': '#A23B72', 
                'common': '#4CAF50'
            }

            def categorize_feature(feature_name):
                if 'Urine' in feature_name:
                    return 'urine'
                elif 'Blood' in feature_name:
                    return 'blood'
                elif feature_name in ['Urine_Gender', 'Urine_Age']:
                    return 'common'
                else:
                    return 'blood'

            # 计算全局特征重要性
            shap_values_array = shap_values.values if hasattr(shap_values, 'values') else shap_values
            global_importance = np.mean(np.abs(shap_values_array), axis=0)

            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Global_Importance': global_importance,
                'Category': [categorize_feature(feat) for feat in feature_names]
            }).sort_values('Global_Importance', ascending=True)

            positive_class_contrib = np.mean(np.maximum(shap_values_array, 0), axis=0)
            negative_class_contrib = np.mean(np.maximum(-shap_values_array, 0), axis=0)

            # 创建图表 - 增加高度避免顶部被遮挡
            fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=(20, 14), 
                                                    gridspec_kw={'width_ratios': [3, 1]})

            sorted_features = feature_importance_df['Feature'].tolist()
            y_pos = np.arange(len(sorted_features))

            # 主图区域
            ax_top = ax_main.twiny()

            max_importance = np.max(global_importance)
            importance_ratios = global_importance / max_importance
            max_bar_value = np.max(global_importance) * 1.2

            # 绘制堆叠条形图（无小圆点）
            for i, feature in enumerate(sorted_features):
                category = categorize_feature(feature)
                color = category_colors[category]
                
                feature_idx = feature_names.index(feature)
                total_importance = global_importance[feature_idx]
                pos_contrib = positive_class_contrib[feature_idx]
                neg_contrib = negative_class_contrib[feature_idx]
                
                ax_top.barh(i, total_importance, color=color, alpha=0.3, height=0.8, 
                           edgecolor=color, linewidth=0.5, left=0)
                ax_top.barh(i, pos_contrib, left=0, color=color, alpha=0.8, height=0.6)
                ax_top.barh(i, neg_contrib, left=pos_contrib, color=color, alpha=0.5, height=0.6)

            ax_top.set_xlim(0, max_bar_value)
            ax_top.set_xlabel('Global Feature Importance\n(Mean |SHAP Value|)', 
                            fontsize=11, fontweight='bold', labelpad=10)
            ax_top.spines['top'].set_visible(True)
            ax_top.tick_params(axis='x', which='major', labelsize=9)
            ax_top.grid(axis='x', alpha=0.3, linestyle='--')

            # 蜂窝图部分
            sorted_indices = [feature_names.index(feat) for feat in sorted_features]
            sorted_shap_values = shap_values_array[:, sorted_indices]
            
            shap_abs_max = np.max(np.abs(sorted_shap_values)) * 1.1
            ax_main.set_xlim(-shap_abs_max, shap_abs_max)

            scatter_plot = None
            for i, feature in enumerate(sorted_features):
                shap_vals = sorted_shap_values[:, i]
                
                scatter = ax_main.scatter(shap_vals, 
                                       [i + np.random.normal(0, 0.08) for _ in range(len(shap_vals))], 
                                       c=shap_vals, cmap='coolwarm', 
                                       s=6, alpha=0.7, edgecolors='none', zorder=5)
                scatter_plot = scatter
                
                # 右侧重要性竖线
                feature_importance = global_importance[feature_names.index(feature)]
                importance_ratio = feature_importance / max_importance
                
                if importance_ratio > 0.66:
                    line_color = '#FF6B6B'
                    line_width = 2.5
                elif importance_ratio > 0.33:
                    line_color = '#FFA726'
                    line_width = 2.0
                else:
                    line_color = '#4CAF50'
                    line_width = 1.5
                
                line_x = shap_abs_max * 1.015
                ax_main.plot([line_x, line_x], [i - 0.35, i + 0.35], 
                           color=line_color, linewidth=line_width, alpha=0.9, zorder=3)

            ax_main.set_xlabel('SHAP Value (Impact on Model Output)', 
                             fontsize=11, fontweight='bold', labelpad=10)
            ax_main.set_ylabel('Features (Sorted by Global Importance)', fontsize=12, fontweight='bold')
            ax_main.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=0.8, zorder=1)
            ax_main.grid(axis='x', alpha=0.2, zorder=0)
            ax_main.tick_params(axis='x', which='major', labelsize=9)

            ax_main.set_yticks(y_pos)
            ax_main.set_yticklabels(sorted_features, fontsize=9)
            ax_main.set_ylim(-0.5, len(sorted_features) - 0.5)

            # 设置主标题
            ax_main.set_title('Fusion XGBoost: Dual-Axis SHAP Analysis', 
                           fontsize=14, fontweight='bold', pad=20)

            # 把Top/Bottom说明放在整个图形的顶部 - 在tight_layout之前添加
            fig.text(0.02, 0.95, 'Top: Global Importance Stacking | Bottom: SHAP Distribution', 
                   fontsize=11, fontweight='normal',
                   verticalalignment='top', horizontalalignment='left')

            # 右侧饼图
            ax_legend.clear()
            category_counts = feature_importance_df['Category'].value_counts()
            colors_pie = [category_colors[cat] for cat in category_counts.index]

            wedges, texts, autotexts = ax_legend.pie(
                category_counts.values,
                labels=[f'{cat.capitalize()}' for cat in category_counts.index],
                colors=colors_pie,
                autopct='%1.1f%%',
                startangle=90,
                explode=[0.03] * len(category_counts),
                shadow=False,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                textprops={'fontsize': 10, 'fontweight': 'bold', 'color': 'white'}
            )

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            ax_legend.set_title('Feature Categories\nDistribution', 
                              fontsize=12, fontweight='bold', pad=20)

            # 图例
            importance_legend = [
                plt.Line2D([0], [0], color='#FF6B6B', linewidth=2.5, label='High Importance'),
                plt.Line2D([0], [0], color='#FFA726', linewidth=2.0, label='Medium Importance'),
                plt.Line2D([0], [0], color='#4CAF50', linewidth=1.5, label='Low Importance'),
            ]

            ax_legend.legend(handles=importance_legend, loc='lower center', frameon=True, 
                            fontsize=9, bbox_to_anchor=(0.5, -0.15))

            if scatter_plot is not None:
                cax = fig.add_axes([0.78, 0.82, 0.15, 0.02])
                cbar = plt.colorbar(scatter_plot, cax=cax, orientation='horizontal')
                cbar.set_label('Feature Value Impact', fontsize=9, fontweight='bold')

            ax_legend.axis('off')
            
            # 在添加所有文字后调用tight_layout
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            plt.close()
            
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            return f"data:image/png;base64,{data}"
            
    except Exception as e:
        st.error(f"SHAP可视化错误: {str(e)}")
        return None

# ==================== 尿常规分级指南 ====================
def show_compact_urine_guide():
    with st.expander(f"🔍 {get_text('urine_guide')}", expanded=False):
        if st.session_state.language == 'chinese':
            st.markdown("""
            **分级标准:**
            - **尿糖**: 0(-),1(1+),2(2+),3(3+),4(4+)
            - **尿蛋白**: 0(-),1(±),2(1+),3(2+),4(3+)
            - **尿酮体**: 0(-),1(±),2(1+),3(2+),4(3+)
            - **尿潜血**: 0(-),1(±),2(1+),3(2+),4(3+)
            - **尿比重**: 1(<1.010),2(1.010-1.025),3(>1.025),4(异常)
            """)
        else:
            st.markdown("""
            **Grading Standards:**
            - **Glucose**: 0(-),1(1+),2(2+),3(3+),4(4+)
            - **Protein**: 0(-),1(±),2(1+),3(2+),4(3+)
            - **Ketone**: 0(-),1(±),2(1+),3(2+),4(3+)
            - **Occult Blood**: 0(-),1(±),2(1+),3(2+),4(3+)
            - **Specific Gravity**: 1(<1.010),2(1.010-1.025),3(>1.025),4(Abnormal)
            """)

# ==================== 输入表单 ====================
st.markdown("---")
st.subheader(get_text("patient_info"))
show_compact_urine_guide()

with st.form("prediction_form"):
    st.write(f"**{get_text('all_indicators')}**")
    
    col1, col2, col3 = st.columns(3)
    
    input_values = {}
    
    with col1:
        input_values['Urine_GlucoseGrade'] = st.selectbox("Urine Glucose Grade" if st.session_state.language == 'english' else "尿糖等级", [0, 1, 2, 3, 4])
        input_values['Blood_MediumFluorescenceReticulocyte'] = st.number_input("Medium Fluorescence Reticulocyte(%)" if st.session_state.language == 'english' else "中荧光网织红细胞(%)", value=1.5, step=0.1)
        input_values['Urine_ProteinGrade'] = st.selectbox("Urine Protein Grade" if st.session_state.language == 'english' else "尿蛋白等级", [0, 1, 2, 3, 4])
        input_values['Blood_LowFluorescenceReticulocyte'] = st.number_input("Low Fluorescence Reticulocyte(%)" if st.session_state.language == 'english' else "低荧光网织红细胞(%)", value=80.0, step=0.1)
        input_values['Urine_Gender'] = st.selectbox("Gender" if st.session_state.language == 'english' else "性别", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male" if st.session_state.language == 'english' else "女性" if x == 0 else "男性")
    
    with col2:
        input_values['Blood_LymphocyteCount'] = st.number_input("Lymphocyte Count" if st.session_state.language == 'english' else "淋巴细胞计数", value=2.0, step=0.1)
        input_values['Blood_RDW_CV'] = st.number_input("RDW-CV" if st.session_state.language == 'english' else "红细胞分布宽度CV", value=13.0, step=0.1)
        input_values['Blood_MCHC'] = st.number_input("MCHC" if st.session_state.language == 'english' else "平均血红蛋白浓度", value=330.0, step=1.0)
        input_values['Urine_UrineSpecificGravity'] = st.number_input("Urine Specific Gravity" if st.session_state.language == 'english' else "尿比重", value=1.015, step=0.001)
        input_values['Blood_LargePlateletRatio'] = st.number_input("Large Platelet Ratio(%)" if st.session_state.language == 'english' else "大血小板比率(%)", value=30.0, step=0.1)
    
    with col3:
        input_values['Urine_KetoneGrade'] = st.selectbox("Urine Ketone Grade" if st.session_state.language == 'english' else "尿酮体等级", [0, 1, 2, 3, 4])
        input_values['Blood_PlateletDistributionWidth'] = st.number_input("Platelet Distribution Width" if st.session_state.language == 'english' else "血小板分布宽度", value=10.0, step=0.1)
        input_values['Blood_PlateletCount'] = st.number_input("Platelet Count" if st.session_state.language == 'english' else "血小板计数", value=250.0, step=1.0)
        input_values['Blood_BasophilCount'] = st.number_input("Basophil Count" if st.session_state.language == 'english' else "嗜碱性粒细胞计数", value=0.02, step=0.01)
        input_values['Urine_SpecificGravityGrade'] = st.selectbox("Specific Gravity Grade" if st.session_state.language == 'english' else "尿比重等级", [0, 1, 2, 3, 4])
    
    col4, col5 = st.columns(2)
    with col4:
        input_values['Urine_Age'] = st.number_input("Age" if st.session_state.language == 'english' else "年龄", value=45)
        input_values['Blood_MCH'] = st.number_input("MCH" if st.session_state.language == 'english' else "平均血红蛋白量", value=30.0, step=0.1)
    with col5:
        input_values['Urine_OccultBloodGrade'] = st.selectbox("Occult Blood Grade" if st.session_state.language == 'english' else "尿潜血等级", [0, 1, 2, 3, 4])
        input_values['Blood_EosinophilCount'] = st.number_input("Eosinophil Count" if st.session_state.language == 'english' else "嗜酸性粒细胞计数", value=0.1, step=0.01)
    
    submitted = st.form_submit_button(get_text("predict_button"))

# ==================== 预测结果 ====================
if submitted:
    try:
        features = [[input_values[feature] for feature in feature_names]]
        input_data = pd.DataFrame(features, columns=feature_names)
        
        with st.spinner('正在进行预测...' if st.session_state.language == 'chinese' else 'Predicting...'):
            probability = model.predict_proba(input_data)[0][1] * 100
        
        if probability > 70:
            risk_level = get_text("high_risk")
            suggestion = get_text("high_risk_suggestion")
        elif probability > 30:
            risk_level = get_text("medium_risk")
            suggestion = get_text("medium_risk_suggestion")
        else:
            risk_level = get_text("low_risk")
            suggestion = get_text("low_risk_suggestion")
        
        st.markdown("---")
        st.subheader(get_text("result_title"))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(get_text("probability"), f"{probability:.1f}%")
        with col2:
            st.metric(get_text("risk_level"), risk_level)
        with col3:
            st.metric(get_text("samples"), "19 indicators")
        
        st.progress(float(probability / 100))
        st.info(f"**{get_text('medical_advice')}**: {suggestion}")
        
        # SHAP可视化
        if SHAP_AVAILABLE and explainer is not None:
            st.markdown("---")
            st.subheader(get_text("shap_analysis"))
            
            # 在SHAP图上方添加间距，避免被遮挡
            st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
            
            shap_image = create_clean_shap_plot(input_data, probability)
            if shap_image:
                st.image(shap_image, use_container_width=True)
            
            # ==================== 完整免责声明（使用expander） ====================
            st.markdown("---")
            expander_label = f"📝 {get_text('disclaimer_title')} - 点击展开阅读完整声明" if st.session_state.language == 'chinese' else f"📝 {get_text('disclaimer_title')} - Click to expand full disclaimer"
            with st.expander(expander_label, expanded=True):
                st.warning(get_text("disclaimer_content"))
            
    except Exception as e:
        st.error(f"预测失败: {str(e)}" if st.session_state.language == 'chinese' else f"Prediction failed: {str(e)}")

# 侧边栏
with st.sidebar:
    st.header("System Information" if st.session_state.language == 'english' else "系统信息")
    st.info("XGBoost-based Diabetes Risk Prediction System" if st.session_state.language == 'english' else "基于XGBoost的糖尿病风险预测系统")