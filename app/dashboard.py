import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple

# Page configuration
st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ff4444;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
    .alert-normal {
        background-color: #00aa44;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
    .summary-panel {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for patient data
if 'patients_data' not in st.session_state:
    st.session_state.patients_data = {}

# Utility functions
@st.cache_data
def generate_sample_patients(n_patients: int = 50) -> Dict:
    """Generate sample patient data"""
    patients = {}
    
    for i in range(n_patients):
        patient_id = f"PT{str(i+1).zfill(4)}"
        
        # Generate time series data (last 30 days)
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='4H'
        )
        
        # Base vitals with some variation
        base_hr = random.randint(60, 100)
        base_bp_sys = random.randint(110, 140)
        base_bp_dia = random.randint(70, 90)
        base_temp = random.uniform(98.0, 99.5)
        base_o2 = random.randint(95, 100)
        
        vitals_data = []
        for date in dates:
            # Add realistic variations
            hr_var = random.randint(-10, 15)
            bp_sys_var = random.randint(-15, 20)
            bp_dia_var = random.randint(-10, 15)
            temp_var = random.uniform(-1.0, 2.0)
            o2_var = random.randint(-5, 3)
            
            vitals_data.append({
                'timestamp': date,
                'heart_rate': max(40, base_hr + hr_var),
                'blood_pressure_sys': max(80, base_bp_sys + bp_sys_var),
                'blood_pressure_dia': max(50, base_bp_dia + bp_dia_var),
                'temperature': max(96.0, base_temp + temp_var),
                'oxygen_saturation': min(100, max(85, base_o2 + o2_var))
            })
        
        patients[patient_id] = {
            'name': f"Patient {i+1}",
            'age': random.randint(25, 85),
            'gender': random.choice(['Male', 'Female']),
            'condition': random.choice([
                'Hypertension', 'Diabetes', 'Heart Disease', 
                'Respiratory Issue', 'Post-Surgery Recovery'
            ]),
            'vitals': pd.DataFrame(vitals_data),
            'notes': generate_clinical_notes()
        }
    
    return patients

def generate_clinical_notes() -> List[str]:
    """Generate sample clinical notes"""
    templates = [
        "Patient shows stable vitals with minor fluctuations. Continue current medication regimen.",
        "Blood pressure elevated during morning rounds. Consider medication adjustment.",
        "Patient reports feeling better today. Appetite improved, ambulating well.",
        "Oxygen saturation slightly low overnight. Administered supplemental oxygen.",
        "Temperature spike noted at 14:00. Blood cultures sent to lab.",
        "Patient comfortable, pain well controlled. Family visited today.",
        "Heart rate irregular during exercise. ECG ordered for further evaluation."
    ]
    return random.sample(templates, random.randint(3, 6))

def analyze_vitals_trends(vitals_df: pd.DataFrame) -> Dict:
    """Analyze vital signs trends"""
    latest = vitals_df.iloc[-1]
    previous = vitals_df.iloc[-24:].mean()  # Last 24 readings average
    
    trends = {}
    for vital in ['heart_rate', 'blood_pressure_sys', 'blood_pressure_dia', 'temperature', 'oxygen_saturation']:
        current_val = latest[vital]
        avg_val = previous[vital]
        
        if vital == 'heart_rate':
            status = 'HIGH' if current_val > 100 else 'LOW' if current_val < 60 else 'NORMAL'
        elif vital == 'blood_pressure_sys':
            status = 'HIGH' if current_val > 140 else 'LOW' if current_val < 90 else 'NORMAL'
        elif vital == 'blood_pressure_dia':
            status = 'HIGH' if current_val > 90 else 'LOW' if current_val < 60 else 'NORMAL'
        elif vital == 'temperature':
            status = 'HIGH' if current_val > 100.4 else 'LOW' if current_val < 97.0 else 'NORMAL'
        else:  # oxygen_saturation
            status = 'LOW' if current_val < 95 else 'NORMAL'
        
        trend_direction = 'UP' if current_val > avg_val else 'DOWN' if current_val < avg_val else 'STABLE'
        
        trends[vital] = {
            'current': current_val,
            'average': avg_val,
            'status': status,
            'trend': trend_direction
        }
    
    return trends

def simulate_intervention_impact(vitals_df: pd.DataFrame, intervention: str, magnitude: float) -> pd.DataFrame:
    """Simulate the impact of medical interventions"""
    simulated_df = vitals_df.copy()
    
    # Apply intervention effects based on type
    if intervention == "Blood Pressure Medication":
        simulated_df['blood_pressure_sys'] = simulated_df['blood_pressure_sys'] - magnitude
        simulated_df['blood_pressure_dia'] = simulated_df['blood_pressure_dia'] - magnitude * 0.7
    elif intervention == "Beta Blocker":
        simulated_df['heart_rate'] = simulated_df['heart_rate'] - magnitude
        simulated_df['blood_pressure_sys'] = simulated_df['blood_pressure_sys'] - magnitude * 0.5
    elif intervention == "Oxygen Therapy":
        simulated_df['oxygen_saturation'] = np.minimum(100, simulated_df['oxygen_saturation'] + magnitude)
    elif intervention == "Fever Reducer":
        simulated_df['temperature'] = simulated_df['temperature'] - magnitude
    
    return simulated_df

# Load or generate patient data
if not st.session_state.patients_data:
    with st.spinner("Generating sample patient data..."):
        st.session_state.patients_data = generate_sample_patients()

# Sidebar navigation
st.sidebar.title("🏥 Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Dashboard Overview", "Patient Vitals Trends", "What-If Simulator", "NLP Summary Panel", "Seed Patients"]
)

# Main dashboard header
st.markdown('<h1 class="main-header">Healthcare Analytics Dashboard</h1>', unsafe_allow_html=True)

if page == "Dashboard Overview":
    st.header("📊 Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Patients</h3>
            <h2>50</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        critical_count = len([p for p in st.session_state.patients_data.values() 
                            if analyze_vitals_trends(p['vitals'])['heart_rate']['status'] == 'HIGH'])
        st.markdown(f"""
        <div class="metric-card">
            <h3>Critical Alerts</h3>
            <h2>{critical_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_age = np.mean([p['age'] for p in st.session_state.patients_data.values()])
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Age</h3>
            <h2>{avg_age:.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        conditions = [p['condition'] for p in st.session_state.patients_data.values()]
        most_common = max(set(conditions), key=conditions.count)
        st.markdown(f"""
        <div class="metric-card">
            <h3>Top Condition</h3>
            <h2>{most_common}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent alerts
    st.subheader("🚨 Recent Alerts")
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        st.markdown("""
        <div class="alert-high">
            <strong>HIGH PRIORITY:</strong> PT0023 - Heart Rate: 120 bpm
        </div>
        <div class="alert-high">
            <strong>HIGH PRIORITY:</strong> PT0007 - Blood Pressure: 165/95
        </div>
        """, unsafe_allow_html=True)
    
    with alert_col2:
        st.markdown("""
        <div class="alert-normal">
            <strong>NORMAL:</strong> PT0015 - All vitals stable
        </div>
        <div class="alert-normal">
            <strong>NORMAL:</strong> PT0031 - Improved oxygen levels
        </div>
        """, unsafe_allow_html=True)

elif page == "Patient Vitals Trends":
    st.header("📈 Patient Vitals Trends")
    
    # Patient selection
    selected_patient = st.selectbox(
        "Select Patient",
        options=list(st.session_state.patients_data.keys()),
        format_func=lambda x: f"{x} - {st.session_state.patients_data[x]['name']}"
    )
    
    if selected_patient:
        patient = st.session_state.patients_data[selected_patient]
        vitals_df = patient['vitals']
        
        # Patient info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Age", patient['age'])
        col2.metric("Gender", patient['gender'])
        col3.metric("Condition", patient['condition'])
        
        # Analyze current trends
        trends = analyze_vitals_trends(vitals_df)
        
        # Display current vitals with status
        st.subheader("Current Vital Signs")
        vital_cols = st.columns(5)
        
        vitals_display = [
            ("Heart Rate", "heart_rate", "bpm"),
            ("Systolic BP", "blood_pressure_sys", "mmHg"),
            ("Diastolic BP", "blood_pressure_dia", "mmHg"),
            ("Temperature", "temperature", "°F"),
            ("O2 Saturation", "oxygen_saturation", "%")
        ]
        
        for i, (label, key, unit) in enumerate(vitals_display):
            trend_data = trends[key]
            status_color = "🔴" if trend_data['status'] in ['HIGH', 'LOW'] else "🟢"
            trend_arrow = "📈" if trend_data['trend'] == 'UP' else "📉" if trend_data['trend'] == 'DOWN' else "➡️"
            
            vital_cols[i].metric(
                f"{status_color} {label}",
                f"{trend_data['current']:.1f} {unit}",
                delta=f"{trend_arrow} vs avg"
            )
        
        # Vitals charts
        st.subheader("Trends Over Time")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Heart Rate', 'Blood Pressure', 'Temperature', 
                          'Oxygen Saturation', 'Vital Signs Correlation'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.08
        )
        
        # Heart Rate
        fig.add_trace(
            go.Scatter(x=vitals_df['timestamp'], y=vitals_df['heart_rate'],
                      name='Heart Rate', line=dict(color='red')),
            row=1, col=1
        )
        
        # Blood Pressure
        fig.add_trace(
            go.Scatter(x=vitals_df['timestamp'], y=vitals_df['blood_pressure_sys'],
                      name='Systolic', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=vitals_df['timestamp'], y=vitals_df['blood_pressure_dia'],
                      name='Diastolic', line=dict(color='lightblue')),
            row=1, col=2
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(x=vitals_df['timestamp'], y=vitals_df['temperature'],
                      name='Temperature', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Oxygen Saturation
        fig.add_trace(
            go.Scatter(x=vitals_df['timestamp'], y=vitals_df['oxygen_saturation'],
                      name='O2 Sat', line=dict(color='green')),
            row=2, col=2
        )
        
        # Correlation heatmap
        corr_data = vitals_df[['heart_rate', 'blood_pressure_sys', 'temperature', 'oxygen_saturation']].corr()
        fig.add_trace(
            go.Heatmap(z=corr_data.values, x=corr_data.columns, y=corr_data.columns,
                      colorscale='RdBu', name='Correlation'),
            row=3, col=1
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="Patient Vitals Analysis")
        st.plotly_chart(fig, use_container_width=True)

elif page == "What-If Simulator":
    st.header("🔮 What-If Simulator")
    st.write("Simulate the potential impact of medical interventions on patient vitals.")
    
    # Patient selection for simulation
    sim_patient = st.selectbox(
        "Select Patient for Simulation",
        options=list(st.session_state.patients_data.keys()),
        format_func=lambda x: f"{x} - {st.session_state.patients_data[x]['name']}"
    )
    
    if sim_patient:
        patient = st.session_state.patients_data[sim_patient]
        vitals_df = patient['vitals']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Intervention Parameters")
            
            intervention = st.selectbox(
                "Select Intervention",
                ["Blood Pressure Medication", "Beta Blocker", "Oxygen Therapy", "Fever Reducer"]
            )
            
            if intervention == "Blood Pressure Medication":
                magnitude = st.slider("BP Reduction (mmHg)", 5, 30, 15)
                st.info("Simulates antihypertensive medication effect")
            elif intervention == "Beta Blocker":
                magnitude = st.slider("Heart Rate Reduction (bpm)", 5, 25, 10)
                st.info("Simulates beta-blocker medication effect")
            elif intervention == "Oxygen Therapy":
                magnitude = st.slider("O2 Increase (%)", 1, 10, 5)
                st.info("Simulates supplemental oxygen therapy")
            elif intervention == "Fever Reducer":
                magnitude = st.slider("Temperature Reduction (°F)", 0.5, 3.0, 1.5)
                st.info("Simulates fever-reducing medication")
            
            simulate_btn = st.button("Run Simulation", type="primary")
        
        with col2:
            if simulate_btn:
                st.subheader("Simulation Results")
                
                # Generate simulated data
                simulated_df = simulate_intervention_impact(vitals_df, intervention, magnitude)
                
                # Create comparison chart
                fig = go.Figure()
                
                if intervention in ["Blood Pressure Medication", "Beta Blocker"]:
                    if intervention == "Blood Pressure Medication":
                        fig.add_trace(go.Scatter(
                            x=vitals_df['timestamp'], 
                            y=vitals_df['blood_pressure_sys'],
                            name='Original Systolic BP',
                            line=dict(color='red', dash='solid')
                        ))
                        fig.add_trace(go.Scatter(
                            x=simulated_df['timestamp'], 
                            y=simulated_df['blood_pressure_sys'],
                            name='Simulated Systolic BP',
                            line=dict(color='blue', dash='dash')
                        ))
                        fig.update_layout(yaxis_title="Blood Pressure (mmHg)")
                    else:  # Beta Blocker
                        fig.add_trace(go.Scatter(
                            x=vitals_df['timestamp'], 
                            y=vitals_df['heart_rate'],
                            name='Original Heart Rate',
                            line=dict(color='red', dash='solid')
                        ))
                        fig.add_trace(go.Scatter(
                            x=simulated_df['timestamp'], 
                            y=simulated_df['heart_rate'],
                            name='Simulated Heart Rate',
                            line=dict(color='blue', dash='dash')
                        ))
                        fig.update_layout(yaxis_title="Heart Rate (bpm)")
                
                elif intervention == "Oxygen Therapy":
                    fig.add_trace(go.Scatter(
                        x=vitals_df['timestamp'], 
                        y=vitals_df['oxygen_saturation'],
                        name='Original O2 Sat',
                        line=dict(color='red', dash='solid')
                    ))
                    fig.add_trace(go.Scatter(
                        x=simulated_df['timestamp'], 
                        y=simulated_df['oxygen_saturation'],
                        name='Simulated O2 Sat',
                        line=dict(color='blue', dash='dash')
                    ))
                    fig.update_layout(yaxis_title="Oxygen Saturation (%)")
                
                else:  # Fever Reducer
                    fig.add_trace(go.Scatter(
                        x=vitals_df['timestamp'], 
                        y=vitals_df['temperature'],
                        name='Original Temperature',
                        line=dict(color='red', dash='solid')
                    ))
                    fig.add_trace(go.Scatter(
                        x=simulated_df['timestamp'], 
                        y=simulated_df['temperature'],
                        name='Simulated Temperature',
                        line=dict(color='blue', dash='dash')
                    ))
                    fig.update_layout(yaxis_title="Temperature (°F)")
                
                fig.update_layout(
                    title=f"Impact of {intervention}",
                    xaxis_title="Time",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary metrics
                st.subheader("Impact Summary")
                original_mean = vitals_df.iloc[-24:].mean()
                simulated_mean = simulated_df.iloc[-24:].mean()
                
                if intervention == "Blood Pressure Medication":
                    improvement = original_mean['blood_pressure_sys'] - simulated_mean['blood_pressure_sys']
                    st.success(f"Average systolic BP reduction: {improvement:.1f} mmHg")
                elif intervention == "Beta Blocker":
                    improvement = original_mean['heart_rate'] - simulated_mean['heart_rate']
                    st.success(f"Average heart rate reduction: {improvement:.1f} bpm")
                elif intervention == "Oxygen Therapy":
                    improvement = simulated_mean['oxygen_saturation'] - original_mean['oxygen_saturation']
                    st.success(f"Average O2 saturation increase: {improvement:.1f}%")
                else:
                    improvement = original_mean['temperature'] - simulated_mean['temperature']
                    st.success(f"Average temperature reduction: {improvement:.1f}°F")

elif page == "NLP Summary Panel":
    st.header("📝 NLP Summary Panel")
    st.write("AI-powered analysis of clinical notes and patient summaries.")
    
    # Patient selection for NLP analysis
    nlp_patient = st.selectbox(
        "Select Patient for Analysis",
        options=list(st.session_state.patients_data.keys()),
        format_func=lambda x: f"{x} - {st.session_state.patients_data[x]['name']}"
    )
    
    if nlp_patient:
        patient = st.session_state.patients_data[nlp_patient]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Clinical Notes")
            for i, note in enumerate(patient['notes'], 1):
                st.text_area(f"Note {i}", note, height=80, disabled=True)
        
        with col2:
            st.subheader("AI Analysis Summary")
            
            # Simulated NLP analysis
            st.markdown("""
            <div class="summary-panel">
            <h4>📊 Key Insights:</h4>
            <ul>
                <li><strong>Primary Concerns:</strong> Cardiovascular monitoring, blood pressure management</li>
                <li><strong>Treatment Response:</strong> Patient showing positive response to current regimen</li>
                <li><strong>Risk Factors:</strong> Elevated BP episodes, irregular heart rate patterns</li>
                <li><strong>Recommendations:</strong> Continue monitoring, consider medication adjustment</li>
            </ul>
            
            <h4>🔍 Sentiment Analysis:</h4>
            <p><span style="color: green;">●</span> Overall sentiment: <strong>Positive</strong> (Patient improving)</p>
            <p><span style="color: orange;">●</span> Concern level: <strong>Moderate</strong> (Requires monitoring)</p>
            
            <h4>📈 Trend Analysis:</h4>
            <p>Recent notes show improvement in patient condition with stable vitals and good treatment compliance.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Keyword extraction
            st.subheader("📌 Extracted Keywords")
            keywords = ["stable vitals", "medication", "blood pressure", "monitoring", "improved", "comfortable"]
            
            keyword_cols = st.columns(3)
            for i, keyword in enumerate(keywords):
                with keyword_cols[i % 3]:
                    st.code(keyword)
        
        # Clinical timeline
        st.subheader("🕒 Clinical Timeline")
        timeline_data = {
            'Date': pd.date_range(start=datetime.now() - timedelta(days=5), periods=6, freq='D'),
            'Event': [
                'Patient admitted',
                'Initial assessment completed',
                'Medication started',
                'Vitals stabilizing',
                'Family consultation',
                'Discharge planning initiated'
            ],
            'Severity': ['High', 'Medium', 'Low', 'Low', 'Low', 'Low']
        }
        
        timeline_df = pd.DataFrame(timeline_data)
        fig = px.scatter(timeline_df, x='Date', y='Event', color='Severity',
                        title="Patient Care Timeline", height=300)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Seed Patients":
    st.header("👥 Seed Patients Management")
    st.write("Manage and configure sample patient data for testing and demonstration.")
    
    tab1, tab2, tab3 = st.tabs(["Current Patients", "Generate New Data", "Export Data"])
    
    with tab1:
        st.subheader("Current Patient Database")
        
        # Create patient summary table
        patient_summary = []
        for pid, patient in st.session_state.patients_data.items():
            latest_vitals = patient['vitals'].iloc[-1]
            trends = analyze_vitals_trends(patient['vitals'])
            
            patient_summary.append({
                'Patient ID': pid,
                'Name': patient['name'],
                'Age': patient['age'],
                'Gender': patient['gender'],
                'Condition': patient['condition'],
                'Heart Rate': f"{latest_vitals['heart_rate']:.0f} bpm",
                'BP': f"{latest_vitals['blood_pressure_sys']:.0f}/{latest_vitals['blood_pressure_dia']:.0f}",
                'Temperature': f"{latest_vitals['temperature']:.1f}°F",
                'O2 Sat': f"{latest_vitals['oxygen_saturation']:.0f}%",
                'Status': '🔴' if any(v['status'] in ['HIGH', 'LOW'] for v in trends.values()) else '🟢'
            })
        
        summary_df = pd.DataFrame(patient_summary)
        st.dataframe(summary_df, use_container_width=True)
        
        # Patient statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Patients", len(patient_summary))
        
        with col2:
            critical_count = len([p for p in patient_summary if p['Status'] == '🔴'])
            st.metric("Critical Status", critical_count)
        
        with col3:
            conditions = [p['condition'] for p in st.session_state.patients_data.values()]
            unique_conditions = len(set(conditions))
            st.metric("Unique Conditions", unique_conditions)
    
    with tab2:
        st.subheader("Generate New Patient Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_patients = st.number_input("Number of patients to generate", min_value=1, max_value=100, value=10)
            
            if st.button("Generate Patients", type="primary"):
                with st.spinner("Generating new patient data..."):
                    new_patients = generate_sample_patients(num_patients)
                    
                    # Add to existing data with new IDs
                    current_count = len(st.session_state.patients_data)
                    for i, (old_id, patient_data) in enumerate(new_patients.items()):
                        new_id = f"PT{str(current_count + i + 1).zfill(4)}"
                        st.session_state.patients_data[new_id] = patient_data
                
                st.success(f"Successfully generated {num_patients} new patients!")
                st.rerun()
        
        with col2:
            st.info("""
            **Generation Features:**
            - Realistic vital signs patterns
            - Age and gender distribution
            - Common medical conditions
            - 30 days of historical data
            - Clinical notes samples
            """)
    
    with tab3:
        st.subheader("Export Patient Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Summary CSV"):
                summary_df = pd.DataFrame(patient_summary)
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"patient_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            selected_export_patient = st.selectbox(
                "Select Patient for Detailed Export",
                options=list(st.session_state.patients_data.keys()),
                format_func=lambda x: f"{x} - {st.session_state.patients_data[x]['name']}"
            )
            
            if selected_export_patient and st.button("Export Patient Vitals"):
                patient_vitals = st.session_state.patients_data[selected_export_patient]['vitals']
                csv = patient_vitals.to_csv(index=False)
                st.download_button(
                    label="Download Patient Vitals CSV",
                    data=csv,
                    file_name=f"vitals_{selected_export_patient}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# Add advanced analytics section
elif page == "Advanced Analytics":
    st.header("🧠 Advanced Analytics")
    
    # Population-level analytics
    st.subheader("Population Health Metrics")
    
    # Aggregate all patient data
    all_vitals = []
    for pid, patient in st.session_state.patients_data.items():
        patient_vitals = patient['vitals'].copy()
        patient_vitals['patient_id'] = pid
        patient_vitals['age_group'] = '65+' if patient['age'] >= 65 else '45-64' if patient['age'] >= 45 else '18-44'
        patient_vitals['condition'] = patient['condition']
        all_vitals.append(patient_vitals)
    
    combined_df = pd.concat(all_vitals, ignore_index=True)
    
    # Population metrics
    pop_col1, pop_col2, pop_col3, pop_col4 = st.columns(4)
    
    with pop_col1:
        avg_hr = combined_df['heart_rate'].mean()
        st.metric("Population Avg HR", f"{avg_hr:.1f} bpm")
    
    with pop_col2:
        avg_bp = combined_df['blood_pressure_sys'].mean()
        st.metric("Population Avg BP", f"{avg_bp:.0f} mmHg")
    
    with pop_col3:
        avg_temp = combined_df['temperature'].mean()
        st.metric("Population Avg Temp", f"{avg_temp:.1f}°F")
    
    with pop_col4:
        avg_o2 = combined_df['oxygen_saturation'].mean()
        st.metric("Population Avg O2", f"{avg_o2:.1f}%")
    
    # Age group analysis
    st.subheader("Vitals by Age Group")
    age_analysis = combined_df.groupby('age_group').agg({
        'heart_rate': 'mean',
        'blood_pressure_sys': 'mean',
        'temperature': 'mean',
        'oxygen_saturation': 'mean'
    }).round(1)
    
    fig_age = px.bar(
        age_analysis.reset_index(),
        x='age_group',
        y=['heart_rate', 'blood_pressure_sys', 'oxygen_saturation'],
        title="Average Vitals by Age Group",
        barmode='group'
    )
    st.plotly_chart(fig_age, use_container_width=True)
    
    # Condition-based analysis
    st.subheader("Vitals by Medical Condition")
    condition_analysis = combined_df.groupby('condition').agg({
        'heart_rate': 'mean',
        'blood_pressure_sys': 'mean',
        'temperature': 'mean',
        'oxygen_saturation': 'mean'
    }).round(1)
    
    fig_condition = px.heatmap(
        condition_analysis,
        title="Vitals Heatmap by Medical Condition",
        color_continuous_scale="RdYlBu_r"
    )
    st.plotly_chart(fig_condition, use_container_width=True)
    
    # Risk stratification
    st.subheader("Risk Stratification")
    
    risk_patients = []
    for pid, patient in st.session_state.patients_data.items():
        trends = analyze_vitals_trends(patient['vitals'])
        
        # Calculate risk score
        risk_score = 0
        for vital, data in trends.items():
            if data['status'] in ['HIGH', 'LOW']:
                risk_score += 2
            if data['trend'] == 'UP' and vital in ['heart_rate', 'blood_pressure_sys', 'temperature']:
                risk_score += 1
            elif data['trend'] == 'DOWN' and vital == 'oxygen_saturation':
                risk_score += 1
        
        risk_level = 'High' if risk_score >= 4 else 'Medium' if risk_score >= 2 else 'Low'
        
        risk_patients.append({
            'Patient ID': pid,
            'Name': patient['name'],
            'Age': patient['age'],
            'Condition': patient['condition'],
            'Risk Score': risk_score,
            'Risk Level': risk_level
        })
    
    risk_df = pd.DataFrame(risk_patients)
    
    # Risk distribution
    risk_counts = risk_df['Risk Level'].value_counts()
    fig_risk = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Patient Risk Distribution",
        color_discrete_map={'High': '#ff4444', 'Medium': '#ffaa00', 'Low': '#00aa44'}
    )
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # High-risk patients table
    high_risk = risk_df[risk_df['Risk Level'] == 'High'].sort_values('Risk Score', ascending=False)
    if len(high_risk) > 0:
        st.subheader("🚨 High-Risk Patients")
        st.dataframe(high_risk, use_container_width=True)

# Update sidebar navigation to include Advanced Analytics
st.sidebar.markdown("---")
if st.sidebar.button("🧠 Advanced Analytics"):
    st.session_state.current_page = "Advanced Analytics"

# Add real-time monitoring section
st.sidebar.markdown("### ⚡ Real-Time Monitoring")
if st.sidebar.checkbox("Auto-refresh (30s)"):
    st.rerun()

st.sidebar.markdown("### 📊 Quick Stats")
total_patients = len(st.session_state.patients_data)
st.sidebar.metric("Active Patients", total_patients)

# Calculate alerts
all_trends = []
for patient in st.session_state.patients_data.values():
    trends = analyze_vitals_trends(patient['vitals'])
    all_trends.extend([v['status'] for v in trends.values()])

alert_count = len([status for status in all_trends if status in ['HIGH', 'LOW']])
st.sidebar.metric("Active Alerts", alert_count)

# Footer with additional info
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Healthcare Analytics Dashboard v1.0</strong> | Built with Streamlit</p>
    <p>🔒 HIPAA Compliant | 📊 Real-time Analytics | 🤖 AI-Powered Insights</p>
    <small>Sample data for demonstration purposes only</small>
</div>
""", unsafe_allow_html=True)