import streamlit as st
import matplotlib.pyplot as plt
import random, joblib, json, pandas as pd
from sklearn.preprocessing import LabelEncoder

dt_classifier    = joblib.load('dt_classifier_model.joblib')
linear_reg_model = joblib.load('linear_reg_model.joblib')
with open('exercises_kb.json')  as f: exercises_kb  = json.load(f)
with open('diet_plans_kb.json') as f: diet_plans_kb = json.load(f)

le_gender = LabelEncoder(); le_gender.fit(['Male', 'Female'])
gender_mapping = dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))
le_activity = LabelEncoder()
le_activity.fit(['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active'])
activity_level_mapping = dict(zip(le_activity.classes_, le_activity.transform(le_activity.classes_)))
le_goal = LabelEncoder(); le_goal.fit(['Weight Loss', 'Muscle Gain', 'Maintenance'])
goal_mapping = dict(zip(le_goal.classes_, le_goal.transform(le_goal.classes_)))
le_bmi_cat = LabelEncoder(); le_bmi_cat.fit(['Underweight', 'Normal', 'Overweight', 'Obese'])
bmi_category_mapping = dict(zip(le_bmi_cat.classes_, le_bmi_cat.transform(le_bmi_cat.classes_)))
le_fitness = LabelEncoder(); le_fitness.fit(['Beginner', 'Intermediate', 'Advanced'])
fitness_mapping = dict(zip(le_fitness.classes_, le_fitness.transform(le_fitness.classes_)))
le_sleep = LabelEncoder(); le_sleep.fit(['Poor', 'Average', 'Good'])
sleep_mapping = dict(zip(le_sleep.classes_, le_sleep.transform(le_sleep.classes_)))
le_stress = LabelEncoder(); le_stress.fit(['Low', 'Medium', 'High'])
stress_mapping = dict(zip(le_stress.classes_, le_stress.transform(le_stress.classes_)))

WEEK_DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def get_bmi_category(bmi):
    if bmi < 18.5:   return 'Underweight'
    elif bmi < 24.9: return 'Normal'
    elif bmi < 29.9: return 'Overweight'
    else:            return 'Obese'

def pick_meal(options, dtype):
    if dtype == 'Vegetarian':
        veg = [m for m in options if not any(
            w in m.lower() for w in ['chicken','beef','tuna','salmon','turkey',
                                     'fish','shrimp','pork','lamb','bacon','steak'])]
        return random.choice(veg) if veg else random.choice(options)
    elif dtype == 'Vegan':
        vegan = [m for m in options if not any(
            w in m.lower() for w in ['chicken','beef','tuna','salmon','turkey','fish',
                                     'shrimp','pork','lamb','bacon','steak','egg',
                                     'cheese','milk','yogurt','cream','butter','honey'])]
        return random.choice(vegan) if vegan else random.choice(options)
    else:
        return random.choice(options)

def generate_weekly_plan(recs):
    ud        = recs['user_details']
    pr        = recs['predictions']
    goal      = ud['goal']
    bmi_cat   = ud['bmi_category']
    diet_type = ud['diet_type']
    intensity = pr['workout_intensity']
    wpw       = ud['workouts_per_week']

    goal_key     = goal    if goal    in diet_plans_kb           else 'Maintenance'
    bmi_key      = bmi_cat if bmi_cat in diet_plans_kb[goal_key] else 'Normal'
    diet_section = diet_plans_kb[goal_key][bmi_key]

    if wpw <= 2:   rest_days = ['Wednesday','Thursday','Friday','Saturday','Sunday']
    elif wpw == 3: rest_days = ['Tuesday','Thursday','Saturday','Sunday']
    elif wpw == 4: rest_days = ['Wednesday','Saturday','Sunday']
    elif wpw == 5: rest_days = ['Wednesday','Sunday']
    elif wpw == 6: rest_days = ['Sunday']
    else:          rest_days = []

    weekly_plan = {}
    for day in WEEK_DAYS:
        is_rest = day in rest_days
        if is_rest:
            exercise = {
                'type': 'Rest Day 🛌',
                'cardio':   ['Light walking (20-30 min)', 'Gentle stretching (15 min)'],
                'strength': ['Full body stretch', 'Foam rolling']
            }
        else:
            exercise = {
                'type': f'Workout Day 💪 ({intensity})',
                'cardio':   random.sample(exercises_kb['Cardio'][intensity],
                                          min(2, len(exercises_kb['Cardio'][intensity]))),
                'strength': random.sample(exercises_kb['Strength'][intensity],
                                          min(2, len(exercises_kb['Strength'][intensity])))
            }
        weekly_plan[day] = {
            'exercise': exercise,
            'diet': {
                'Morning':   pick_meal(diet_section['Morning'],   diet_type),
                'Afternoon': pick_meal(diet_section['Afternoon'], diet_type),
                'Night':     pick_meal(diet_section['Night'],     diet_type),
                'Snacks':    pick_meal(diet_section['Snacks'],    diet_type)
            }
        }
    return weekly_plan

def recommendation_engine(age, height, weight, gender, activity_level, goal,
                           fitness_level, diet_type, sleep_quality, stress_level,
                           water_intake, medical_condition, workouts_per_week, workout_duration):
    height_m     = height / 100
    bmi          = weight / (height_m ** 2)
    bmi_category = get_bmi_category(bmi)
    gender_encoded   = gender_mapping.get(gender)
    activity_encoded = activity_level_mapping.get(activity_level)
    goal_encoded     = goal_mapping.get(goal)
    bmi_cat_encoded  = bmi_category_mapping.get(bmi_category)
    fitness_encoded  = fitness_mapping.get(fitness_level)
    sleep_encoded    = sleep_mapping.get(sleep_quality)
    stress_encoded   = stress_mapping.get(stress_level)
    if None in [gender_encoded, activity_encoded, goal_encoded,
                bmi_cat_encoded, fitness_encoded, sleep_encoded, stress_encoded]:
        return {"error": "Invalid input."}
    dt_features = pd.DataFrame(
        [[age, gender_encoded, activity_encoded, goal_encoded,
          bmi_cat_encoded, fitness_encoded, sleep_encoded, stress_encoded, workouts_per_week]],
        columns=['Age','Gender_Encoded','ActivityLevel_Encoded','Goal_Encoded',
                 'BMICategory_Encoded','FitnessLevel_Encoded',
                 'SleepQuality_Encoded','StressLevel_Encoded','WorkoutsPerWeek'])
    workout_intensity = dt_classifier.predict(dt_features)[0]
    if workout_intensity not in exercises_kb['Cardio']: workout_intensity = 'Medium'
    lr_features = pd.DataFrame(
        [[age, weight, bmi, activity_encoded, goal_encoded,
          sleep_encoded, stress_encoded, workouts_per_week, workout_duration]],
        columns=['Age','Weight','BMI','ActivityLevel_Encoded','Goal_Encoded',
                 'SleepQuality_Encoded','StressLevel_Encoded',
                 'WorkoutsPerWeek','WorkoutDuration_min'])
    target_weight_change = linear_reg_model.predict(lr_features)[0]

    warnings = []
    if medical_condition == 'Diabetes':
        warnings.append("⚠️ Diabetic: Avoid high-sugar foods. Monitor carb intake.")
    if medical_condition == 'Hypertension':
        warnings.append("⚠️ Hypertension: Avoid high-sodium foods. Limit caffeine.")
    if medical_condition == 'Heart Condition':
        warnings.append("⚠️ Heart Condition: Avoid high-intensity workouts. Consult doctor.")
    if sleep_quality == 'Poor':
        warnings.append("😴 Poor sleep: Prioritize 7-8 hrs sleep for better results.")
    if stress_level == 'High':
        warnings.append("🧘 High stress: Add yoga or meditation to your routine.")
    if water_intake < 2.0:
        warnings.append("💧 Low water intake: Aim for at least 2.5L per day.")

    return {
        'user_details': {
            'age': age, 'height': height, 'weight': weight, 'gender': gender,
            'activity_level': activity_level, 'goal': goal,
            'bmi': round(bmi, 2), 'bmi_category': bmi_category,
            'fitness_level': fitness_level, 'diet_type': diet_type,
            'sleep_quality': sleep_quality, 'stress_level': stress_level,
            'water_intake': water_intake, 'medical_condition': medical_condition,
            'workouts_per_week': workouts_per_week, 'workout_duration': workout_duration
        },
        'predictions': {
            'workout_intensity': workout_intensity,
            'predicted_monthly_weight_change': round(target_weight_change, 2)
        },
        'warnings': warnings
    }

# ── Streamlit UI ──
st.set_page_config(page_title="UVFIT", layout="wide", page_icon="🏋️")
st.title("🏋️ UVFIT: Personalized Fitness & Nutrition Recommender")

st.sidebar.header("📋 Your Profile")
age            = st.sidebar.slider("Age", 18, 65, 30)
height         = st.sidebar.slider("Height (cm)", 150, 200, 170)
weight         = st.sidebar.slider("Weight (kg)", 40, 150, 70)
gender         = st.sidebar.selectbox("Gender", ['Male', 'Female'])
activity_level = st.sidebar.selectbox("Activity Level",
                     ['Sedentary','Lightly Active','Moderately Active','Very Active'])
goal           = st.sidebar.selectbox("Goal", ['Weight Loss','Muscle Gain','Maintenance'])

st.sidebar.header("🏃 Fitness Details")
fitness_level     = st.sidebar.selectbox("Fitness Level", ['Beginner','Intermediate','Advanced'])
workouts_per_week = st.sidebar.slider("Workouts Per Week", 0, 7, 3)
workout_duration  = st.sidebar.slider("Workout Duration (min)", 15, 90, 45)

st.sidebar.header("🥗 Lifestyle")
diet_type     = st.sidebar.selectbox("Diet Type", ['Non-Vegetarian','Vegetarian','Vegan'])
water_intake  = st.sidebar.slider("Water Intake (L/day)", 1.0, 4.0, 2.5, step=0.1)
sleep_quality = st.sidebar.selectbox("Sleep Quality", ['Poor','Average','Good'])
stress_level  = st.sidebar.selectbox("Stress Level", ['Low','Medium','High'])

st.sidebar.header("🏥 Medical")
medical_condition = st.sidebar.selectbox("Medical Condition",
                        ['None','Diabetes','Hypertension','Heart Condition'])

# ── Session state ──
current_inputs = (age, height, weight, gender, activity_level, goal,
                  fitness_level, diet_type, sleep_quality, stress_level,
                  water_intake, medical_condition, workouts_per_week, workout_duration)
if 'last_inputs' not in st.session_state:
    st.session_state.last_inputs = None
    st.session_state.last_result = None
    st.session_state.weekly_plan = None

if st.sidebar.button("🚀 Generate My Plan"):
    if current_inputs != st.session_state.last_inputs:
        result = recommendation_engine(*current_inputs)
        if "error" not in result:
            st.session_state.last_result  = result
            st.session_state.weekly_plan  = generate_weekly_plan(result)
            st.session_state.last_inputs  = current_inputs

# ── Display ──
if st.session_state.get('last_result'):
    recs        = st.session_state.last_result
    weekly_plan = st.session_state.weekly_plan
    ud = recs['user_details']
    pr = recs['predictions']
    ws = recs['warnings']

    st.header("📊 Your Personalized Fitness Report")

    # Warnings
    if ws:
        st.subheader("⚠️ Health Warnings")
        for w in ws: st.warning(w)

    # Profile row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("👤 Profile")
        st.write(f"**Age:** {ud['age']} yrs | **Gender:** {ud['gender']}")
        st.write(f"**Height:** {ud['height']} cm | **Weight:** {ud['weight']} kg")
        st.write(f"**Goal:** {ud['goal']} | **Activity:** {ud['activity_level']}")
        st.write(f"**Fitness Level:** {ud['fitness_level']}")
    with col2:
        st.subheader("📐 Body & Lifestyle")
        st.write(f"**BMI:** {ud['bmi']} ({ud['bmi_category']})")
        st.write(f"**Diet Type:** {ud['diet_type']}")
        st.write(f"**Sleep:** {ud['sleep_quality']} | **Stress:** {ud['stress_level']}")
        st.write(f"**Water:** {ud['water_intake']} L/day")
        st.write(f"**Medical:** {ud['medical_condition']}")
    with col3:
        st.subheader("🔮 Predictions")
        st.write(f"**Workout Intensity:** {pr['workout_intensity']}")
        change = pr['predicted_monthly_weight_change']
        arrow  = "📉" if change < 0 else "📈"
        st.write(f"**Monthly Change:** {arrow} {change:+.2f} kg/month")
        st.write(f"**Workouts/Week:** {ud['workouts_per_week']}x")
        st.write(f"**Session Duration:** {ud['workout_duration']} min")

    st.divider()

    # ── 7-Day Plan ──
    st.header("📅 Your 7-Day Fitness & Diet Plan")
    day_colors = {
        'Monday': '🔵', 'Tuesday': '🟢', 'Wednesday': '🟡',
        'Thursday': '🟠', 'Friday': '🔴', 'Saturday': '🟣', 'Sunday': '⚪'
    }
    for day, plan in weekly_plan.items():
        is_rest = 'Rest' in plan['exercise']['type']
        with st.expander(f"{day_colors[day]} {day}  —  {plan['exercise']['type']}", expanded=False):
            col_ex, col_diet = st.columns(2)
            with col_ex:
                st.markdown("**🏃 Exercise**")
                st.markdown("*Cardio:*")
                for ex in plan['exercise']['cardio']:
                    st.markdown(f"- {ex}")
                st.markdown("*Strength:*")
                for ex in plan['exercise']['strength']:
                    st.markdown(f"- {ex}")
            with col_diet:
                st.markdown("**🍽️ Meals**")
                st.markdown(f"🌅 **Morning:** {plan['diet']['Morning']}")
                st.markdown(f"☀️ **Afternoon:** {plan['diet']['Afternoon']}")
                st.markdown(f"🌙 **Night:** {plan['diet']['Night']}")
                st.markdown(f"🍎 **Snack:** {plan['diet']['Snacks']}")

    st.divider()

    # Weight chart
    st.subheader("📈 Predicted Weight Progress")
    cw     = ud['weight']
    change = pr['predicted_monthly_weight_change']
    months = ['Now'] + [f'Month {i+1}' for i in range(6)]
    ws_vals = [cw]
    for _ in range(6): cw += change; ws_vals.append(round(cw, 2))
    color = '#4CAF50' if ud['goal'] == 'Weight Loss' else '#2196F3' if ud['goal'] == 'Muscle Gain' else '#FF9800'
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(months, ws_vals, color=color, alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, ws_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{val}kg', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xlabel('Timeline', fontsize=11)
    ax.set_ylabel('Weight (kg)', fontsize=11)
    ax.set_title(f'6-Month Weight Forecast — {ud["goal"]}', fontsize=13, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("👈 Fill in your profile in the sidebar and click **Generate My Plan**!")
