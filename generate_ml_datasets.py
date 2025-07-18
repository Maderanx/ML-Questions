import numpy as np
import pandas as pd
from faker import Faker
import random

fake = Faker()
np.random.seed(42)
random.seed(42)

def random_n():
    return random.randint(5000, 8000)

# 1. Tirupati Queue Wait Time Prediction (Regression)
def generate_tirupati_queue():
    N = random_n()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    darshan_types = ['Free', '₹300', 'VIP', 'Senior']
    weather = ['Sunny', 'Cloudy', 'Rainy']
    gates = ['Q-Complex I', 'Q-Complex II', 'Vaikuntam', 'Special Entry']
    data = []
    for _ in range(N):
        day = random.choice(days)
        is_holiday = np.random.binomial(1, 0.15)
        festival = np.random.binomial(1, 0.10)
        darshan = random.choice(darshan_types)
        weather_cond = random.choices(weather, weights=[0.6,0.3,0.1])[0]
        temp = np.random.normal(28, 4) + (3 if weather_cond=='Sunny' else -2 if weather_cond=='Rainy' else 0)
        buses = max(0, np.random.poisson(8 + 5*is_holiday + 10*festival))
        gate = random.choice(gates)
        vip = np.random.binomial(1, 0.05)
        sec_lag = max(0, np.random.normal(7, 2) + 3*festival)
        headcount = int(max(0, np.random.normal(3000, 800) + 2000*festival + 1000*is_holiday))
        is_weekend = 1 if day in ['Sat','Sun'] else 0
        online_peak = np.random.binomial(1, 0.2)
        wait = (buses*2 + headcount/100 + 10*festival + 8*is_holiday + 15*vip + sec_lag + 10*online_peak + np.random.normal(0,5))
        data.append([day, is_holiday, festival, darshan, weather_cond, temp, buses, gate, vip, sec_lag, headcount, is_weekend, online_peak, wait])
    columns = ['Day_Of_Week','Is_Public_Holiday','Festival_Flag','Darshan_Type','Weather_Condition','Temperature_C','Buses_Arrived_Last_15_Min','Entry_Gate','VIP_Visit_Today','Security_Check_Lag_Minutes','Approx_Head_Count','Is_Weekend','Online_Booking_Peak_Flag','Estimated_Wait_Time_Minutes']
    df = pd.DataFrame(data, columns=columns)
    # Inject missing values only (no negative outliers for impossible columns)
    for col in ['Temperature_C','Security_Check_Lag_Minutes','Buses_Arrived_Last_15_Min','Approx_Head_Count']:
        if col in df:
            idx = np.random.choice(df.index, size=int(0.01*N), replace=False)
            df.loc[idx, col] = np.nan
    for col in ['Day_Of_Week','Darshan_Type','Weather_Condition','Entry_Gate']:
        if col in df:
            idx = np.random.choice(df.index, size=int(0.01*N), replace=False)
            df.loc[idx, col] = np.nan
    print(f"Tirupati Queue: {len(df)} rows")
    return df

# 2. GPay Monthly Balance Drain Prediction (Regression)
def generate_gpay_balance():
    N = random_n()
    categories = ['Food', 'Shopping', 'Travel', 'Bills', 'Groceries', 'Entertainment', 'Other']
    data = []
    for _ in range(N):
        # Only allow negative outliers in Starting_Balance (rare, overdraft)
        start_bal = int(np.random.uniform(500, 20000))
        if random.random() < 0.01:
            start_bal = -int(np.random.uniform(1, 5000))
        income = int(np.random.uniform(1000, 50000))
        avg_spend = int(np.random.uniform(100, 2000))
        cashback = int(np.random.choice([0, np.random.uniform(10, 500)], p=[0.7, 0.3]))
        bill_pays = int(np.random.poisson(2))
        num_txn = int(np.random.poisson(20))
        high_value = int(np.random.rand() < 0.25)
        top3 = random.sample(categories, 3)
        refills = int(np.random.poisson(2))
        has_credit = np.random.binomial(1, 0.2)
        has_limit = np.random.binomial(1, 0.3)
        avg_per_cat = {cat: int(np.random.uniform(0, avg_spend)) for cat in categories}
        drain_rate = avg_spend * (1.1 if high_value else 1) + 0.5*bill_pays + 0.2*num_txn
        days_until_min = int((start_bal + income + cashback - 0.5*sum(avg_per_cat.values())) / (drain_rate+1))
        days_until_min = max(0, min(days_until_min, 30))
        data.append([start_bal, income, avg_spend, cashback, bill_pays, num_txn, high_value, ','.join(top3), refills, has_credit, has_limit, str(avg_per_cat), days_until_min])
    columns = ['Starting_Balance','Monthly_Income','Avg_Daily_Spend','Cashback_Received','Bill_Payments_This_Month','Num_Transactions','High_Value_Spend_Flag','Top_3_Spend_Categories','Wallet_Refills','Has_Credit_Linked','Has_Spend_Limit_Set','Avg_Spend_Per_Category','Days_Until_Min_Balance']
    df = pd.DataFrame(data, columns=columns)
    # Inject missing values only (no negative outliers except Starting_Balance)
    for col in ['Starting_Balance','Monthly_Income','Avg_Daily_Spend','Cashback_Received','Bill_Payments_This_Month','Num_Transactions','Wallet_Refills']:
        if col in df:
            idx = np.random.choice(df.index, size=int(0.01*N), replace=False)
            df.loc[idx, col] = np.nan
    for col in ['Top_3_Spend_Categories']:
        if col in df:
            idx = np.random.choice(df.index, size=int(0.01*N), replace=False)
            df.loc[idx, col] = np.nan
    print(f"GPay Balance: {len(df)} rows")
    return df

# 3. Minimum Study Hours for Highest Grade (Regression)
def generate_study_hours():
    N = random_n()
    grades = ['A','B','C','D']
    study_modes = ['Group','Solo','Online']
    data = []
    for _ in range(N):
        diff = np.random.choice([1,2,3], p=[0.4,0.4,0.2])
        prev_grade = random.choice(grades)
        attendance = np.clip(np.random.normal(90, 7), 60, 100)
        avg_study = np.clip(np.random.normal(2.5, 1), 0.5, 8)
        screen_time = np.clip(np.random.normal(4, 1.5), 1, 10)
        coaching = np.random.binomial(1, 0.35)
        assign_rate = np.clip(np.random.normal(0.85, 0.1), 0.5, 1)
        mode = random.choice(study_modes)
        n_books = np.random.poisson(2)
        sleep = np.clip(np.random.normal(6.5, 1), 3, 10)
        stress = np.random.randint(1,6)
        base = 2 + 0.8*diff + 0.5*(grades.index('A')-grades.index(prev_grade)) - 0.5*coaching - 0.2*assign_rate + 0.1*screen_time - 0.2*sleep + 0.1*stress
        req_hours = np.clip(base + np.random.normal(0,0.5), 1, 10)
        data.append([diff, prev_grade, attendance, avg_study, screen_time, coaching, assign_rate, mode, n_books, sleep, stress, req_hours])
    columns = ['Subject_Difficulty_Level','Previous_Grade','Attendance_Rate','Avg_Study_Hours_Last_Sem','Screen_Time_Hours','Coaching_Enrolled','Assignment_Completion_Rate','Preferred_Study_Mode','Number_of_Reference_Books_Used','Sleep_Hours','Stress_Self_Rating','Required_Study_Hours']
    df = pd.DataFrame(data, columns=columns)
    # Inject missing values only (no negative outliers for impossible columns)
    for col in ['Attendance_Rate','Avg_Study_Hours_Last_Sem','Screen_Time_Hours','Assignment_Completion_Rate','Sleep_Hours']:
        if col in df:
            idx = np.random.choice(df.index, size=int(0.01*N), replace=False)
            df.loc[idx, col] = np.nan
    for col in ['Previous_Grade','Preferred_Study_Mode']:
        if col in df:
            idx = np.random.choice(df.index, size=int(0.01*N), replace=False)
            df.loc[idx, col] = np.nan
    print(f"Study Hours: {len(df)} rows")
    return df

# 4. Patient Appointment No-Show Classification
def generate_patient_noshow():
    N = random_n()
    genders = ['M','F','O']
    appt_types = ['Routine','Follow-up','Emergency','Screening']
    transport = ['Self','Public','Ambulance']
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    data = []
    for _ in range(N):
        age = np.random.randint(0, 90)
        days_before = np.random.randint(0, 30)
        gender = random.choice(genders)
        appt_day = random.choice(days)
        sched_day = random.choice(days)
        reminder = np.random.binomial(1, 0.7)
        chronic = np.random.binomial(1, 0.2)
        past_noshow = int(np.random.poisson(0.5))
        appt_type = random.choice(appt_types)
        trans = random.choice(transport)
        rain = np.random.binomial(1, 0.1)
        dist = abs(np.random.normal(7, 5))
        prob_no_show = 0.15 + 0.1*chronic + 0.1*(past_noshow>0) + 0.1*(reminder==0) + 0.05*(dist>15) + 0.05*rain
        y = np.random.choice(['Show','No Show'], p=[1-prob_no_show, prob_no_show])
        data.append([age, gender, appt_day, sched_day, days_before, reminder, chronic, past_noshow, appt_type, trans, rain, dist, y])
    columns = ['Patient_Age','Gender','Appointment_Day','Scheduled_Day','Days_Before_Appointment','Reminder_Sent','Has_Chronic_Condition','Past_No_Show_Count','Appointment_Type','Transportation_Mode','Rain_On_That_Day','Distance_From_Hospital_KM','Target']
    df = pd.DataFrame(data, columns=columns)
    # Inject missing values only (no negative outliers for impossible columns)
    for col in ['Patient_Age','Days_Before_Appointment','Past_No_Show_Count','Distance_From_Hospital_KM']:
        if col in df:
            idx = np.random.choice(df.index, size=int(0.01*N), replace=False)
            df.loc[idx, col] = np.nan
    for col in ['Gender','Appointment_Type','Transportation_Mode','Appointment_Day']:
        if col in df:
            idx = np.random.choice(df.index, size=int(0.01*N), replace=False)
            df.loc[idx, col] = np.nan
    print(f"Patient No-Show: {len(df)} rows")
    return df

# 5. Nobel Prize Win Classification
def generate_nobel_prize():
    N = random_n()
    fields = ['Physics','Chemistry','Medicine','Literature','Peace','Economics']
    data = []
    for _ in range(N):
        pubs = int(np.random.poisson(40))
        cites = int(np.random.normal(2000, 1500))
        hidx = int(np.clip(np.random.normal(25, 10), 5, 80))
        field = random.choice(fields)
        inst_rank = np.random.randint(1, 201)
        avg_auth = np.clip(np.random.normal(4, 2), 1, 20)
        major_award = np.random.binomial(1, 0.15)
        top_journals = int(np.random.poisson(3))
        intl_collab = int(np.random.poisson(2))
        yrs_since_phd = int(np.random.randint(1, 50))
        coauthor_win = int(np.random.poisson(1))
        prob_win = 0.01 + 0.02*(inst_rank<=10) + 0.03*(major_award) + 0.01*(top_journals>5) + 0.01*(coauthor_win>0) + 0.01*(cites>5000)
        y = np.random.choice(['No Win','Win'], p=[1-prob_win, prob_win])
        data.append([pubs, max(0, cites), hidx, field, inst_rank, avg_auth, major_award, top_journals, intl_collab, yrs_since_phd, coauthor_win, y])
    columns = ['Total_Publications','Citations','h_Index','Field_Of_Study','Institution_Ranking','Avg_Authors_Per_Paper','Has_Won_Major_Award','Papers_In_Top_Journals','International_Collaborations','Years_Since_PhD','CoAuthor_Connections_With_Winners','Target']
    df = pd.DataFrame(data, columns=columns)
    # Inject missing values only (no negative outliers for impossible columns)
    for col in ['Total_Publications','Citations','h_Index','Avg_Authors_Per_Paper','Papers_In_Top_Journals','International_Collaborations','Years_Since_PhD','CoAuthor_Connections_With_Winners']:
        if col in df:
            idx = np.random.choice(df.index, size=int(0.01*N), replace=False)
            df.loc[idx, col] = np.nan
    for col in ['Field_Of_Study']:
        if col in df:
            idx = np.random.choice(df.index, size=int(0.01*N), replace=False)
            df.loc[idx, col] = np.nan
    print(f"Nobel Prize: {len(df)} rows")
    return df

# 6. Sports Player Selection Classification
def generate_sports_selection():
    N = random_n()
    roles = ['Forward','Midfielder','Defender','Goalkeeper']
    recov = ['Full','Partial','Rehab']
    match_imp = ['Low','Medium','High']
    data = []
    for _ in range(N):
        games = int(np.random.randint(0, 6))
        perf = float(np.clip(np.random.normal(6.5, 2), 1, 10))
        injury = int(np.random.poisson(0.5))
        rec = random.choice(recov)
        disc = int(np.random.poisson(0.2))
        role = random.choice(roles)
        match = random.choice(match_imp)
        fitness = float(np.clip(np.random.normal(7, 1.5), 3, 10))
        fatigue = float(np.clip(np.random.normal(3, 1.5), 0, 10))
        coach_pref = np.random.binomial(1, 0.2)
        team_need = np.random.binomial(1, 0.5)
        prob_sel = 0.2 + 0.2*(fitness>8) + 0.1*(coach_pref) + 0.1*(team_need) - 0.1*(injury>0) - 0.1*(rec=='Rehab') - 0.1*(disc>0) - 0.1*(fatigue>7)
        prob_sel = np.clip(prob_sel, 0, 1)
        y = np.random.choice(['Not Selected','Selected'], p=[1-prob_sel, prob_sel])
        data.append([games, perf, injury, rec, disc, role, match, fitness, fatigue, coach_pref, team_need, y])
    columns = ['Games_Played_Last_5','Avg_Performance_Rating','Injury_Count_Last_Season','Recovery_Status','Disciplinary_Actions','Player_Role','Match_Importance','Fitness_Score','Travel_Fatigue_Index','Coach_Preference_Flag','Team_Need_For_Role','Target']
    df = pd.DataFrame(data, columns=columns)
    # Inject missing values only (no negative outliers for impossible columns)
    for col in ['Games_Played_Last_5','Avg_Performance_Rating','Injury_Count_Last_Season','Disciplinary_Actions','Fitness_Score','Travel_Fatigue_Index']:
        if col in df:
            idx = np.random.choice(df.index, size=int(0.01*N), replace=False)
            df.loc[idx, col] = np.nan
    for col in ['Player_Role','Recovery_Status','Match_Importance']:
        if col in df:
            idx = np.random.choice(df.index, size=int(0.01*N), replace=False)
            df.loc[idx, col] = np.nan
    print(f"Sports Selection: {len(df)} rows")
    return df

# 7. Exoplanet Habitability Classification
def generate_exoplanet_habitability():
    N = random_n()
    star_types = ['M','K','G','F','A']
    atmos = ['CO2','O2','N2','H2O','CH4','He','Ar']
    data = []
    for _ in range(N):
        mass = abs(np.random.normal(1, 0.7))
        radius = abs(np.random.normal(1, 0.4))
        temp = np.random.normal(280, 40)
        dist = abs(np.random.normal(1, 0.5))
        star = random.choice(star_types)
        period = abs(np.random.normal(365, 100))
        atm = ','.join(random.sample(atmos, np.random.randint(1,4)))
        ecc = np.clip(np.random.beta(2,5), 0, 1)
        water = np.random.binomial(1, 0.2)
        flux = abs(np.random.normal(1, 0.5))
        tidal = np.random.binomial(1, 0.3)
        hab = (0.5<mass<5) and (0.5<radius<2.5) and (230<temp<350) and (0.5<dist<2) and ('O2' in atm or 'H2O' in atm) and (water==1) and (flux<2)
        y = 'Habitable' if hab else 'Non-Habitable'
        data.append([mass, radius, temp, dist, star, period, atm, ecc, water, flux, tidal, y])
    columns = ['Mass_Earth_Relative','Radius_Earth_Relative','Surface_Temperature_K','Star_Distance_AU','Star_Type','Orbital_Period_Days','Atmosphere_Composition','Eccentricity','Water_Vapor_Detected','Receives_Stellar_Flux','Tidal_Locking_Potential','Target']
    df = pd.DataFrame(data, columns=columns)
    # Inject missing values only (no negative outliers for impossible columns)
    for col in ['Mass_Earth_Relative','Radius_Earth_Relative','Surface_Temperature_K','Star_Distance_AU','Orbital_Period_Days','Eccentricity','Receives_Stellar_Flux']:
        if col in df:
            idx = np.random.choice(df.index, size=int(0.01*N), replace=False)
            df.loc[idx, col] = np.nan
    for col in ['Star_Type','Atmosphere_Composition']:
        if col in df:
            idx = np.random.choice(df.index, size=int(0.01*N), replace=False)
            df.loc[idx, col] = np.nan
    print(f"Exoplanet Habitability: {len(df)} rows")
    return df

if __name__ == '__main__':
    print('Generating datasets...')
    generate_tirupati_queue().to_csv('ML/ML/datasets/tirupati_queue.csv', index=False)
    generate_gpay_balance().to_csv('ML/ML/datasets/gpay_balance.csv', index=False)
    generate_study_hours().to_csv('ML/ML/datasets/study_hours.csv', index=False)
    generate_patient_noshow().to_csv('ML/ML/datasets/patient_noshow.csv', index=False)
    generate_nobel_prize().to_csv('ML/ML/datasets/nobel_prize.csv', index=False)
    generate_sports_selection().to_csv('ML/ML/datasets/sports_selection.csv', index=False)
    generate_exoplanet_habitability().to_csv('ML/ML/datasets/exoplanet_habitability.csv', index=False)
    print('All datasets generated in ML/ML/datasets/') 