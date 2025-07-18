# Machine Learning Problem Statements, Dataset Schemas & Links

## REGRESSION PROBLEMS

### 1. Tirupati Temple Queue Estimation Without Historical Data
Every day, thousands of pilgrims journey to the sacred Tirumala Tirupati Temple, often unsure how long they’ll need to wait in the ever-changing darshan queues. With sudden influxes due to festivals, weather shifts, or VIP visits, accurate planning becomes difficult.
Imagine an on-ground system that could predict how long a pilgrim will have to wait right now — just by observing the current crowd conditions, bus arrivals, and queue behavior. No need for historic time-series logs — just snapshot data from sensors and staff observations.
- **Dataset:** [tirupati_queue.csv](./tirupati_queue.csv)
- **Target Variable:** `Estimated_Wait_Time_Minutes`
- **Schema:**
    - `Day_Of_Week` (Categorical): Weekday name (Mon–Sun)
    - `Is_Public_Holiday` (Binary): 1 if public holiday
    - `Festival_Flag` (Binary): 1 if a religious festival is ongoing
    - `Darshan_Type` (Categorical): {'Free', '₹300', 'VIP', 'Senior'}
    - `Weather_Condition` (Categorical): {'Sunny', 'Cloudy', 'Rainy'}
    - `Temperature_C` (Numeric): Ambient temperature
    - `Buses_Arrived_Last_15_Min` (Integer): Total number of buses at entry point
    - `Entry_Gate` (Categorical): {'Q-Complex I', 'Q-Complex II', etc.}
    - `VIP_Visit_Today` (Binary): 1 if VIP is scheduled today
    - `Security_Check_Lag_Minutes` (Float): Avg. time at security checkpoint
    - `Approx_Head_Count` (Integer): Real-time headcount estimate via sensors
    - `Is_Weekend` (Binary): 1 if Sat or Sun
    - `Online_Booking_Peak_Flag` (Binary): 1 if slots are fully booked online
    - `Estimated_Wait_Time_Minutes` (Float): **Target**

### 2. How Long Until I’m Broke on GPay
Meet Aarti. She starts the month with ₹3000 in her GPay wallet. Between spontaneous Swiggy cravings and UPI splits with friends, her balance tends to vanish well before payday. She wonders: “When exactly will I run out this month?”
By analyzing her past transaction habits, cashback behavior, and spending categories, we can predict the number of days until her balance drops below ₹500 — helping her plan better and avoid last-minute top-ups.
- **Dataset:** [gpay_balance.csv](./gpay_balance.csv)
- **Target Variable:** `Days_Until_Min_Balance`
- **Schema:**
    - `Starting_Balance` (Integer): Wallet balance at month's start
    - `Monthly_Income` (Integer): Monthly wallet top-up/income
    - `Avg_Daily_Spend` (Integer): Mean daily expense over last 3 months
    - `Cashback_Received` (Integer): Cashback amount for current month
    - `Bill_Payments_This_Month` (Integer): Number of recurring payments made
    - `Num_Transactions` (Integer): Total UPI spends this month
    - `High_Value_Spend_Flag` (Binary): 1 if any spend > ₹1000
    - `Top_3_Spend_Categories` (Multi-categorical): e.g., Food, Shopping, Travel
    - `Wallet_Refills` (Integer): Number of times wallet was recharged
    - `Has_Credit_Linked` (Binary): 1 if linked to credit line
    - `Has_Spend_Limit_Set` (Binary): 1 if user has a set budget limit
    - `Avg_Spend_Per_Category` (Dict-like): Avg. amount in each category
    - `Days_Until_Min_Balance` (Integer): **Target**

### 3. The Smart Study Hour Advisor
Raj is aiming for an A+ in Data Structures, but he’s unsure how much he really needs to study. He doesn't want to overwork, but he also can’t afford to underprepare. If only he knew the minimum effort needed to succeed based on his learning style and past performance.
With data on Raj’s academic history, attendance, and study habits, we can build a model to predict the ideal study hours required to achieve a top grade — optimizing effort and maximizing results.
- **Dataset:** [study_hours.csv](./study_hours.csv)
- **Target Variable:** `Required_Study_Hours`
- **Schema:**
    - `Subject_Difficulty_Level` (Ordinal): {1: Easy, 2: Medium, 3: Hard}
    - `Previous_Grade` (Categorical): {‘A’, ‘B’, ‘C’, ‘D’}
    - `Attendance_Rate` (Float): % attendance in the semester
    - `Avg_Study_Hours_Last_Sem` (Float): Average daily study hours
    - `Screen_Time_Hours` (Float): Daily average screen time (mobile/laptop)
    - `Coaching_Enrolled` (Binary): 1 if student has external help
    - `Assignment_Completion_Rate` (Float): % of assignments submitted
    - `Preferred_Study_Mode` (Categorical): {‘Group’, ‘Solo’, ‘Online’}
    - `Number_of_Reference_Books_Used` (Integer): Total resources used
    - `Sleep_Hours` (Float): Avg. daily sleep hours
    - `Stress_Self_Rating` (Ordinal): {1–5} from self-reported survey
    - `Required_Study_Hours` (Float): **Target**

## CLASSIFICATION PROBLEMS

### 1. Will the Patient Show Up or Ghost the Clinic
Hospitals waste precious slots daily due to patient no-shows. Sahana, a hospital administrator, wants to reduce this chaos. She wonders if it’s possible to predict ahead of time which patients are likely to skip appointments.
By analyzing demographic data, prior attendance behavior, and scheduling patterns, we can classify patients into likely Show or No Show, enabling hospitals to send reminders or open slots to waitlisted patients.
- **Dataset:** [patient_noshow.csv](./patient_noshow.csv)
- **Target Variable:** `Target` (`Show` or `No Show`)
- **Schema:**
    - `Patient_Age` (Integer): Age of patient
    - `Gender` (Categorical): Gender
    - `Appointment_Day` (Categorical): Day of appointment
    - `Scheduled_Day` (Categorical): Day appointment was scheduled
    - `Days_Before_Appointment` (Integer): Days between scheduling and appointment
    - `Reminder_Sent` (Binary): 1 if reminder sent
    - `Has_Chronic_Condition` (Binary): 1 if patient has chronic condition
    - `Past_No_Show_Count` (Integer): Number of previous no-shows
    - `Appointment_Type` (Categorical): Routine, Follow-up, etc.
    - `Transportation_Mode` (Categorical): Self, Public, Ambulance
    - `Rain_On_That_Day` (Binary): 1 if it rained
    - `Distance_From_Hospital_KM` (Float): Distance to hospital
    - `Target` (Categorical): **Target**

### 2. Who Might Win a Nobel Prize
Dr. Kannan is a brilliant scientist with a growing list of publications. But is he on a Nobel-worthy path? Using data on publication quality, citations, collaborations, and institutional reputation, we can train a model to spot researchers with the hallmarks of Nobel winners.
This classification could help funding bodies and governments recognize academic excellence before the spotlight does.
- **Dataset:** [nobel_prize.csv](./nobel_prize.csv)
- **Target Variable:** `Target` (`Win` or `No Win`)
- **Schema:**
    - `Total_Publications` (Integer): Number of publications
    - `Citations` (Integer): Total citations
    - `h_Index` (Integer): h-index
    - `Field_Of_Study` (Categorical): Field of research
    - `Institution_Ranking` (Ordinal): Institutional rank
    - `Avg_Authors_Per_Paper` (Float): Average authors per paper
    - `Has_Won_Major_Award` (Binary): 1 if won major award
    - `Papers_In_Top_Journals` (Integer): Number of papers in top journals
    - `International_Collaborations` (Integer): Number of international collaborations
    - `Years_Since_PhD` (Integer): Years since PhD
    - `CoAuthor_Connections_With_Winners` (Integer): Number of coauthors who are Nobel winners
    - `Target` (Categorical): **Target**

### 3. Will This Player Make the Team
In professional sports, selection decisions are critical. Coaches must weigh injury history, recent performance, and team needs — often under time pressure. Imagine an AI assistant that helps predict if a player will be selected for the next game.
Such a tool could assist team selectors or even help fantasy league players draft smarter teams.
- **Dataset:** [sports_selection.csv](./sports_selection.csv)
- **Target Variable:** `Target` (`Selected` or `Not Selected`)
- **Schema:**
    - `Games_Played_Last_5` (Integer): Number of games played in last 5
    - `Avg_Performance_Rating` (Float): Average performance rating
    - `Injury_Count_Last_Season` (Integer): Number of injuries last season
    - `Recovery_Status` (Categorical): Full, Partial, Rehab
    - `Disciplinary_Actions` (Integer): Number of disciplinary actions
    - `Player_Role` (Categorical): Player's role
    - `Match_Importance` (Categorical): Importance of match
    - `Fitness_Score` (Float): Fitness score
    - `Travel_Fatigue_Index` (Float): Fatigue index
    - `Coach_Preference_Flag` (Binary): 1 if coach prefers player
    - `Team_Need_For_Role` (Binary): 1 if team needs this role
    - `Target` (Categorical): **Target**

### 4. Which Exoplanets Could Host Life
Astronomers have discovered thousands of exoplanets, but only a few may actually support life. With data on planet size, star distance, temperature, and atmospheric clues, we can classify which ones are truly worth further exploration.
An ML model can help astronomers prioritize observation efforts, pushing humanity one step closer to answering: Are we alone?
- **Dataset:** [exoplanet_habitability.csv](./exoplanet_habitability.csv)
- **Target Variable:** `Target` (`Habitable` or `Non-Habitable`)
- **Schema:**
    - `Mass_Earth_Relative` (Float): Mass relative to Earth
    - `Radius_Earth_Relative` (Float): Radius relative to Earth
    - `Surface_Temperature_K` (Float): Surface temperature in Kelvin
    - `Star_Distance_AU` (Float): Distance from star in AU
    - `Star_Type` (Categorical): Star type
    - `Orbital_Period_Days` (Float): Orbital period in days
    - `Atmosphere_Composition` (Categorical): Main atmospheric components
    - `Eccentricity` (Float): Orbital eccentricity
    - `Water_Vapor_Detected` (Binary): 1 if water vapor detected
    - `Receives_Stellar_Flux` (Float): Stellar flux received
    - `Tidal_Locking_Potential` (Binary): 1 if likely tidally locked
    - `Target` (Categorical): **Target** 