import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and clean data
data = pd.read_csv('Road.csv', header=0, sep=",")
data.dropna(axis=0, inplace=True)

# Step 2: Define features and target
X = data.drop(columns=['Accident_severity'])
y = data['Accident_severity']

# Step 3: Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 4: Preprocessing for categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 5: Build and train the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 7: Save the model
joblib.dump(model, 'road_accident_model.pkl')

# Step 8: Make a prediction
hypothetical_data = pd.DataFrame({
    'Time': ['17:02:00'],
    'Day_of_week': ['Monday'],
    'Age_band_of_driver': ['18-30'],
    'Sex_of_driver': ['Male'],
    'Educational_level': ['Above high school'],
    'Vehicle_driver_relation': ['Employee'],
    'Driving_experience': ['1-2yr'],
    'Type_of_vehicle': ['Automobile'],
    'Owner_of_vehicle': ['Owner'],
    'Service_year_of_vehicle': ['Above 10yr'],
    'Defect_of_vehicle': ['No defect'],
    'Area_accident_occured': ['Residential areas'],
    'Lanes_or_Medians': ['Undivided Two way'],
    'Road_allignment': ['Tangent road with flat terrain'],
    'Types_of_Junction': ['No junction'],
    'Road_surface_type': ['Asphalt roads'],
    'Road_surface_conditions': ['Dry'],
    'Light_conditions': ['Daylight'],
    'Weather_conditions': ['Normal'],
    'Type_of_collision': ['Collision with roadside-parked vehicles'],
    'Number_of_vehicles_involved': [2],
    'Number_of_casualties': [2],
    'Vehicle_movement': ['Going straight'],
    'Casualty_class': ['na'],
    'Sex_of_casualty': ['na'],
    'Age_band_of_casualty': ['na'],
    'Casualty_severity': ['na'],
    'Work_of_casuality': ['na'],
    'Fitness_of_casuality': ['na'],
    'Pedestrian_movement': ['Not a Pedestrian'],
    'Cause_of_accident': ['Moving Backward']
})

predicted_severity_encoded = model.predict(hypothetical_data)
predicted_severity = label_encoder.inverse_transform([int(round(predicted_severity_encoded[0]))])
print(f'Predicted Accident Severity: {predicted_severity[0]}')

# Step 9: Visualizations
# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.savefig('residual_plot.png')  # Save the plot as an image
plt.show()

# Actual vs. Predicted Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linestyle='--')  # Diagonal line
plt.title('Actual vs. Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.savefig('actual_vs_predicted.png')  # Save the plot as an image
plt.show()

# Bar Plot for Predicted Severity
predicted_severity_rounded = [int(round(p)) for p in y_pred]  # Round predictions
predicted_severity_counts = pd.Series(label_encoder.inverse_transform(predicted_severity_rounded)).value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=predicted_severity_counts.index, y=predicted_severity_counts.values)
plt.title('Distribution of Predicted Accident Severity')
plt.xlabel('Accident Severity')
plt.ylabel('Count')
plt.savefig('predicted_severity_distribution.png')  # Save the plot as an image
plt.show()

# Step 10: Save results to an HTML file
with open('results.html', 'w') as f:
    f.write(f"<h1>Road Accident Severity Analysis</h1>")
    f.write(f"<h2>Mean Squared Error: {mse}</h2>")
    f.write(f"<h2>Predicted Accident Severity: {predicted_severity[0]}</h2>")
    f.write("<h2>Visualizations</h2>")
    f.write("<h3>Residual Plot</h3>")
    f.write('<img src="residual_plot.png" alt="Residual Plot">')
    f.write("<h3>Actual vs. Predicted Plot</h3>")
    f.write('<img src="actual_vs_predicted.png" alt="Actual vs. Predicted Plot">')
    f.write("<h3>Predicted Severity Distribution</h3>")
    f.write('<img src="predicted_severity_distribution.png" alt="Predicted Severity Distribution">')

print("Results saved to results.html")