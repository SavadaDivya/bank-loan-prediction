from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the model and data
def main():
    global scaler, X, loan_data, model, max_min_loan_amounts
    
    file_path = r"C:\Users\divya\OneDrive\flask\loans (1).csv"  # Your CSV file path

    # Load and preprocess loan data
    loan_data = pd.read_csv(file_path)
    loan_data['loan_start'] = pd.to_datetime(loan_data['loan_start'])
    loan_data['loan_end'] = pd.to_datetime(loan_data['loan_end'])
    loan_data['loan_duration'] = (loan_data['loan_end'] - loan_data['loan_start']).dt.days
    loan_data = pd.get_dummies(loan_data, columns=['loan_type'])

    # Prepare features and target
    X = loan_data.drop(columns=['loan_id', 'loan_start', 'loan_end', 'client_id'])
    y = loan_data['loan_amount']  # This is continuous data, so it's a regression task

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model (Regression instead of Classification)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Identify loan type columns
    loan_type_columns = [col for col in X.columns if col.startswith('loan_type_')]

    # Predict loan amounts and calculate ranges
    y_train_pred = model.predict(X_train)
    df_train_pred = pd.DataFrame(y_train_pred, columns=['predicted_loan_amount'], index=y_train.index)
    for col in loan_type_columns:
        df_train_pred[col] = loan_data.iloc[y_train.index][col].values

    # Melt data for loan type ranges
    df_melted = df_train_pred.melt(id_vars=['predicted_loan_amount'], value_vars=loan_type_columns, var_name='loan_type', value_name='value')
    df_melted = df_melted[df_melted['value'] == 1].drop(columns=['value'])

    # Calculate min and max loan amounts for each loan type
    max_min_loan_amounts = df_melted.groupby('loan_type')['predicted_loan_amount'].agg(['max', 'min']).reset_index()

# Initialize data and model
main()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form inputs
        new_loan_start = request.form['loan_start']
        new_loan_end = request.form['loan_end']
        new_interest_rate = float(request.form['interest_rate'])
        new_loan_type = request.form['loan_type']
        new_loan_amount = float(request.form['loan_amount'])

        # Calculate loan duration
        new_loan_start = datetime.strptime(new_loan_start, '%Y-%m-%d')
        new_loan_end = datetime.strptime(new_loan_end, '%Y-%m-%d')
        new_loan_duration = (new_loan_end - new_loan_start).days

        # Prepare new loan data
        new_loan_data = pd.DataFrame({
            'rate': [new_interest_rate],
            'loan_duration': [new_loan_duration]
        })

        # One-hot encoding for loan type
        for col in X.columns:
            if col.startswith('loan_type_'):
                new_loan_data[col] = 0

        # Validate loan type
        loan_type_col = f'loan_type_{new_loan_type}'
        if loan_type_col in new_loan_data.columns:
            new_loan_data[loan_type_col] = 1
        else:
            return f"Loan type '{new_loan_type}' is not recognized."

        # Standardize new loan data
        new_loan_data = new_loan_data.reindex(columns=X.columns, fill_value=0)
        new_loan_data_scaled = scaler.transform(new_loan_data)

        # Get the loan amount range for the selected loan type
        loan_type_range = max_min_loan_amounts[max_min_loan_amounts['loan_type'] == loan_type_col]
        if loan_type_range.empty:
            return f"Loan type '{new_loan_type}' is not recognized."

        min_amount, max_amount = loan_type_range['min'].values[0], loan_type_range['max'].values[0]

        # Check if loan amount is within the allowed range
        if new_loan_amount < min_amount or new_loan_amount > max_amount:
            eligibility_message = f"Loan amount not within the allowed range ({min_amount} - {max_amount})."
        else:
            eligibility_message = "Eligible for loan. Loan amount is within the allowed range."

        # Return the prediction result with loan range
        return render_template('result.html', eligibility_message=eligibility_message, min_amount=min_amount, max_amount=max_amount)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
