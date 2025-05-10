# Electric Vehicle Price Prediction

## Objective
This project aims to predict the price of Electric Vehicles (EVs) based on various features. The primary goal is to answer the question: `Given a car's model, make, and other parameters, what price can this vehicle be bought or sold for?` This is achieved by building and evaluating a Support Vector Machine (SVM) model on the 'Electric Vehicle Data' dataset provided by Kaggle as part of the Electric Vehicle Price Prediction competition.

## Dataset
The dataset contains information on Battery Electric Vehicles (BEVs) and Plug-in Hybrid Electric Vehicles (PHEVs) registered with the Washington State Department of Licensing (DOL).

* **Dataset Link**: [Electric Vehicle Population Dataset](https://drive.google.com/file/d/1kZ299dY3rKLvvnfTsaPtfrUbZb7k31of/view)
* **Source Notebook**: `Support_vector_machines_checkpoint.ipynb`

### Columns Explanation
* **VIN (1-10)**: The 1st 10 characters of each vehicle's Vehicle Identification Number (VIN).
* **County**: The county in which the registered owner resides.
* **City**: The city in which the registered owner resides.
* **State**: The state in which the registered owner resides.
* **ZIP Code**: The 5-digit zip code in which the registered owner resides.
* **Model Year**: The model year of the vehicle, determined by decoding the VIN.
* **Make**: The manufacturer of the vehicle, determined by decoding the VIN.
* **Model**: The model of the vehicle, determined by decoding the VIN.
* **Electric Vehicle Type**: Distinguishes the vehicle as all-electric (BEV) or a plug-in hybrid (PHEV).
* **Clean Alternative Fuel Vehicle (CAFV) Eligibility**: Categorizes vehicles as Clean Alternative Fuel Vehicles (CAFVs) based on fuel and electric-only range requirements.
* **Electric Range**: Describes how far a vehicle can travel purely on its electric charge.
* **Base MSRP**: The lowest Manufacturer's Suggested Retail Price (MSRP) for any trim level of the model.
* **Legislative District**: The specific section of Washington State where the vehicle's owner resides, as represented in the state legislature.
* **DOL Vehicle ID**: Unique number assigned to each vehicle by the Department of Licensing.
* **Vehicle Location**: The center of the ZIP Code for the registered vehicle.
* **Electric Utility**: The electric power retail service territory serving the address of the registered vehicle.
* **Expected Price ($1k)**: This is the target variable, representing the expected price of the vehicle in thousands of dollars.

## Project Workflow

The project follows these key steps:

1.  **Data Import and Basic Exploration**:
    * Loaded the `Electric_cars_dataset.csv` file into a pandas DataFrame.
    * Displayed general information about the dataset using `data.info()`, `data.head()`, and `data.describe()`.

2.  **Pandas Profiling Report**:
    * Generated a comprehensive pandas profiling report using `ydata-profiling` to gain deeper insights into data types, missing values, correlations, and distributions.
    * The report was saved as `Electric_Vehicle_Data_profiling_report.html`.

3.  **Handling Missing and Corrupted Values**:
    * Identified columns with missing values.
    * Missing numerical values were imputed using the median, and missing categorical values were imputed using the mode.
    * Checked for string representations of 'NA' or 'NULL' in object columns.

4.  **Duplicate Removal**:
    * Checked for duplicate rows in the dataset. The notebook indicates no duplicate rows were found after initial cleaning.

5.  **Outlier Handling**:
    * Implemented functions to identify outliers using IQR, Z-score, and Modified Z-score methods.
    * Visualized outliers using scatter plots.
    * Handled outliers, likely using capping (clipping values to the 1st and 99th percentiles) or removal, though the specific method applied to each column would be detailed in the notebook.

6.  **Categorical Feature Encoding**:
    * Categorical features were converted into numerical representations suitable for the SVM model. The notebook implies Label Encoding was used as `encoded_data` is created and subsequently used.

7.  **Feature and Target Selection**:
    * **Features (X)**: `Model Year`, `Electric Range`, `Base MSRP`, `ZIP Code`, `Make`, `Model`, `Electric Vehicle Type`, `Clean Alternative Fuel Vehicle (CAFV) Eligibility`, `County`, `City`, `State`, `Electric Utility`.
    * **Target Variable (y)**: `Expected Price ($1k)`.

8.  **Data Splitting**:
    * The dataset was split into training (80%) and testing (20%) sets using `train_test_split` with `random_state=42`.

9.  **Feature Scaling**:
    * Numerical features in the training and testing sets were scaled using `StandardScaler` to normalize the data, which is beneficial for SVM models.

10. **SVM Model Building and Training**:
    * A Support Vector Regressor (`SVR`) model was built and trained on the scaled training data.
    * Default hyperparameters for `SVR` were initially used (e.g., C=1.0, kernel='rbf', epsilon=0.1).

11. **Model Performance Assessment**:
    * Predictions were made on the scaled test set.
    * The model's performance was evaluated using:
        * Mean Squared Error (MSE)
        * Root Mean Squared Error (RMSE)
        * R-squared (R2)
    * A scatter plot of actual vs. predicted prices was generated to visualize prediction accuracy.

## Results
The SVM model achieved the following performance on the test set:
* **Mean Squared Error (MSE)**: 1106.68
* **Root Mean Squared Error (RMSE)**: 33.27 (This suggests the average difference between actual and predicted prices is around $33.27k, assuming the target is in $1k units as named)
* **R-squared (R2)**: 0.67 (This implies the SVM model can explain about 67% of the variation in the expected price.)

The scatter plot showed a generally positive relationship between actual and predicted prices, though with some scatter and outliers, indicating areas for potential improvement.

## Conclusion
The SVM model provides a reasonably effective baseline for predicting electric vehicle prices, with an R-squared value of 0.67. However, the presence of outliers and deviations in the scatter plot suggests that further improvements could be made. Potential next steps include more advanced feature engineering, hyperparameter tuning for the SVR model, or exploring alternative machine learning algorithms.

## How to Run

1.  **Prerequisites/Dependencies**:
    * Python 3.x
    * Jupyter Notebook or JupyterLab
    * The following Python libraries (and their dependencies):
        * `pandas`
        * `numpy`
        * `scikit-learn`
        * `matplotlib`
        * `seaborn`
        * `ydata-profiling` (install via `pip install ydata-profiling`)
        * `tabulate`

2.  **Dataset**:
    * Download the 'Electric Vehicle Population Data' from Kaggle (or the specific link used).
    * Ensure the dataset is named `Electric_cars_dataset.csv` and placed in the correct path as referenced in the notebook (e.g., `/content/` if running in Colab with the file uploaded there, or update the path in the notebook).

3.  **Execution**:
    * Open and run the `Support_vector_machines_checkpoint.ipynb` Jupyter Notebook.
    * Ensure that if you are running in Google Colab, you mount your Google Drive if the dataset or output paths refer to it (as seen in the notebook for saving the profiling report).
