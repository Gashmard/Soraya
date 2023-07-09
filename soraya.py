
import pandas as pd
import numpy as np
import sys
import shap
from tqdm.notebook import tqdm_notebook
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Function to read and convert file to DataFrame
def read_input_file(input_file):
    global dataframe_input_file

    # Determine the file format based on the file extension
    file_extension = input_file.split('.')[-1].lower()
    if file_extension == 'csv':
        # If the input file is in CSV format, read it directly as a DataFrame
        dataframe_input_file = pd.read_csv(input_file)
    elif file_extension in ['xlsx', 'xls']:
        # If the input file is in Excel format (XLSX or XLS), read it as a DataFrame
        dataframe_input_file = pd.read_excel(input_file)
    elif file_extension == 'json':
        # If the input file is in JSON format, read it as a DataFrame
        dataframe_input_file = pd.read_json(input_file)
    else:
        print(f"Unsupported file format: '{file_extension}'.", 
        "Please convert the input file format to one of the following formats:'csv', 'xlsx', 'xls' or 'json'.")
    return dataframe_input_file

def delete_unnamed_columns(dataframe_input_file):
    # Get the column names
    columns_Name = dataframe_input_file.columns
    
    # Create a list of column names to delete
    columns_Name_to_delete = [col for col in columns_Name if 'Unnamed: 0' in col or 'Unnamed' in col or 'Unnamed: 1' in col]
    
    # Drop the columns from the DataFrame
    dataframe_input_file.drop(columns_Name_to_delete, axis=1, inplace=True)
    
    return dataframe_input_file

def rename_target_column(input_file):
    global Target
    global df_features

    Target_column = input("Please enter the name of the column you want to \"Target\" for example: Price or Tc or ... .Then press the Enter button.")

    # Check if the target column exists in the DataFrame
    if Target_column not in input_file.columns:
        print("Invalid \"Target\" column!!. The columns did not contain such a name.")
        sys.exit()

    # Rename the target column to "Target"
    input_file.rename(columns={Target_column: "Target"}, inplace=True)
    print("The column of ",Target_column," was considered the \"Target\" of the problem.", '\n')

    Target = input_file["Target"]
    df_features = input_file.drop(["Target"], axis=1)
    List_df_features_Names=list(df_features.columns)
    print("The list of columns that were considered as features=",List_df_features_Names)
    return df_features, Target

def Select_Problem_Type_and_Algorithm():
    problem_type = input("Enter the problem type (R for Regression, C for Classification). Then press the Enter button.")

    if problem_type.lower() == "r":
        print("\n", "You entered the letter R, so your type of problem is regression.")

        Select_Algorithm = input("Select the algorithm for the feature selection process (R for  RandomForest, X for XGBoost): Running time for XGBoost > RandomForest. Then press the Enter button. ")
        if Select_Algorithm.lower() == "r":
            First_Step_Regression_RandomForest()
            Last_Step_Regression_RandomForest()
        elif Select_Algorithm.lower() == "x":
            First_Step_Regression_XGBoost()
            Last_Step_Regression_XGBoost()
        else:
            print("Invalid input. Please enter R for RandomForest or X for XGBoost.")

    elif problem_type.lower() == "c":
        print("\n", "You entered the letter C, so your type of problem is classification.")

        Select_Algorithm = input("Select the algorithm for the feature selection process (R for  RandomForest, X for XGBoost): Running time for XGBoost > RandomForest. Then press the Enter button. ")
        if Select_Algorithm.lower() == "r":
            First_Step_Classification_RandomForest()
            Last_Step_Classification_RandomForest()
        elif Select_Algorithm.lower() == "x":
            First_Step_Classification_XGBoost()
            Last_Step_Classification_XGBoost()
        else:
            print("Invalid input. Please enter R for RandomForest or X for XGBoost.")
    else:
        print("Invalid input. Please enter R for Regression or C for Classification.")

def First_Step_Regression_RandomForest():
    
    global Final_Features_Selection
    global Remaining_important_features
    global x_total_feature
    global Number_select_most_importance_Feature_shap_RandomForest
    global Test_size
    RandomForestReg_model = RandomForestRegressor(n_estimators=100, random_state=101, max_depth=5)

    Test_size = input("Please enter the test size for example: 0.15 or 0.2 or ... .Then press the Enter button.")
    X_train, X_test, y_train, y_test = train_test_split(df_features, Target, test_size=float(Test_size), random_state=42)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    for i in tqdm_notebook(range(1), desc='Please wait, the first step is running ... '):

        # Build a Dataframe with Correlation between Features
        corr_matrix = X_train.corr()
        # Take absolute values of correlated coefficients
        corr_matrix = corr_matrix.abs().unstack()
        corr_matrix = corr_matrix.sort_values(ascending=False)
        # Take only features with correlation above threshold of 0.8
        corr_matrix = corr_matrix[corr_matrix >= 0.8]
        corr_matrix = corr_matrix[corr_matrix < 1]
        corr_matrix = pd.DataFrame(corr_matrix).reset_index()
        corr_matrix.columns = ['feature1', 'feature2', 'Correlation']
        # print(corr_matrix)

        # Here we will unite the two columns of the features so that no features will be forgotten and then remove the repetitive features.
        feature_column_1=corr_matrix.feature1.values
        feature_column_2=corr_matrix.feature2.values
        arr_total_features = np.concatenate((feature_column_1, feature_column_2))
        # len(arr_total_features)
        arr_total_features_list = list(arr_total_features)
        after_drop_duplicate_arr_total_features_list = list(dict.fromkeys(arr_total_features_list)) # Duplicate items will be removed
        # print(len(after_drop_duplicate_arr_total_features_list))
        Total_features_Name=df_features.columns.values
        Total_features_Name_List=list(Total_features_Name) # Put the names of all the features in a list

        # Here we put the features that were of little correlation with each other, such as the correlation of less than 0.8.
        different_Features = [x for x in Total_features_Name_List if x not in after_drop_duplicate_arr_total_features_list]
        # print("The number of features with low correlation=",len(different_Features))
        features_with_low_correlation=df_features[different_Features]
        # print("\n","\n",  "Features_with_low_correlatio=","\n", features_with_low_correlation)

        # Get groups of features that are correlated amongs themselves
        grouped_features = []
        correlated_groups = []
        for feature in corr_matrix.feature1.unique():
            if feature not in grouped_features:
                # Find all features correlated to a single feature
                correlated_block = corr_matrix[corr_matrix.feature1 == feature]
                grouped_features = grouped_features + list(correlated_block.feature2.unique()) + [feature]
                # Append block of features to the list
                correlated_groups.append(correlated_block)

        # Here, separately and individually the value and efficiency of each feature of each clusters will be examined and 
        # we will be the first feature of the cluster that is the most important.
        select_most_importance_Featuers=[]
        for group in range(len(correlated_groups)):
            # print(correlated_groups[group])
            features_column1 = list(correlated_groups[group].feature1.unique())
            features_column2 = list(correlated_groups[group].feature2.unique())
            # print('features_column1=', features_column1)
            features_column1.extend(features_column2)
            # print('len_features_column1=', len(features_column1))
            RandomForestReg_model.fit(X_train[features_column1], y_train)
            importance = pd.concat([pd.Series(features_column1), pd.Series(RandomForestReg_model.feature_importances_)], axis=1)
            importance.columns = ['feature', 'importance']
            select_most_importance=importance.sort_values(by='importance', ascending=False).head(1)
            # print(select_most_importance.feature.values[0])
            select_most_importance_Featuers.append(select_most_importance.feature.values[0])

        # Here we delete duplicate features
        after_drop_duplicate_select_most_importance_Featuers = list(dict.fromkeys(select_most_importance_Featuers))
        print("The number of most important features selected up to first step =", len(after_drop_duplicate_select_most_importance_Featuers),"features","\n")
        most_importance_Featuers_from_each_group=df_features[after_drop_duplicate_select_most_importance_Featuers]
        concat_most_importance_and_low_correlation_Features=pd.concat([most_importance_Featuers_from_each_group,features_with_low_correlation],axis=1,join='inner')

    for i in tqdm_notebook(range(1), desc='Please wait, the first step is running ... '):  
        
        RandomForestReg_model_Shap = RandomForestRegressor(n_estimators=750, random_state=101, max_depth=5)
        RandomForestReg_model_Shap.fit(concat_most_importance_and_low_correlation_Features, Target)
        explainer = shap.TreeExplainer(RandomForestReg_model_Shap)
        shap_values = explainer.shap_values(concat_most_importance_and_low_correlation_Features)
        featuers_name=concat_most_importance_and_low_correlation_Features.columns
        shap.summary_plot(shap_values, features=concat_most_importance_and_low_correlation_Features,feature_names=featuers_name, plot_type="bar" )
        shap_sum = np.abs(shap_values).mean(axis=0)
        importance_df_shap = pd.DataFrame([concat_most_importance_and_low_correlation_Features.columns.tolist(), shap_sum.tolist()]).T
        importance_df_shap.columns = ['column_name', 'shap_importance']
        importance_df_shap = importance_df_shap.sort_values('shap_importance', ascending=False)

        if len(importance_df_shap) >= 10:
            Total_number_most_important_features = len(importance_df_shap)
            percentage = 10  # The percentage of the most important features that are selected.
            Number_select_most_importance_Feature_shap_RandomForest = int(Total_number_most_important_features * percentage / 100) # The number of features that are selected using the percentage specified in the line above
            # print("Number_select_most_importance_Feature_shap_RandomForest==",Number_select_most_importance_Feature_shap_RandomForest)
        else:
            Number_select_most_importance_Feature_shap_RandomForest = 1 #If the total number of the most important features is less than 10, the first feature is selected here

        Select_most_importance_Feature_shap_xgboost=importance_df_shap.head(Number_select_most_importance_Feature_shap_RandomForest)
        most_importance_Feature_shap_xgboost_column_name=Select_most_importance_Feature_shap_xgboost.column_name.values
        List_column_name_most_importance_Feature_shap_xgboost=list(most_importance_Feature_shap_xgboost_column_name)
        Selection_First_most_importance_Features_shap_xgboost=df_features[List_column_name_most_importance_Feature_shap_xgboost] # Here we select the first few features that are most important for example, the first 5 features
        drop_First_most_importance_Features=concat_most_importance_and_low_correlation_Features.drop(List_column_name_most_importance_Feature_shap_xgboost ,axis=1)
        Final_Features_Selection=Selection_First_most_importance_Features_shap_xgboost.copy()
        Remaining_important_features=drop_First_most_importance_Features.copy()
        x_total_feature=drop_First_most_importance_Features.copy()
        print("\n", "The execution of the first step of the program has been completed.", "\n")

def First_Step_Regression_XGBoost():
    
    global Final_Features_Selection
    global Remaining_important_features
    global x_total_feature
    global Number_select_most_importance_Feature_shap_xgboost
    global Test_size
    XGBoostReg_model = xgb.XGBRegressor(n_estimators=150,learning_rate=0.08,subsample=0.75,colsample_bytree=1, max_depth=7,gamma=0, verbosity=0)

    Test_size = input("Please enter the test size for example: 0.15 or 0.2 or ... .Then press the Enter button.")
    X_train, X_test, y_train, y_test = train_test_split(df_features, Target, test_size=float(Test_size), random_state=42)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    for i in tqdm_notebook(range(1), desc='Please wait, the first step is running ... '):

        # Build a Dataframe with Correlation between Features
        corr_matrix = X_train.corr()
        # Take absolute values of correlated coefficients
        corr_matrix = corr_matrix.abs().unstack()
        corr_matrix = corr_matrix.sort_values(ascending=False)
        # Take only features with correlation above threshold of 0.8
        corr_matrix = corr_matrix[corr_matrix >= 0.8]
        corr_matrix = corr_matrix[corr_matrix < 1]
        corr_matrix = pd.DataFrame(corr_matrix).reset_index()
        corr_matrix.columns = ['feature1', 'feature2', 'Correlation']
        # print(corr_matrix)

        # Here we will unite the two columns of the features so that no features will be forgotten and then remove the repetitive features.
        feature_column_1=corr_matrix.feature1.values
        feature_column_2=corr_matrix.feature2.values
        arr_total_features = np.concatenate((feature_column_1, feature_column_2))
        # len(arr_total_features)
        arr_total_features_list = list(arr_total_features)
        after_drop_duplicate_arr_total_features_list = list(dict.fromkeys(arr_total_features_list)) # Duplicate items will be removed
        # print(len(after_drop_duplicate_arr_total_features_list))
        Total_features_Name=df_features.columns.values
        Total_features_Name_List=list(Total_features_Name) # Put the names of all the features in a list

        # Here we put the features that were of little correlation with each other, such as the correlation of less than 0.8.
        different_Features = [x for x in Total_features_Name_List if x not in after_drop_duplicate_arr_total_features_list]
        # print("The number of features with low correlation=",len(different_Features))
        features_with_low_correlation=df_features[different_Features]
        # print("\n","\n",  "Features_with_low_correlatio=","\n", features_with_low_correlation)

        # Get groups of features that are correlated amongs themselves
        grouped_features = []
        correlated_groups = []
        for feature in corr_matrix.feature1.unique():
            if feature not in grouped_features:
                # Find all features correlated to a single feature
                correlated_block = corr_matrix[corr_matrix.feature1 == feature]
                grouped_features = grouped_features + list(correlated_block.feature2.unique()) + [feature]
                # Append block of features to the list
                correlated_groups.append(correlated_block)

        # Here, separately and individually the value and efficiency of each feature of each clusters will be examined and 
        # we will be the first feature of the cluster that is the most important.
        select_most_importance_Featuers=[]
        for group in range(len(correlated_groups)):
            # print(correlated_groups[group])
            features_column1 = list(correlated_groups[group].feature1.unique())
            features_column2 = list(correlated_groups[group].feature2.unique())
            # print('features_column1=', features_column1)
            features_column1.extend(features_column2)
            # print('len_features_column1=', len(features_column1))
            XGBoostReg_model.fit(X_train[features_column1], y_train)
            importance = pd.concat([pd.Series(features_column1), pd.Series(XGBoostReg_model.feature_importances_)], axis=1)
            importance.columns = ['feature', 'importance']
            select_most_importance=importance.sort_values(by='importance', ascending=False).head(1)
            # print(select_most_importance.feature.values[0])
            select_most_importance_Featuers.append(select_most_importance.feature.values[0])

        # Here we delete duplicate features
        after_drop_duplicate_select_most_importance_Featuers = list(dict.fromkeys(select_most_importance_Featuers))
        print("The number of most important features selected up to first step =", len(after_drop_duplicate_select_most_importance_Featuers),"features","\n")
        most_importance_Featuers_from_each_group=df_features[after_drop_duplicate_select_most_importance_Featuers]
        concat_most_importance_and_low_correlation_Features=pd.concat([most_importance_Featuers_from_each_group,features_with_low_correlation],axis=1,join='inner')

    for i in tqdm_notebook(range(1), desc='Please wait, the first step is running ... '):  
        XGBoostReg_model_Shap = xgb.XGBRegressor(n_estimators=1500,learning_rate=0.08,subsample=0.75,colsample_bytree=1, max_depth=7,gamma=0, verbosity=0)
        XGBoostReg_model_Shap.fit(concat_most_importance_and_low_correlation_Features, Target)
        explainer = shap.TreeExplainer(XGBoostReg_model_Shap, verbosity=0)
        shap_values = explainer.shap_values(concat_most_importance_and_low_correlation_Features)
        featuers_name=concat_most_importance_and_low_correlation_Features.columns
        shap.summary_plot(shap_values, features=concat_most_importance_and_low_correlation_Features,feature_names=featuers_name, plot_type="bar" )
        shap_sum = np.abs(shap_values).mean(axis=0)
        importance_df_shap = pd.DataFrame([concat_most_importance_and_low_correlation_Features.columns.tolist(), shap_sum.tolist()]).T
        importance_df_shap.columns = ['column_name', 'shap_importance']
        importance_df_shap = importance_df_shap.sort_values('shap_importance', ascending=False)

        if len(importance_df_shap) >= 10:
            Total_number_most_important_features = len(importance_df_shap)
            percentage = 10  # The percentage of the most important features that are selected.
            Number_select_most_importance_Feature_shap_xgboost = int(Total_number_most_important_features * percentage / 100) # The number of features that are selected using the percentage specified in the line above
            # print("Number_select_most_importance_Feature_shap_xgboost==",Number_select_most_importance_Feature_shap_xgboost)
        else:
            Number_select_most_importance_Feature_shap_xgboost = 1 #If the total number of the most important features is less than 10, the first feature is selected here

        Select_most_importance_Feature_shap_xgboost=importance_df_shap.head(Number_select_most_importance_Feature_shap_xgboost)
        most_importance_Feature_shap_xgboost_column_name=Select_most_importance_Feature_shap_xgboost.column_name.values
        List_column_name_most_importance_Feature_shap_xgboost=list(most_importance_Feature_shap_xgboost_column_name)
        Selection_First_most_importance_Features_shap_xgboost=df_features[List_column_name_most_importance_Feature_shap_xgboost] # Here we select the first few features that are most important for example, the first 5 features
        drop_First_most_importance_Features=concat_most_importance_and_low_correlation_Features.drop(List_column_name_most_importance_Feature_shap_xgboost ,axis=1)
        Final_Features_Selection=Selection_First_most_importance_Features_shap_xgboost.copy()
        Remaining_important_features=drop_First_most_importance_Features.copy()
        x_total_feature=drop_First_most_importance_Features.copy()
        print("\n", "The execution of the first step of the program has been completed.", "\n")

def First_Step_Classification_RandomForest():
    
    global Final_Features_Selection
    global Remaining_important_features
    global x_total_feature
    global Number_select_most_importance_Feature_shap_RandomForest
    global Test_size
    RandomForestClass_model = RandomForestClassifier(n_estimators=100, random_state=101, max_depth=5)

    Test_size = input("Please enter the test size for example: 0.15 or 0.2 or ... .Then press the Enter button.")
    X_train, X_test, y_train, y_test = train_test_split(df_features, Target, test_size=float(Test_size), random_state=42)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    for i in tqdm_notebook(range(1), desc='Please wait, the first step is running ... '):

        # Build a Dataframe with Correlation between Features
        corr_matrix = X_train.corr()
        # Take absolute values of correlated coefficients
        corr_matrix = corr_matrix.abs().unstack()
        corr_matrix = corr_matrix.sort_values(ascending=False)
        # Take only features with correlation above threshold of 0.8
        corr_matrix = corr_matrix[corr_matrix >= 0.8]
        corr_matrix = corr_matrix[corr_matrix < 1]
        corr_matrix = pd.DataFrame(corr_matrix).reset_index()
        corr_matrix.columns = ['feature1', 'feature2', 'Correlation']
        # print(corr_matrix)

        # Here we will unite the two columns of the features so that no features will be forgotten and then remove the repetitive features.
        feature_column_1=corr_matrix.feature1.values
        feature_column_2=corr_matrix.feature2.values
        arr_total_features = np.concatenate((feature_column_1, feature_column_2))
        # len(arr_total_features)
        arr_total_features_list = list(arr_total_features)
        after_drop_duplicate_arr_total_features_list = list(dict.fromkeys(arr_total_features_list)) # Duplicate items will be removed
        # print(len(after_drop_duplicate_arr_total_features_list))
        Total_features_Name=df_features.columns.values
        Total_features_Name_List=list(Total_features_Name) # Put the names of all the features in a list

        # Here we put the features that were of little correlation with each other, such as the correlation of less than 0.8.
        different_Features = [x for x in Total_features_Name_List if x not in after_drop_duplicate_arr_total_features_list]
        # print("The number of features with low correlation=",len(different_Features))
        features_with_low_correlation=df_features[different_Features]
        # print("\n","\n",  "Features_with_low_correlatio=","\n", features_with_low_correlation)

        # Get groups of features that are correlated amongs themselves
        grouped_features = []
        correlated_groups = []
        for feature in corr_matrix.feature1.unique():
            if feature not in grouped_features:
                # Find all features correlated to a single feature
                correlated_block = corr_matrix[corr_matrix.feature1 == feature]
                grouped_features = grouped_features + list(correlated_block.feature2.unique()) + [feature]
                # Append block of features to the list
                correlated_groups.append(correlated_block)

        # Here, separately and individually the value and efficiency of each feature of each clusters will be examined and 
        # we will be the first feature of the cluster that is the most important.
        select_most_importance_Featuers=[]
        for group in range(len(correlated_groups)):
            # print(correlated_groups[group])
            features_column1 = list(correlated_groups[group].feature1.unique())
            features_column2 = list(correlated_groups[group].feature2.unique())
            # print('features_column1=', features_column1)
            features_column1.extend(features_column2)
            # print('len_features_column1=', len(features_column1))
            RandomForestClass_model.fit(X_train[features_column1], y_train)
            importance = pd.concat([pd.Series(features_column1), pd.Series(RandomForestClass_model.feature_importances_)], axis=1)
            importance.columns = ['feature', 'importance']
            select_most_importance=importance.sort_values(by='importance', ascending=False).head(1)
            # print(select_most_importance.feature.values[0])
            select_most_importance_Featuers.append(select_most_importance.feature.values[0])

        # Here we delete duplicate features
        after_drop_duplicate_select_most_importance_Featuers = list(dict.fromkeys(select_most_importance_Featuers))
        print("The number of most important features selected up to first step =", len(after_drop_duplicate_select_most_importance_Featuers),"features","\n")
        most_importance_Featuers_from_each_group=df_features[after_drop_duplicate_select_most_importance_Featuers]
        concat_most_importance_and_low_correlation_Features=pd.concat([most_importance_Featuers_from_each_group,features_with_low_correlation],axis=1,join='inner')

    for i in tqdm_notebook(range(1), desc='Please wait, the first step is running ... '):  
        RandomForestClass_model_Shap = RandomForestClassifier(n_estimators=750, random_state=101, max_depth=5)
        RandomForestClass_model_Shap.fit(concat_most_importance_and_low_correlation_Features, Target)
        explainer = shap.TreeExplainer(RandomForestClass_model_Shap)
        shap_values = explainer.shap_values(concat_most_importance_and_low_correlation_Features)
        featuers_name=concat_most_importance_and_low_correlation_Features.columns
        shap.summary_plot(shap_values, features=concat_most_importance_and_low_correlation_Features,feature_names=featuers_name, plot_type="bar" )
        shap_sum = np.abs(shap_values).mean(axis=0)
        importance_df_shap = pd.DataFrame([concat_most_importance_and_low_correlation_Features.columns.tolist(), shap_sum.tolist()]).T
        importance_df_shap.columns = ['column_name', 'shap_importance']
        importance_df_shap = importance_df_shap.sort_values('shap_importance', ascending=False)

        if len(importance_df_shap) >= 10:
            Total_number_most_important_features = len(importance_df_shap)
            percentage = 10  # The percentage of the most important features that are selected.
            Number_select_most_importance_Feature_shap_RandomForest = int(Total_number_most_important_features * percentage / 100) # The number of features that are selected using the percentage specified in the line above
            print("Number_select_most_importance_Feature_shap_RandomForest==",Number_select_most_importance_Feature_shap_RandomForest)
        else:
            Number_select_most_importance_Feature_shap_RandomForest = 1 #If the total number of the most important features is less than 10, the first feature is selected here

        Select_most_importance_Feature_shap_xgboost=importance_df_shap.head(Number_select_most_importance_Feature_shap_RandomForest)
        most_importance_Feature_shap_xgboost_column_name=Select_most_importance_Feature_shap_xgboost.column_name.values
        List_column_name_most_importance_Feature_shap_xgboost=list(most_importance_Feature_shap_xgboost_column_name)
        Selection_First_most_importance_Features_shap_xgboost=df_features[List_column_name_most_importance_Feature_shap_xgboost] # Here we select the first few features that are most important for example, the first 5 features
        drop_First_most_importance_Features=concat_most_importance_and_low_correlation_Features.drop(List_column_name_most_importance_Feature_shap_xgboost ,axis=1)
        Final_Features_Selection=Selection_First_most_importance_Features_shap_xgboost.copy()
        Remaining_important_features=drop_First_most_importance_Features.copy()
        x_total_feature=drop_First_most_importance_Features.copy()
        print("\n", "The execution of the first step of the program has been completed.", "\n")

def First_Step_Classification_XGBoost():
    
    global Final_Features_Selection
    global Remaining_important_features
    global x_total_feature
    global Number_select_most_importance_Feature_shap_xgboost
    global Test_size
    XGBoostClass_model = xgb.XGBClassifier(n_estimators=150,learning_rate=0.08,subsample=0.75,colsample_bytree=1, max_depth=7,gamma=0, verbosity=0)

    Test_size = input("Please enter the test size for example: 0.15 or 0.2 or ... .Then press the Enter button.")
    X_train, X_test, y_train, y_test = train_test_split(df_features, Target, test_size=float(Test_size), random_state=42)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    for i in tqdm_notebook(range(1), desc='Please wait, the first step is running ... '):

        # Build a Dataframe with Correlation between Features
        corr_matrix = X_train.corr()
        # Take absolute values of correlated coefficients
        corr_matrix = corr_matrix.abs().unstack()
        corr_matrix = corr_matrix.sort_values(ascending=False)
        # Take only features with correlation above threshold of 0.8
        corr_matrix = corr_matrix[corr_matrix >= 0.8]
        corr_matrix = corr_matrix[corr_matrix < 1]
        corr_matrix = pd.DataFrame(corr_matrix).reset_index()
        corr_matrix.columns = ['feature1', 'feature2', 'Correlation']
        # print(corr_matrix)

        # Here we will unite the two columns of the features so that no features will be forgotten and then remove the repetitive features.
        feature_column_1=corr_matrix.feature1.values
        feature_column_2=corr_matrix.feature2.values
        arr_total_features = np.concatenate((feature_column_1, feature_column_2))
        # len(arr_total_features)
        arr_total_features_list = list(arr_total_features)
        after_drop_duplicate_arr_total_features_list = list(dict.fromkeys(arr_total_features_list)) # Duplicate items will be removed
        # print(len(after_drop_duplicate_arr_total_features_list))
        Total_features_Name=df_features.columns.values
        Total_features_Name_List=list(Total_features_Name) # Put the names of all the features in a list

        # Here we put the features that were of little correlation with each other, such as the correlation of less than 0.8.
        different_Features = [x for x in Total_features_Name_List if x not in after_drop_duplicate_arr_total_features_list]
        # print("The number of features with low correlation=",len(different_Features))
        features_with_low_correlation=df_features[different_Features]
        # print("\n","\n",  "Features_with_low_correlatio=","\n", features_with_low_correlation)

        # Get groups of features that are correlated amongs themselves
        grouped_features = []
        correlated_groups = []
        for feature in corr_matrix.feature1.unique():
            if feature not in grouped_features:
                # Find all features correlated to a single feature
                correlated_block = corr_matrix[corr_matrix.feature1 == feature]
                grouped_features = grouped_features + list(correlated_block.feature2.unique()) + [feature]
                # Append block of features to the list
                correlated_groups.append(correlated_block)

        # Here, separately and individually the value and efficiency of each feature of each clusters will be examined and 
        # we will be the first feature of the cluster that is the most important.
        select_most_importance_Featuers=[]
        for group in range(len(correlated_groups)):
            # print(correlated_groups[group])
            features_column1 = list(correlated_groups[group].feature1.unique())
            features_column2 = list(correlated_groups[group].feature2.unique())
            # print('features_column1=', features_column1)
            features_column1.extend(features_column2)
            # print('len_features_column1=', len(features_column1))
            XGBoostClass_model.fit(X_train[features_column1], y_train)
            importance = pd.concat([pd.Series(features_column1), pd.Series(XGBoostClass_model.feature_importances_)], axis=1)
            importance.columns = ['feature', 'importance']
            select_most_importance=importance.sort_values(by='importance', ascending=False).head(1)
            # print(select_most_importance.feature.values[0])
            select_most_importance_Featuers.append(select_most_importance.feature.values[0])

        # Here we delete duplicate features
        after_drop_duplicate_select_most_importance_Featuers = list(dict.fromkeys(select_most_importance_Featuers))
        print("The number of most important features selected up to first step =", len(after_drop_duplicate_select_most_importance_Featuers),"features","\n")
        most_importance_Featuers_from_each_group=df_features[after_drop_duplicate_select_most_importance_Featuers]
        concat_most_importance_and_low_correlation_Features=pd.concat([most_importance_Featuers_from_each_group,features_with_low_correlation],axis=1,join='inner')

    for i in tqdm_notebook(range(1), desc='Please wait, the first step is running ... '):  
        XGBoostClass_model_Shap  = xgb.XGBClassifier(n_estimators=1500,learning_rate=0.08,subsample=0.75,colsample_bytree=1, max_depth=7,gamma=0, verbosity=0)
        XGBoostClass_model_Shap.fit(concat_most_importance_and_low_correlation_Features, Target)
        explainer = shap.TreeExplainer(XGBoostClass_model_Shap, verbosity=0)
        shap_values = explainer.shap_values(concat_most_importance_and_low_correlation_Features)
        featuers_name=concat_most_importance_and_low_correlation_Features.columns
        shap.summary_plot(shap_values, features=concat_most_importance_and_low_correlation_Features,feature_names=featuers_name, plot_type="bar" )
        shap_sum = np.abs(shap_values).mean(axis=0)
        importance_df_shap = pd.DataFrame([concat_most_importance_and_low_correlation_Features.columns.tolist(), shap_sum.tolist()]).T
        importance_df_shap.columns = ['column_name', 'shap_importance']
        importance_df_shap = importance_df_shap.sort_values('shap_importance', ascending=False)

        if len(importance_df_shap) >= 10:
            Total_number_most_important_features = len(importance_df_shap)
            percentage = 10  # The percentage of the most important features that are selected.
            Number_select_most_importance_Feature_shap_xgboost = int(Total_number_most_important_features * percentage / 100) # The number of features that are selected using the percentage specified in the line above
            # print("Number_select_most_importance_Feature_shap_xgboost==",Number_select_most_importance_Feature_shap_xgboost)
        else:
            Number_select_most_importance_Feature_shap_xgboost = 1 #If the total number of the most important features is less than 10, the first feature is selected here

        Select_most_importance_Feature_shap_xgboost=importance_df_shap.head(Number_select_most_importance_Feature_shap_xgboost)
        most_importance_Feature_shap_xgboost_column_name=Select_most_importance_Feature_shap_xgboost.column_name.values
        List_column_name_most_importance_Feature_shap_xgboost=list(most_importance_Feature_shap_xgboost_column_name)
        Selection_First_most_importance_Features_shap_xgboost=df_features[List_column_name_most_importance_Feature_shap_xgboost] # Here we select the first few features that are most important for example, the first 5 features
        drop_First_most_importance_Features=concat_most_importance_and_low_correlation_Features.drop(List_column_name_most_importance_Feature_shap_xgboost ,axis=1)
        Final_Features_Selection=Selection_First_most_importance_Features_shap_xgboost.copy()
        Remaining_important_features=drop_First_most_importance_Features.copy()
        x_total_feature=drop_First_most_importance_Features.copy()
        print("\n", "The execution of the first step of the program has been completed.", "\n")

def Last_Step_Regression_RandomForest():
    global Final_Features_Selection
    global Remaining_important_features
    global x_total_feature
    global Number_select_most_importance_Feature_shap_RandomForest
    global Test_size
    global Total_final_features_that_user_wants_to_remain

    Total_final_features_that_user_wants_to_remain = input("Please enter the total number of the most important features that you want to be selected, for example 15 or 20 or 40 or... .Then press the Enter button.")
    Number_loop_executions=(int(Total_final_features_that_user_wants_to_remain) - Number_select_most_importance_Feature_shap_RandomForest)
    R2_List_Total=[]
    print("\n", "Note: Depending on your data set, the last step may take some time.") 
    for i in tqdm_notebook(range(Number_loop_executions), desc='Please wait, the last step is running ... '):  
        df_score_Total=pd.DataFrame()
        for j in range(len(Remaining_important_features.columns)):
            # print("j=",j)
            column_name=Remaining_important_features.columns[j] 
            Final_Features_Selection[Remaining_important_features.columns[j]]=Remaining_important_features.loc[:,Remaining_important_features.columns[j]]

            x_train, x_test, y_train, y_test = train_test_split(Final_Features_Selection, Target, test_size=float(Test_size), random_state=42)
            model_New = RandomForestRegressor(n_estimators=100, random_state=101, max_depth=4)

            model_New.fit(x_train, y_train)
            predictions = model_New.predict(x_test)
            r2_New=r2_score(y_test, predictions)
            # print('r2_new=',r2_New )
            import warnings
            warnings.filterwarnings("ignore", message="The frame.append method is deprecated and will be removed from pandas in a future version.")
            df_score_Total=df_score_Total.append({'Feature_Name' : column_name,'Score' :r2_New },ignore_index=True);
            Final_Features_Selection.drop([column_name], axis=1, inplace=True)

        first_row_for_most_importance_features=df_score_Total.sort_values(by=['Score'], ascending=False).head(1)
        most_importance_feature=first_row_for_most_importance_features.Feature_Name.values
        Score_of_most_importance_features=first_row_for_most_importance_features.Score.values[0]
        print("Score when add most importance feature=", Score_of_most_importance_features)
        R2_List_Total.append(Score_of_most_importance_features)
        print('Name of most importance features=', most_importance_feature)
        List_most_importance_feature=list(most_importance_feature)
        New_Feature=x_total_feature[List_most_importance_feature]
        Final_Features_Selection= pd.concat([Final_Features_Selection, New_Feature], axis=1, join='inner')
        Remaining_important_features.drop([most_importance_feature[0]], axis=1, inplace=True)
        print("=========================================================================", "\n")

def Last_Step_Regression_XGBoost():
    global Final_Features_Selection
    global Remaining_important_features
    global x_total_feature
    global Number_select_most_importance_Feature_shap_xgboost
    global Test_size
    global Total_final_features_that_user_wants_to_remain

    Total_final_features_that_user_wants_to_remain = input("Please enter the total number of the most important features that you want to be selected, for example 15 or 20 or 40 or... .Then press the Enter button.")
    Number_loop_executions=(int(Total_final_features_that_user_wants_to_remain) - Number_select_most_importance_Feature_shap_xgboost)
    R2_List_Total=[]
    print("\n", "Note: Depending on your data set, the last step may take some time.") 
    for i in tqdm_notebook(range(Number_loop_executions), desc='Please wait, the last step is running ... '):  
        df_score_Total=pd.DataFrame()
        for j in range(len(Remaining_important_features.columns)):
            # print("j=",j)
            column_name=Remaining_important_features.columns[j] 
            Final_Features_Selection[Remaining_important_features.columns[j]]=Remaining_important_features.loc[:,Remaining_important_features.columns[j]]

            x_train, x_test, y_train, y_test = train_test_split(Final_Features_Selection, Target, test_size=float(Test_size), random_state=42)
            model_New = xgb.XGBRegressor(n_estimators=750,learning_rate=0.08,subsample=0.75,colsample_bytree=1, max_depth=6,gamma=0, verbosity=0)

            model_New.fit(x_train, y_train)
            predictions = model_New.predict(x_test)
            r2_New=r2_score(y_test, predictions)
            # print('r2_new=',r2_New )
            import warnings
            warnings.filterwarnings("ignore", message="The frame.append method is deprecated and will be removed from pandas in a future version.")
            df_score_Total=df_score_Total.append({'Feature_Name' : column_name,'Score' :r2_New },ignore_index=True);
            Final_Features_Selection.drop([column_name], axis=1, inplace=True)

        first_row_for_most_importance_features=df_score_Total.sort_values(by=['Score'], ascending=False).head(1)
        most_importance_feature=first_row_for_most_importance_features.Feature_Name.values
        Score_of_most_importance_features=first_row_for_most_importance_features.Score.values[0]
        print("Score when add most importance feature=", Score_of_most_importance_features)
        R2_List_Total.append(Score_of_most_importance_features)
        print('Name of most importance features=', most_importance_feature)
        List_most_importance_feature=list(most_importance_feature)
        New_Feature=x_total_feature[List_most_importance_feature]
        Final_Features_Selection= pd.concat([Final_Features_Selection, New_Feature], axis=1, join='inner')
        Remaining_important_features.drop([most_importance_feature[0]], axis=1, inplace=True)
        print("=========================================================================", "\n")

def Last_Step_Classification_RandomForest():
    global Final_Features_Selection
    global Remaining_important_features
    global x_total_feature
    global Number_select_most_importance_Feature_shap_RandomForest
    global Test_size
    global Total_final_features_that_user_wants_to_remain

    Total_final_features_that_user_wants_to_remain = input("Please enter the total number of the most important features that you want to be selected, for example 15 or 20 or 40 or... .Then press the Enter button.")
    Number_loop_executions=(int(Total_final_features_that_user_wants_to_remain) - Number_select_most_importance_Feature_shap_RandomForest)
    R2_List_Total=[]
    print("\n", "Note: Depending on your data set, the last step may take some time.") 
    for i in tqdm_notebook(range(Number_loop_executions), desc='Please wait, the last step is running ... '):  
        df_score_Total=pd.DataFrame()
        for j in range(len(Remaining_important_features.columns)):
            column_name=Remaining_important_features.columns[j] 
            Final_Features_Selection[Remaining_important_features.columns[j]]=Remaining_important_features.loc[:,Remaining_important_features.columns[j]]

            x_train, x_test, y_train, y_test = train_test_split(Final_Features_Selection, Target, test_size=float(Test_size), random_state=42)
            model_New = RandomForestClassifier(n_estimators=100, random_state=101, max_depth=4)

            model_New.fit(x_train, y_train)
            predictions = model_New.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)
            # print('r2_new=',r2_New )
            import warnings
            warnings.filterwarnings("ignore", message="The frame.append method is deprecated and will be removed from pandas in a future version.")
            df_score_Total=df_score_Total.append({'Feature_Name' : column_name,'Score' :accuracy },ignore_index=True); 
            Final_Features_Selection.drop([column_name], axis=1, inplace=True)

        first_row_for_most_importance_features=df_score_Total.sort_values(by=['Score'], ascending=False).head(1)
        most_importance_feature=first_row_for_most_importance_features.Feature_Name.values
        Score_of_most_importance_features=first_row_for_most_importance_features.Score.values[0]
        print("Score when add most importance feature=", Score_of_most_importance_features)
        R2_List_Total.append(Score_of_most_importance_features)
        print('Name of most importance features=', most_importance_feature)
        List_most_importance_feature=list(most_importance_feature)
        New_Feature=x_total_feature[List_most_importance_feature]
        Final_Features_Selection= pd.concat([Final_Features_Selection, New_Feature], axis=1, join='inner')
        Remaining_important_features.drop([most_importance_feature[0]], axis=1, inplace=True)
        print("=========================================================================", "\n")

def Last_Step_Classification_XGBoost():
    global Final_Features_Selection
    global Remaining_important_features
    global x_total_feature
    global Number_select_most_importance_Feature_shap_xgboost
    global Test_size
    global Total_final_features_that_user_wants_to_remain

    Total_final_features_that_user_wants_to_remain = input("Please enter the total number of the most important features that you want to be selected, for example 15 or 20 or 40 or... .Then press the Enter button.")
    Number_loop_executions=(int(Total_final_features_that_user_wants_to_remain) - Number_select_most_importance_Feature_shap_xgboost)
    # print("Number_loop_executions=", Number_loop_executions)
    R2_List_Total=[]
    print("\n", "Note: Depending on your data set, the last step may take some time.") 
    for i in tqdm_notebook(range(Number_loop_executions), desc='Please wait, the last step is running ... '):  
        df_score_Total=pd.DataFrame()
        for j in range(len(Remaining_important_features.columns)):
            # print("j=",j)
            column_name=Remaining_important_features.columns[j] 
            Final_Features_Selection[Remaining_important_features.columns[j]]=Remaining_important_features.loc[:,Remaining_important_features.columns[j]]

            x_train, x_test, y_train, y_test = train_test_split(Final_Features_Selection, Target, test_size=float(Test_size), random_state=42)
            model_New = xgb.XGBClassifier(n_estimators=750,learning_rate=0.08,subsample=0.75,colsample_bytree=1, max_depth=6,gamma=0, verbosity=0)

            model_New.fit(x_train, y_train)
            predictions = model_New.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)
            # print('r2_new=',r2_New )
            import warnings
            warnings.filterwarnings("ignore", message="The frame.append method is deprecated and will be removed from pandas in a future version.")

            df_score_Total=df_score_Total.append({'Feature_Name' : column_name,'Score' :accuracy },ignore_index=True);
            Final_Features_Selection.drop([column_name], axis=1, inplace=True)

        first_row_for_most_importance_features=df_score_Total.sort_values(by=['Score'], ascending=False).head(1)
        most_importance_feature=first_row_for_most_importance_features.Feature_Name.values
        Score_of_most_importance_features=first_row_for_most_importance_features.Score.values[0]

        print("Score when add most importance feature=", Score_of_most_importance_features)
        R2_List_Total.append(Score_of_most_importance_features)
        print('Name of most importance features=', most_importance_feature)
        List_most_importance_feature=list(most_importance_feature)
        New_Feature=x_total_feature[List_most_importance_feature]
        Final_Features_Selection= pd.concat([Final_Features_Selection, New_Feature], axis=1, join='inner')
        Remaining_important_features.drop([most_importance_feature[0]], axis=1, inplace=True)
        print("=========================================================================", "\n")

def save_dataframe():
    output_file_path = input("Please enter a path to save the output file in your system like the example: /home/Test. Then press the Enter button. ")
    output_file_name = "output.csv"  # Desired output file name
    Final_Features_Selection.to_csv(output_file_path + "/" + output_file_name, index=False)
    print("The output dataframe containing the top",Total_final_features_that_user_wants_to_remain, "important features was successfully saved.")
  
def main():

    input_file = input("Please enter the file path as in the example: home/Test/example.csv or example.xlsx or example.json... .Then press the Enter button. ")
    read_input_file(input_file)
    delete_unnamed_columns(dataframe_input_file)
    rename_target_column(dataframe_input_file)
    Select_Problem_Type_and_Algorithm()
    save_dataframe()

if __name__ == "__main__":
    main()

