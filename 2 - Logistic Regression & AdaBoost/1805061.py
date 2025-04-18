import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
    
    def preprocess(self):
        if self.dataset_name == 'telco-customer-churn':
            return self.telco_customer_churn_dataset_preprocessing()
        elif self.dataset_name == 'adult':
            return self.adult_dataset_preprocessing()
        elif self.dataset_name == 'credit-card-fraud':
            return self.credit_card_fraud_dataset_preprocessing()
        
    def telco_customer_churn_dataset_preprocessing(self):
        X = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv', na_values=' ')
        
        # drop unnecessary column
        X.drop(['customerID'], inplace=True, axis=1)

        # separate target column
        X['Churn'].replace({'Yes' : 1, 'No' : 0}, inplace=True)
        y = X['Churn']
        X.drop(['Churn'], inplace=True, axis=1)

        # replace data
        X.replace({'No internet service' : 'No'}, inplace=True)
        X.replace({'No phone service' : 'No'}, inplace=True)

        # handle missing values
        # print count of missing values per column
        # print(X.isnull().sum())
        X['TotalCharges'].replace({' ': np.nan}, inplace=True)
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X['TotalCharges'] = imputer.fit_transform(X[['TotalCharges']])

        # encode categorical data
        categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
        encoder = OneHotEncoder(sparse_output=False) 
        for key in categorical_cols:
            encoded_data = encoder.fit_transform(X[[key]])
            encoded_columns = encoder.get_feature_names_out([key])
            X[encoded_columns] = encoded_data
            X.drop(key, axis=1, inplace=True)

        # split training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # normalize 
        scaler = StandardScaler()
        for key in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            X_train[key] = scaler.fit_transform(X_train[[key]])
            X_test[key] = scaler.fit_transform(X_test[[key]])
        
        # align columns
        X_test = X_test[X_train.columns]

        # convert to numpy
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy() 
        X_test  =  X_test.to_numpy() 
        y_test  = y_test.to_numpy()

        return X_train, y_train, X_test, y_test        
    
    def adult_dataset_preprocessing(self):
        X_train = pd.read_csv('adult.data', na_values=' ', header=None)
        X_test = pd.read_csv('adult.test', na_values=' ', skiprows=1, header=None)

        # add column names
        cols = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                'marital_status', 'occupation', 'relationship', 'race', 'sex',
                'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
                'income']
        X_train.columns = cols
        X_test.columns = cols

        # replace data
        X_train.replace({' <=50K' : 0, ' >50K' : 1}, inplace=True)
        X_test.replace({' <=50K.' : 0, ' >50K.' : 1}, inplace=True)

        # separate target column
        # train
        y_train = X_train['income']
        X_train.drop(['income'], inplace=True, axis=1)
        # test
        y_test = X_test['income']
        X_test.drop(['income'], inplace=True, axis=1)

        
        # handle missing values
        # print count of missing values per column
        # print(X_train.isnull().sum())
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        for key in ['workclass', 'occupation', 'native_country']:
            # train
            X_train[key].replace({' ?': np.nan}, inplace=True)
            X_train[key] = imputer.fit_transform(X_train[[key]])
            # test
            X_test[key].replace({' ?': np.nan}, inplace=True)
            X_test[key] = imputer.fit_transform(X_test[[key]])

        # encode categorical data
        categorical_cols=['workclass', 'education', 'marital_status', 'occupation', 
                          'relationship', 'race', 'sex', 'native_country']
        encoder = OneHotEncoder(sparse_output=False) 
        for key in categorical_cols:
            # train
            encoded_data = encoder.fit_transform(X_train[[key]])
            encoded_columns = encoder.get_feature_names_out([key])
            X_train[encoded_columns] = encoded_data
            X_train.drop(key, axis=1, inplace=True)
            # test
            encoded_data = encoder.fit_transform(X_test[[key]])
            encoded_columns = encoder.get_feature_names_out([key])
            X_test[encoded_columns] = encoded_data
            X_test.drop(key, axis=1, inplace=True)

        # add missing cols
        # print(list(set(X_train.columns)-set(X_test.columns)))
        X_test['native_country_ Holand-Netherlands'] = 0

        # normalize 
        scaler = StandardScaler()
        for key in ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']:
            X_train[key] = scaler.fit_transform(X_train[[key]])
            X_test[key] = scaler.fit_transform(X_test[[key]])

        # align columns
        X_test = X_test[X_train.columns]

        # convert to numpy
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy() 
        X_test  =  X_test.to_numpy() 
        y_test  = y_test.to_numpy()

        return X_train, y_train, X_test, y_test 
       
    def credit_card_fraud_dataset_preprocessing(self):
        X = pd.read_csv('creditcard.csv', na_values=' ')

        # Select all rows where 'Class' is equal to 1
        class_1_rows = X[X['Class'] == 1]

        # Select 20,000 random rows where 'Class' is equal to 0
        class_0_rows = X[X['Class'] == 0].sample(n=20000, random_state=1)

        # Concatenate the selected rows
        X = pd.concat([class_1_rows, class_0_rows])
    
        # separate target column
        y = X['Class']
        X.drop(['Class'], inplace=True, axis=1)

        # handle missing values
        # print count of missing values per column
        # print(X.isnull().sum())
        
        # split training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # normalize 
        scaler = StandardScaler()
        for key in X_train.columns:
            X_train[key] = scaler.fit_transform(X_train[[key]])
            X_test[key] = scaler.fit_transform(X_test[[key]])

        # align columns
        X_test = X_test[X_train.columns]

        # convert to numpy
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy() 
        X_test  =  X_test.to_numpy() 
        y_test  = y_test.to_numpy()

        return X_train, y_train, X_test, y_test 


class LogisticRegression:
    def __init__(self, learning_rate=1, num_iterations=1000, threshold=0, num_features=None):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.threshold = threshold
        self.w = None
        self.b = None
        self.num_features = num_features
        self.feature_order = None

    def calculate_entropy(self, y):
        p, n = sum(1 for label in y if label == 1), sum(1 for label in y if label != 1)
        probabilities = np.array([p/(p+n), n/(p+n)])
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Adding a small value to avoid log(0)
        return entropy

    def information_gain(self, X, y, num_bins=10):
        initial_entropy = self.calculate_entropy(y)
        gains = []

        for col in range(X.shape[1]):
            num_unique_val = len(np.unique(X[:, col]))
            if num_unique_val <= 2:
                binned_X = self.perform_binning(X[:, col], num_bins)
                unique_values = np.unique(binned_X)
            else:
                unique_values = np.unique(X[:, col])

            weighted_entropy = 0
            for val in unique_values:
                if num_unique_val <= 2:
                    val_indices = np.where(binned_X == val)[0]
                else:
                    val_indices = np.where(X[:, col] == val)[0]

                val_entropy = self.calculate_entropy(y[val_indices])
                weighted_entropy += (len(val_indices) / len(y)) * val_entropy

            gain = initial_entropy - weighted_entropy
            gains.append(gain)

        return np.argsort(gains)[::-1]

    def perform_binning(self, column, num_bins):
        bins = np.array_split(np.sort(column), num_bins)
        binned_column = np.zeros(len(column), dtype=int)

        for idx, val in enumerate(column):
            for bin_idx, bin in enumerate(bins):
                if val <= bin[-1]:
                    binned_column[idx] = bin_idx
                    break

        return binned_column

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def calculate_error(self, y, y_predicted):
        return -np.mean(y*np.log2(y_predicted+1e-10)+(1-y)*np.log2(1-y_predicted+1e-10))

    def fit(self, X, y):
        # select feature
        if self.num_features is None:
            self.num_features = X.shape[1]

        self.feature_order = self.information_gain(X, y)
        selected_features = self.feature_order[:self.num_features]
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[selected_features] = True
        X = X[:, mask]
        # -------------

        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0

        for _ in range(self.num_iterations):
            Xw = np.dot(X, self.w) + self.b
            y_predicted = self.sigmoid(Xw)

            # Update weights and bias
            self.w += self.learning_rate * (1 / num_samples) * np.dot(X.T, (y - y_predicted))
            self.b += self.learning_rate * (1 / num_samples) * np.sum(y - y_predicted)
            
            if self.threshold != 0:
                error = self.calculate_error(y, self.sigmoid(np.dot(X, self.w) + self.b))
                if error < self.threshold:
                    break

    def predict(self, X):
        # select feature
        selected_features = self.feature_order[:self.num_features]
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[selected_features] = True
        X = X[:, mask]
        # -------------

        Xw = np.dot(X, self.w) + self.b
        y_predicted = self.sigmoid(Xw)
        y_predicted = np.where(y_predicted >= 0.5, 1, 0)
        return y_predicted


class AdaBoost:
    def __init__(self, num_iterations = 1000, num_features = None):
        self.hypothesis = None
        self.hypothesis_bias = None
        self.hypothesis_weights = None
        self.num_iterations = num_iterations
        self.num_features = num_features
        self.feature_order = None
    
    def fit(self, X, y, num_rounds):
        # select feature
        if self.num_features is None:
            self.num_features = X.shape[1]
        
        lr = LogisticRegression()
        self.feature_order = lr.information_gain(X, y)
        selected_features = self.feature_order[:self.num_features]
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[selected_features] = True
        X = X[:, mask]
        # -------------
    
        num_samples = X.shape[0]
        w = np.full(num_samples, 1 / num_samples)
        self.hypothesis = []
        self.hypothesis_bias = []
        self.hypothesis_weights = []

        for _ in range(num_rounds):
            sampled_indices = np.random.choice(num_samples, size=num_samples, replace=True, p=w)
            X_sampled = X[sampled_indices]
            y_sampled = y[sampled_indices]

            lr = LogisticRegression(num_iterations=self.num_iterations)
            lr.fit(X_sampled, y_sampled)
            y_predicted = lr.predict(X)

            error = np.sum(w * (y_predicted != y))
            
            if error > 0.5:
                continue

            error = max(error, 1e-10)
            for i in range(num_samples):
                if y_predicted[i]==y[i]:
                    w[i] *= error / (1-error)
            
            w /= np.sum(w)

            self.hypothesis.append(lr.w)
            self.hypothesis_bias.append(lr.b)
            self.hypothesis_weights.append(np.log2((1-error) / error))

        self.hypothesis = np.array(self.hypothesis)
        self.hypothesis_bias = np.array(self.hypothesis_bias)
        self.hypothesis_weights = np.array(self.hypothesis_weights)
    
    def predict(self, X):
        # select feature
        selected_features = self.feature_order[:self.num_features]
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[selected_features] = True
        X = X[:, mask]
        # -------------

        num_samples = X.shape[0]
        num_hypothesis = len(self.hypothesis)
        
        y_predicted = np.zeros(num_samples)

        for i in range(num_samples):
            weighted_sum = 0.0
            for j in range(num_hypothesis):
                w, b = self.hypothesis[j], self.hypothesis_bias[j]
                z = self.hypothesis_weights[j]
                xw = np.dot(X[i], w) + b
                y_pred = 1 if 1 / (1 + np.exp(-xw)) > 0.5 else -1
                weighted_sum += z * y_pred
            
            y_predicted[i] = 1 if weighted_sum > 0 else 0

        return y_predicted
    
            
class Evaluate:
    def __init__(self, y_true, y_predicted):
        self.true_positive = self.true_negative = self.false_positive = self.false_negative = 0
        for i in range(len(y_true)):
            if y_true[i]==1 and y_predicted[i]==1:
                self.true_positive+=1
            elif y_true[i]==0 and y_predicted[i]==0:
                self.true_negative+=1
            elif y_true[i]==0 and y_predicted[i]==1:
                self.false_positive+=1
            elif y_true[i]==1 and y_predicted[i]==0:
                self.false_negative+=1

    def accuracy(self):
        return (self.true_positive+self.true_negative)/(self.true_positive+self.true_negative+self.false_positive+self.false_negative+1e-10)
    
    def recall(self):
        return self.true_positive/(self.true_positive+self.false_negative+1e-10)
    
    def specifity(self):
        return self.true_negative/(self.true_negative+self.false_positive+1e-10)

    def precision(self):
        return self.true_positive/(self.true_positive+self.false_positive+1e-10)

    def false_discovery_rate(self):
        return 1 - self.precision()

    def f1(self):
        return 2*self.true_positive/(2*self.true_positive+self.false_negative+self.false_positive+1e-10)
    
    def print_metrics(self):
        print(
            f'Accuracy: {round(self.accuracy()*100, 2)}%\n'
            f'Sensitivity: {round(self.recall()*100, 2)}%\n'
            f'Specificity: {round(self.specifity()*100, 2)}%\n'
            f'Precision: {round(self.precision()*100, 2)}%\n'
            f'False Discovery Rate: {round(self.false_discovery_rate()*100, 2)}%\n'
            f'F1 Score: {round(self.f1()*100, 2)}%\n'
        )


def main():
    np.random.seed(1)

    print('*************** Logistic Regression ***************\n')

    for key in ['telco-customer-churn', 'adult', 'credit-card-fraud']:    
        print(f'-------------------- {key} --------------------\n')
        dataset = Dataset(key)
        X_train, y_train, X_test, y_test = dataset.preprocess()

        logistic_regression = LogisticRegression()
        logistic_regression.fit(X_train, y_train)
        
        print('Split : Train\n')
        y_predicted = logistic_regression.predict(X_train)
        evaluate = Evaluate(y_train, y_predicted)
        evaluate.print_metrics()

        print('Split : Test\n')
        y_predicted = logistic_regression.predict(X_test)
        evaluate = Evaluate(y_test, y_predicted)
        evaluate.print_metrics()


    print('*************** Ada Boost ***************\n')

    for key in ['telco-customer-churn', 'adult', 'credit-card-fraud']:    
        print(f'-------------------- {key} --------------------\n')
        dataset = Dataset(key)
        X_train, y_train, X_test, y_test = dataset.preprocess()

        for k in range(5, 25, 5):
            adaboost = AdaBoost(num_features=10, num_iterations=100)
            adaboost.fit(X_train, y_train, k)
            
            y_predicted = adaboost.predict(X_train)
            evaluate = Evaluate(y_train, y_predicted)
            print(f'Round: {k}   Split: Train   Accuracy: {round(evaluate.accuracy()*100, 2)}%')

            y_predicted = adaboost.predict(X_test)
            evaluate = Evaluate(y_test, y_predicted)
            print(f'Round: {k}   Split: Test   Accuracy: {round(evaluate.accuracy()*100, 2)}%')

        print()


if __name__ == "__main__":
    main()
    
