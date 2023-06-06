
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

#1-D lattice total energy function evaluator class
#
class Features_selection:
    # Data = None
    # y = None
    path = None

    @classmethod  
    def fitnessFunc(cls,state):
        """
        Feature subset fitness function
        """

        # read dataframe from csv, y is target and X is the feature matrix
        data = pd.read_csv(cls.path, header = None)
        data.drop(data.index[[0,1]])

        # y = data[9]
        # Data = data.drop(data.columns[[9]], axis=1)

        y = data.iloc[:,-1]

        Data = data.drop(data.columns[[-1]], axis=1)


        if(state.count(0) != len(state)):
            # get index with value 0
            cols = [index for index in range(
                len(state)) if state[index] == 0]

            # get features subset
            X_parsed = Data.drop(Data.columns[cols], axis=1)
            X_subset = pd.get_dummies(X_parsed)

            # apply classification algorithm
            clf = LogisticRegression(solver='lbfgs')
            # clf = svm.SVC(kernel='linear', C=1)
           

            return sum(cross_val_score(clf, X_subset, y, cv=5))/5.0
        else:
            return(0,)




