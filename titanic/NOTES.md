# Titanic Competition Notes

Notebook Studied: https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling

## Load and check data

-   Read csvs into train, test

-   Detect outliers using Turkey method

-   Drop outliers:
        train = train.drop(outliers, axis=0).reset_index(drop=True)

-   Join train and test set into dataset
        dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

-   Check for missing or null values


## Feature Analysis

-    Plot heatmap
        sns.heatmap(train[["Survived", "SibSp", "Parch", "Age", "Fare"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")

-   Fare feature has significant correlation with survival probability

-   Check subpopulations of other features for corellations with survival.

-   SibSP
        # TODO google those
        sns.factorplot(x="SibSp", y="Survived", data=train, kind="bar", size=6, palette="muted")
        g.despine(left=True)
        g = g.set_ylabels("survival probability")

    * Passengers having a lot of siblings/spouses have less chance to survive.
    * We can consider a new feature describing these categories.

-   Parch
    * same plot as above
    * Larger families have more chance to survive. Important standard deviation
    in the survival of passengers with 3 parents/children.

-   Age
        g = sns.FacetGrid(train, col='Survived')
        g = g.map(sns.distplot, "Age")

    * Seems to be a tailed distribution, maybe a gaussian distribution.
    * Age distributions not the same in the survived and not survived subpopulations.
    * So if "Age" is not correlated with "Survived" w e can see that there is
    age categories of passengers that have more or less chance to survive.

    * Further explore age distribution
        g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
        g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
        g.set_xlabel("Age")
        g.set_ylabel("Frequency")
        g = g.legend(["Not Survived","Survived"])
    * When we superimpose the two densities, we clearly see a peak corresponding to babies and very young childrens.

-   Fare
    * Fill missing values with the median
        dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
    * Explore distribution
        g = sns.distplot(dataset["Fare"], color="m", label="Skeeness: %.2f"%(dataset["Fare"].skew()))
        g = g.legend(loc="best")
    * The distribution is very skewed. This can lead to overweighting very high values in the model,
    even if it is scaled.
    * Better to transform it with the log function to reduce this skew
        dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    * Plot again and see that skewness is reduced.

-   Sex
        g = sns.barplot(x="Sex", y+"Survived", data=train)
        g = g.set_label("Survival probability")

    * It is clearly obvious that Males have less chance to survive than Females.

-   Pclass
        * Survived by pclass
            g = sns.factorplot(x="Pclass", y="Survived", data=train, kind="bar", size=6, palette="muted")
            g.despine(left=True)
            g = g.set_ylabels("survival probability")
        * Survived pclass vs survived by sex
            g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train, size=6, kind="bar", palette="muted")
            g.despine(left=True)
            g = g.set_ylabels("survival probability")

        * First class passengers have more chance to survive than second class and third class passengers.

-   Embarked
        dataset["Embarked"].isnull().sum()
        dataset["Embarked"] = dataset["Embarked"].fillna("S")

    * Hypothesis: more first class passengers coming from Cherbourg
        kid="count" on factorplot

## Filing missing values

-   Age
        * Since the re are subpopulations with more chance to survive, preferable to keep the age feature
        and impute the missing values.
        * fill missing values with similar rows according to pclass, parch and sibsp

## Feature Engineering

-   Name/Title
    * Keep title instead of name
    * filter catergories

-   Family Size
    * Combine sibsp and parch
    * Plot fsize
    * Create more features
        + Single
        + SmallF
        + MedF
        + LargeF

-   Cabin
    * Keep first letter of non null values
    * Fill null values with X

-   Ticket
    * extract ticket prefix, otherwise X

-   Create dummies for all categorical features
        * dataset = pd.get_dummies(dataset, columns=["Ticket"], prefix="T")
        etc

## Modeling

-   Separate train and test dataset

-   Drop survived from the test dataset

-   Set X, y, drop survived from X

## Simple Modeling

-   Cross validate models
    * Compare 10 popular classifiers and evaluate the mean accuracy of each one of them
    by a statified kfold cross validation procedure.
        + SVC
        + Decision Tree
        + AdaBoost
        + Random Forest
        + Extra Trees
        + Gradient Boosting
        + Multiple layer perceptron (neural net)
        + KNN
        + Logistic Regression
        + Linear Discriminant Analysis

-   kfold = StatifiedKFold(n_splits=10)

-   append each classifier to a classifiers list

-   for cl in classifiers:
        score = cross_val_score(classifier, X_train, y=Y_train, scoring="accuracy", cv=kfold, n_jobs=4)
        cv_means.append(score.mean())
        cv_std.append(score.std())

    # create dataframe with algorithms and results

    # plot barplot showing cross validation scores

-   Choose algorithms for ensembling

## Hyperparameter tuning for best models

-   Perform grid search optimization for AdaBoost, ExtraTrees, RandomForest, GradientBoosting and SVC
    to find the best parameters.

## Plotting learning curves

-   Plotting learning curves is a good way to see if we're overfitting the dataset.
-   Plot learning curves for all the above algorithms.

## Feature importance of tree based classifiers

-   In order to see the most informative features for the prediction of passengers survival,
    display the feature importance for the 4 tree based classifiers (Adaboost, ExtraTrees, RandomForest and GradientBoosting)

-   Note that the four classifiers have different top features according to the relative importance.
    This means that their predictions are not based on the same features. Nevertheless, they share common
    important features for the classification.

-   According to the feature importance of these 4 classifiers, the prediction of survival
    seems to be more associated with Age, Sex, the family size and the social standing of the passengers
    more than the location in the boat.

-   Concatanate test predictions, and plot correllation heatmap.
    * Predictions seem to be pretty similar except when adaboost is compared with the other classifiers.

## Ensemble modeling

-   Choose a voting classifier to combine the predictions coming from the 5 classifiers.
    * voting = 'soft' to take into account the probability of each vote.
