import pydotplus
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split,cross_val_score
iris = load_iris()

X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=2018)

clf = DecisionTreeClassifier(criterion="gini",
                             splitter="best",
                             max_depth=None,
                             min_samples_split=2,
                             min_samples_leaf=1,
                             min_weight_fraction_leaf=0.,
                             max_features=None,
                             random_state=None,
                             max_leaf_nodes=None,
                             class_weight=None,
                             presort=False)

dt_model = clf.fit(X_train,y_train)

print(cross_val_score(clf, iris.data, iris.target, cv=10))


from IPython.display import Image

dot_data = tree.export_graphviz(dt_model, out_file='tree.dot',
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)
print(dot_data)
graph = pydotplus.graph_from_dot_data(dot_data)
# graph = pydotplus.graph_from_dot_file('tree.dot')
Image(graph.create_png())