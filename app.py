import pandas as pd
import numpy as np
import streamlit as st
import time

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Decision Tree Visualizer",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded")

# load dataset
iris=datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)


# constants
min_weight_fraction_leaf=0.0
max_features = None
max_leaf_nodes = None
min_impurity_decrease=0.0




# Load initial graph
fig, ax = plt.subplots()

# Plot initial graph
scatter = ax.scatter(x.T[0], x.T[1], c=y, cmap='rainbow')
ax.set_xlabel(iris.feature_names[0], fontsize=10)
ax.set_ylabel(iris.feature_names[1],fontsize=10)
ax.set_title('Sepal Length vs Sepal Width', fontsize=15)
legend1 = ax.legend(*scatter.legend_elements(),
                     title="Classes",loc="upper right")
ax.add_artist(legend1)
ax.legend()
orig = st.pyplot(fig)

# sidebar elements 
st.sidebar.header(':blue[_Decision Tree_] Algo Visualizer', divider='rainbow')

criterion = st.sidebar.selectbox("Criterion",
                                ("gini", "entropy", "log_loss"),
                                help="""The function to measure the quality of a split.
                                Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” 
                                both for the Shannon information gain""")
max_depth = st.sidebar.number_input("Max Depth",
                            min_value=0,
                            max_value=30,
                            step=1,
                            value=0,
                            help="""The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure""")
if max_depth == 0:
    max_depth=None
min_samples_split = st.sidebar.number_input("Min Sample Split",
                                   min_value=0,
                                    max_value=x_train.shape[0],
                                   value=2,
                                   help="""The minimum number of samples required to split an internal node.
                                   If float, enter between 0 and 1""")
min_samples_leaf = st.sidebar.number_input("Min sample Leaf",
                                  min_value=0,
                                    max_value=x_train.shape[0],
                                  value=1,
                                  help="""The minimum number of samples required to be at a leaf node. 
                                  If float, enter between 0 and 1""")
random_state = st.sidebar.number_input("Random State",
                               min_value=0,
                              value=42)

# advance features
toggle = st.sidebar.toggle("Advance Features")

if toggle:
    min_weight_fraction_leaf = st.sidebar.number_input("Min Weight Fraction Leaf",
                                              min_value=0.0,
                                              max_value=1.0,
                                              value=0.0,
                                              help="""The minimum weighted fraction of the sum total of weights 
                                              (of all the input samples) required to be at a leaf node. """)
    max_features = st.sidebar.selectbox("Max Features",
                                       (None,"sqrt", "log2","Custom"),
                                       help="""The number of features to consider when looking for the best split""")
    if max_features == "Custom":
        max_features = st.sidebar.number_input("Enter Max Features",
                                      value=None,
                                      step=1)
    
    max_leaf_nodes = st.sidebar.number_input("Max Leaf Nodes",
                                            min_value=0,
                                            help="""Grow a tree with max_leaf_nodes in best-first fashion. """)
    if max_leaf_nodes==0:
        max_leaf_nodes=None
    min_impurity_decrease = st.sidebar.number_input("Min Impurity Decrase",
                                                   min_value=0.0,
                                                   help="""A node will be split if this split induces a decrease of the 
                                                   impurity greater than or equal to this value.""")
train = st.sidebar.button("Train Model", type="primary")
if st.sidebar.button("Reset"):
    st.experimental_rerun()
if train:
    orig.empty()

    msg = st.toast('Running', icon='🫸🏼')
    # building model
    clf = DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                max_features=max_features,
                                random_state=random_state,
                                max_leaf_nodes=max_leaf_nodes,
                                min_impurity_decrease=min_impurity_decrease)
    clf.fit(x_train[:, :2], y_train)
    x_pred = clf.predict(x_train[:,:2])
    y_pred = clf.predict(x_test[:, :2])
    st.subheader("Train Accuracy " + str(round(accuracy_score(y_train, x_pred), 2)) + ",  "+ "Test Accuracy  " + str(round(accuracy_score(y_test, y_pred), 2)))
    st.write("Total Depth:  " + str(clf.tree_.max_depth))
    

    # # define ranges for meshgrid
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundaries
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=20)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Decision Boundaries')
    plt.tight_layout()
    plt.savefig('decision_boundary_plot.png')
    plt.close()
    
    # Display decision boundary plot
    st.image("decision_boundary_plot.png")
    
    # Plot decision tree
    plt.figure(figsize=(25, 20))
    tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
    plt.xlim(plt.xlim()[0] * 2, plt.xlim()[1] * 2)
    plt.ylim(plt.ylim()[0] * 2, plt.ylim()[1] * 2)
    plt.savefig("decision_tree.png")
    plt.close()
    
    # Display decision tree plot
    st.image("decision_tree.png")

    msg.toast('Model run successfully!', icon='😎')


