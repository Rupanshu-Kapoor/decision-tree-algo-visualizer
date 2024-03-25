# 📊Decision Tree Visualizer📈
See this app in action [here](https://huggingface.co/spaces/boringnose/Decision_Tree_Visualizer).

## Introduction

- Decision trees are a powerful machine learning algorithm for classification and regression tasks.
- They work by recursively splitting the dataset into smaller subsets based on certain features and conditions.
- This is a Streamlit web application that allows you to visualize a Decision Tree classifier in action.
- The app provides an interactive interface where you can customize various parameters of the Decision Tree model and observe its performance on the Iris dataset.

<br><br>

## Clone this on your local:

1. **Clone the Repository**: Clone this GitHub repository to your local machine using the following command:<br>

   `git clone https://github.com/Rupanshu-Kapoor/decision-tree-algo-visualizer.git`


2. **Install Dependencies**: Make sure you have Python installed on your local machine. Navigate to the directory where the repository is cloned
 and install the required Python dependencies using pip. You can do this by running the following command in your terminal or command prompt:<br>

    ```pip install -r requirements.txt```


3. **Run the Streamlit App**: Once the dependencies are installed, you can run the Streamlit app by executing the Python script `app.py`.
 You can do this by running the following command in your terminal or command prompt:<br>

     `streamlit run app.py`


4. **Access the App**: After running the Streamlit command, you can access the app in your web browser at the specified URL. 

<br><br>

## How to Use the App


1. **Run the app**: Ensure you have the app running.

2. **Observe initial graph**: The app displays an initial scatter plot of the dataset, showcasing the distribution of different classes.
3. **Customize model parameters** : Interact with the sidebar to adjust various hyperparameters of the decision tree model:
  - Criterion (gini, entropy, log_loss)
  - Max depth
  - Min samples split
  - Min samples leaf
  - Random state
  - Advanced features (optional)<br>
Train the model: Click the "Train Model" button to initiate model training with your chosen parameters.
4. **View results**: The app will display:

  - Accuracy scores on training and testing sets
  - Total depth of the generated decision tree
  - A plot of the decision boundaries created by the tree
  - A visual representation of the decision tree itself

<br> <br>

## How It Helped Me Learn

- **Interactive learning:** The visual feedback and ability to experiment with different hyperparameters solidified my understanding of decision trees.
- **Parameter influence:** Observing how changes in hyperparameters impacted the decision boundaries and tree structure enhanced my knowledge of their roles.
- **Algorithm visualization:** Witnessing the tree's decision-making process provided a clear understanding of its inner workings.
- **Practical hands-on experience:** Building a decision tree model and visualizing its results reinforced theoretical knowledge with practical application.

---
If you have any suggestions, feedback, please let me know. I welcome your thoughts and comments!

Thank You for visiting this repository
