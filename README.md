# CREDIT CARD FRAUD DETECTION AS A CLASSIFICATION PROBLEM
By Chimbudike Ohakim

Project Objective: The objective of my project is to develop and implement an effective online credit card fraud detection system using machine learning algorithms. Our aim was to leverage advanced analytics and predictive modeling techniques to accurately identify fraudulent transactions in real-time, thereby minimizing financial losses and enhancing security for both cardholders and financial institutions. By integrating cutting-edge technologies with comprehensive data analysis, our goal was to create a robust and scalable solution capable of adapting to evolving fraud patterns and ensuring the integrity of online transactions

Project background: This project involves Like any other technology that has been introduced in the world, the internet also comes with pros and cons. All of us enjoy the pros as the internet has changed our lifestyle by enhancing our communication. But, at the same time, we are witnessing digital frauds, which include fraudulent transactions through stolen credit cards. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced; the positive class (frauds) account for 0.172% of all transactions. The dataset has been collected and analyzed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Universite Libre de Bruxelles) on big data mining and fraud detection. More details on current and past projects on related topics are available on http://mlg.ulb.ac.be/BuFence and http://mlg.ulb.ac.be/ARTML.

Credit card Transaction data used as input for this model can be found using this link: https://drive.google.com/file/d/1XHUZauuFawrus3ZTAb6irwekqKvuWq_z/view?usp=share_link

# Execution Instructions

# Python version 3.10.4

To create a virtual environment and install requirements in Python 3.10.4 on different operating systems, follow the instructions below:

### For Windows:

Open the Command Prompt by pressing Win + R, typing "cmd", and pressing Enter.

Change the directory to the desired location for your project:


cd C:\path\to\project

Create a new virtual environment using the venv module:


python -m venv myenv

Activate the virtual environment:
myenv\Scripts\activate


Install the project requirements using pip:
pip install -r requirements.txt

### For Linux/Mac:
Open a terminal.

Change the directory to the desired location for your project:

cd /path/to/project

Create a new virtual environment using the venv module:

python3.10 -m venv myenv


Activate the virtual environment:
source myenv/bin/activate

Install the project requirements using pip:
pip install -r requirements.txt

These instructions assume you have Python 3.10.4 installed and added to your system's PATH variable.

## Execution Instructions if Multiple Python Versions Installed

If you have multiple Python versions installed on your system, you can use the Python Launcher to create a virtual environment with Python 3.10.4. Specify the version using the -p or --python flag. Follow the instructions below:

For Windows:
Open the Command Prompt by pressing Win + R, typing "cmd", and pressing Enter.

Change the directory to the desired location for your project:

cd C:\path\to\project

Create a new virtual environment using the Python Launcher:

py -3.10 -m venv myenv

Note: Replace myenv with your desired virtual environment name.

Activate the virtual environment:


myenv\Scripts\activate


Install the project requirements using pip:

pip install -r requirements.txt


### For Linux/Mac:
Open a terminal.

Change the directory to the desired location for your project:

cd /path/to/project

Create a new virtual environment using the Python Launcher:


python3.10 -m venv myenv


Note: Replace myenv with your desired virtual environment name.

Activate the virtual environment:

source myenv/bin/activate


Install the project requirements using pip:

pip install -r requirements.txt


By specifying the version using py -3.10 or python3.10, you can ensure that the virtual environment is created using Python 3.10.4 specifically, even if you have other Python versions installed.





