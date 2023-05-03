# Classification of Portuguese Students Grades

This project contains a solution for classifying students grades based on Bayesian Networks. The data used for this project was obtained from `[1]` and is available in the `data` folder. Additionally, the data are reduced compared to the original dataset.

## 1. Requirements

* Python 3.10.10
* Python-Packages listend in `requirements.txt`

## 2. Repository Structure

* `data` - contains the data used for this project
* `doc` - contains the documentation for this project
  * `doc/cpds` - contains the cpds for the final model
  * `doc/documentation.pdf` - contains additional documentation which is not covered by the jupyter notebooks. *(The documentation is written in German, for a translation please contact me)*
* `src` - contains the source code for this project and additional jupyter notebooks used for testing, data exploration and visualization
  * `src/app` - contains the source code for the application used for the final application
  * `src/helpers` - contains helper functions used for the visualization and data exploration
  * `src/analyse-data.ipynb` - contains the data exploration and correlation analysis
  * `src/model.ipynb` - contains the model creation and evaluation of three different models
  * `src/app.ipynb` - contains the final application
* `tasks` - contains the tasks for this project and the requirements for this project

## 3. Usage

For the usage of the application, open the `src/app.ipynb` notebook and execute the cells. If all requirements are installed, the application should start and you can use it.

## 4. Documentation and Report

The documentation and report for this project is available in the `doc` folder. The documentation is written in German and describes the process of creating the model and the application. Additionally, the documentation contains a description of the data and the used algorithms. In addition to the documentation, the jupyter notebooks contain additional information about the process of exploring the data and creating the model and the application.

## Sources

[1] P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In   A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.
