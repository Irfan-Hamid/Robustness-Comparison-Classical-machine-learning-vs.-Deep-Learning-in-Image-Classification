
Files
- `randomForest.py`: Implements the RandomForest model and evaluates its performance across eight distinct permutations.
- `alexNet.py`: Implements the AlexNet model and evaluates its performance across eight distinct permutations.
- `plotting.py`: Provides a detailed comparative analysis of the performances of both RandomForest and AlexNet across all permutations through graphical representations.


Prerequisite
- Python 3.10


Running the Models

1. RandomForest Model:
   - Navigate to the project directory.
   - Execute `python randomForest.py`.
   - Sequentially, the script will inquire if you wish to evaluate permutations (e.g., Perb1, Perb2, etc.). Respond accordingly to view the performance metrics for specific permutations.

2. AlexNet Model:
   - Navigate to the project directory.
   - Execute `python alexNet.py`.
   - Similar to the RandomForest model, this script will prompt you to select specific permutations for evaluating their performances.

Generating Comparative Graphs
- To visualize and compare the performance of both RandomForest and AlexNet across all permutations, execute:
  
  python plotting.py
  
  - This script is designed to comprehensively generate graphs for each permutation, providing a side-by-side comparison of the performance metrics between RandomForest and AlexNet. Generating these results may take some time

