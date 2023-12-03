# ML Movie Recommender System

This repository is contains source code and reports for the assignment 2 on Practical ML and DL course.

	author: Roman Makarov
	group:  B-20-AI-01
	e-mail: o.makarov@innopolis.university

Look at `reports/final_report.pdf` for detailed explanation of the solution.

## The best results obtained are:
1. Normalized Discounted Cumulative Gain (NDCG) - 0.38 on test data.
2. Mean Average Precision (MAP) - 0.17 on the test data.

## The project can be run in the following steps:

0. Clone github repository with `git clone https://github.com/rmakarovv/pmldl_project.git`
1. Run `pip install requirements.txt`
2. Run/Review `1_0_initial_data_exporation.ipynb` to see the data exploration and data preprocessing ideas.
3. Run/Review `2_0_training_the_model.ipynb` to train the model (this is self-complete notebook and can be run without any other dependencies except those in `requirements.txt`).
4. Run `benchmark/evaluate.py` to test the model on Recall@K, Precision@K, NDCG@K, and MAP@K. You can run it in the following way 
        
        python benchmark/evaluate.py --data_path DATA_PATH --model_path PATH_TO_TRAINED_MODEL --latent_dim NUM_OF_LATENT_DIMS_IN_TRAINED_MODEL --num_layers NUM_LAYERS_IN_TRAINED_MODEL --K K_PARAMETER_FOR_METRICS