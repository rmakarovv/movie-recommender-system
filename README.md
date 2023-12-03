<a name="readme-top"></a>

# ML Movie Recommender System

This repository contains source code and reports for the project on Practical ML and DL course.

	author: Roman Makarov
	group:  B-20-AI-01
	e-mail: o.makarov@innopolis.university

## The structure of the repository is as follows

    movie-recommender-system
    ├── README.md               # The top-level README
    │
    ├── data
    │   ├── interim             # Intermediate data that has been transformed.
    │   └── raw                 # The original data.
    │
    ├── models                  # Trained model weights.
    │
    ├── notebooks 
    │   ├── 1_0_initial_data_exporation.ipynb   # Data exploration and data preprocessing.
    │   └── 2_0_training_the_model.ipynb        # Model creation and training
    │
    ├── references              # References to used external sources.
    │
    ├── reports
    │   ├── figures             # Generated graphics that are used in the report.
    │   └── final_report.pdf    # Report containing data exploration, solution exploration, training process, and evaluation.
    │
    └── benchmark
        ├── data                # Data used for evaluation 
        └── evaluate.py         # Script that performs evaluation of the given model

## Solution
The final recommendation system is based on 4 layers of LightGCN with 256 embedding dimension. The benchmark results are as follows.
1. Normalized Discounted Cumulative Gain (NDCG) - **0.38** on test data
2. Mean Average Precision (MAP) - **0.17** on the test data

## The project can be run in the following steps:

0. Clone github repository with `git clone https://github.com/rmakarovv/pmldl_project.git`
1. Run `pip install requirements.txt`
2. Run/Review `1_0_initial_data_exporation.ipynb` to see the data exploration and data preprocessing ideas
3. Run/Review `2_0_training_the_model.ipynb` to train the model (this is a self-complete notebook and can be run without any other dependencies except those in `requirements.txt`)
4. Run `benchmark/evaluate.py` to test the model on Recall@K, Precision@K, NDCG@K, and MAP@K. You can run it in the following way 
        
        python benchmark/evaluate.py --data_path DATA_PATH --model_path PATH_TO_TRAINED_MODEL --latent_dim NUM_OF_LATENT_DIMS_IN_TRAINED_MODEL --num_layers NUM_LAYERS_IN_TRAINED_MODEL --K K_PARAMETER_FOR_METRICS

## Contact

RomanMakarov - o.makarov@innopolis.university

Project Link: [movie-recommender-system](https://github.com/rmakarovv/movie-recommender-system)

<p align="right">(<a href="#readme-top">back to top</a>)</p>