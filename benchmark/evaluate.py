import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
from tqdm.notebook import tqdm
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

import warnings
warnings.filterwarnings("ignore")

# Set up reproducibility settings
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Set up argparsers
def parse_args():
    parser = argparse.ArgumentParser(description="Movie Recommender System")
    parser.add_argument("--data_path", type=str, default="movie-recommender-system/benchmark/data/df.csv",
                        help="Path to the CSV file containing user-item interactions data")
    parser.add_argument("--model_path", type=str, default="movie-recommender-system/models/model.h5",
                        help="Path to the pre-trained model")
    parser.add_argument("--latent_dim", type=int, default=64, help="Dimension of latent embeddings")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of graph convolution layers")
    parser.add_argument("--K", type=int, default=20, help="Top K recommendations for evaluation")

    return parser.parse_args()


class LightGCNConv(MessagePassing):
    """
    LightGCNConv: Graph Convolutional Layer for LightGCN Model.

    Parameters:
        aggr (str): Aggregation method for message passing. Default is 'add'.
    """

    def __init__(self, aggr='add'):
        super().__init__(aggr=aggr)

    def forward(self, x, edge_index):
        """
        Forward pass for LightGCNConv.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph edge indices.

        Returns:
            torch.Tensor: Output tensor after message passing.
        """
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        """
        Message function for LightGCNConv.

        Args:
            x_j (torch.Tensor): Input tensor from neighboring nodes.
            norm (torch.Tensor): Normalization tensor.

        Returns:
            torch.Tensor: Scaled input tensor for aggregation.
        """
        return norm.view(-1, 1) * x_j


class RecSysGNN(nn.Module):
    def __init__(self, latent_dim, num_layers, num_users, num_items):
        """
        Constructor for RecSysGNN.

        Args:
            latent_dim (int): Dimension of the latent embeddings.
            num_layers (int): Number of graph convolution layers.
            num_users (int): Number of unique user IDs.
            num_items (int): Number of unique item IDs.
        """
        super(RecSysGNN, self).__init__()
        self.embedding = nn.Embedding(num_users + num_items, latent_dim)
        self.convs = nn.ModuleList(LightGCNConv() for _ in range(num_layers))
        self.init_parameters()

    def init_parameters(self):
        """
        Initialize model parameters.

        Authors of LightGCN report higher results with normal initialization.
        """
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index):
        """
        Forward pass for RecSysGNN.

        Args:
            edge_index (torch.Tensor): Graph edge indices.

        Returns:
            tuple: Tuple containing original embeddings and final output.
        """
        emb0 = self.embedding.weight
        embs = [emb0]

        emb = emb0
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)

        out = torch.mean(torch.stack(embs, dim=0), dim=0)

        return emb0, out

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        """
        Encode a minibatch for training.

        Args:
            users (torch.Tensor): User indices.
            pos_items (torch.Tensor): Positive item indices.
            neg_items (torch.Tensor): Negative item indices.
            edge_index (torch.Tensor): Graph edge indices.

        Returns:
            tuple: Tuple containing embeddings for users, positive items, negative items,
            original embeddings for users, positive items, negative items.
        """
        emb0, out = self(edge_index)
        return (
            out[users],
            out[pos_items],
            out[neg_items],
            emb0[users],
            emb0[pos_items],
            emb0[neg_items]
        )


def compute_bpr_loss(users, users_emb, pos_emb, neg_emb, user_emb0, pos_emb0, neg_emb0):
    """
    Compute BPR loss and regularization loss.

    Args:
        users (torch.Tensor): User indices.
        users_emb (torch.Tensor): User embeddings.
        pos_emb (torch.Tensor): Positive item embeddings.
        neg_emb (torch.Tensor): Negative item embeddings.
        user_emb0 (torch.Tensor): Original user embeddings.
        pos_emb0 (torch.Tensor): Original positive item embeddings.
        neg_emb0 (torch.Tensor): Original negative item embeddings.

    Returns:
        tuple: Tuple containing BPR loss and regularization loss.
    """
    # Compute regularization loss from initial embeddings
    reg_loss = (1 / 2) * (
        user_emb0.norm().pow(2) +
        pos_emb0.norm().pow(2) +
        neg_emb0.norm().pow(2)
    ) / float(len(users))

    # Compute BPR loss from user, positive item, and negative item embeddings
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)

    bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

    return bpr_loss, reg_loss


def convert_to_sparse_tensor(dok_mtrx):
    """
    Convert a scipy.sparse matrix to a sparse PyTorch tensor.

    Args:
        dok_mtrx (scipy.sparse.dok_matrix): Input sparse matrix in DOK format.

    Returns:
        torch.sparse.FloatTensor: Sparse PyTorch tensor.
    """
    # Convert DOK matrix to COO format and cast to float32
    dok_mtrx_coo = dok_mtrx.tocoo().astype(np.float32)

    # Extract values and indices
    values = dok_mtrx_coo.data
    indices = np.vstack((dok_mtrx_coo.row, dok_mtrx_coo.col))

    # Convert indices and values to PyTorch tensors
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)

    # Get the shape of the sparse tensor
    shape = dok_mtrx_coo.shape

    # Create the sparse PyTorch tensor
    dok_mtrx_sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return dok_mtrx_sparse_tensor


def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, train_data, test_data, K):
    """
    Compute recommendation metrics including recall, precision, nDCG, and MAP.

    Args:
        user_Embed_wts (torch.Tensor): User embedding weights.
        item_Embed_wts (torch.Tensor): Item embedding weights.
        n_users (int): Number of unique users.
        n_items (int): Number of unique items.
        train_data (pd.DataFrame): Training data.
        test_data (pd.DataFrame): Test data.
        K (int): Top-K recommendations.

    Returns:
        tuple: Tuple containing mean recall, precision, nDCG, and MAP.
    """
    # Initialize user and item embeddings using provided weights
    user_Embedding = nn.Embedding(user_Embed_wts.size()[0], user_Embed_wts.size()[1], _weight=user_Embed_wts)
    item_Embedding = nn.Embedding(item_Embed_wts.size()[0], item_Embed_wts.size()[1], _weight=item_Embed_wts)

    # Extract unique user IDs from the test data
    test_user_ids = torch.LongTensor(test_data['user_id_idx'].unique())

    # Compute relevance scores using matrix multiplication
    relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1))

    # Create a sparse matrix R based on training data interactions
    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    R[train_data['user_id_idx'], train_data['item_id_idx']] = 1.0

    # Convert sparse matrix R to a sparse PyTorch tensor
    R_tensor = convert_to_sparse_tensor(R)
    R_tensor_dense = R_tensor.to_dense()

    # Set interactions in R_tensor_dense to negative infinity
    R_tensor_dense = R_tensor_dense * (-np.inf)
    R_tensor_dense = torch.nan_to_num(R_tensor_dense, nan=0.0).to(device)

    # Add R_tensor_dense to relevance scores
    relevance_score = relevance_score + R_tensor_dense

    # Get top-K relevance scores and indices
    topk_relevance_score = torch.topk(relevance_score, K).values
    topk_relevance_indices = torch.topk(relevance_score, K).indices

    # Create a DataFrame for top-K relevance indices
    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(),
                                              columns=['top_indx_' + str(x + 1) for x in range(K)])

    topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index

    # Create a column with top-K relevant items
    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[
        ['top_indx_' + str(x + 1) for x in range(K)]].values.tolist()
    topk_relevance_indices_df = topk_relevance_indices_df[['user_ID', 'top_rlvnt_itm']]

    # Group test data by user ID and extract lists of interacted items
    test_interacted_items = test_data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()

    # Merge top-K relevance indices with test interacted items
    metrics_df = pd.merge(test_interacted_items, topk_relevance_indices_df, how='left', left_on='user_id_idx',
                          right_on=['user_ID'])
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in
                                  zip(metrics_df.item_id_idx, metrics_df.top_rlvnt_itm)]

    # Compute recall and precision
    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / len(x['item_id_idx']), axis=1)
    metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / K, axis=1)

    # Function to get a hit list indicating whether each top-K item is interacted or not
    def get_hit_list(item_id_idx, top_rlvnt_itm):
        return [1 if x in set(item_id_idx) else 0 for x in top_rlvnt_itm]

    metrics_df['hit_list'] = metrics_df.apply(lambda x: get_hit_list(x['item_id_idx'], x['top_rlvnt_itm']), axis=1)

    # Function to compute DCG and IDCG
    def get_dcg_idcg(item_id_idx, hit_list):
        idcg = sum([1 / np.log1p(idx + 1) for idx in range(min(len(item_id_idx), len(hit_list)))])
        dcg = sum([hit / np.log1p(idx + 1) for idx, hit in enumerate(hit_list)])
        return dcg / idcg

    # Function to compute cumulative sum of hit list
    def get_cumsum(hit_list):
        return np.cumsum(hit_list)

    # Function to compute MAP
    def get_map(item_id_idx, hit_list, hit_list_cumsum):
        return sum([hit_cumsum * hit / (idx + 1) for idx, (hit, hit_cumsum) in
                    enumerate(zip(hit_list, hit_list_cumsum))]) / len(item_id_idx)

    # Compute nDCG, hit list cumulative sum, and MAP
    metrics_df['ndcg'] = metrics_df.apply(lambda x: get_dcg_idcg(x['item_id_idx'], x['hit_list']), axis=1)
    metrics_df['hit_list_cumsum'] = metrics_df.apply(lambda x: get_cumsum(x['hit_list']), axis=1)
    metrics_df['map'] = metrics_df.apply(lambda x: get_map(x['item_id_idx'], x['hit_list'], x['hit_list_cumsum']),
                                         axis=1)

    # Return mean values of recall, precision, nDCG, and MAP
    return metrics_df['recall'].mean(), metrics_df['precision'].mean(), metrics_df['ndcg'].mean(), metrics_df['map'].mean()


def test_and_eval(model, train_df, test_df, train_edge_index, n_users, n_items, K):
    """
    Test the model and evaluate its performance using recommendation metrics.

    Args:
        model (RecSysGNN): The recommendation system GNN model.
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        train_edge_index (torch.Tensor): Edge index for training.
        n_users (int): Number of unique users.
        n_items (int): Number of unique items.
        K (int): Top-K recommendations.

    Returns:
        tuple: Tuple containing rounded values of recall, precision, nDCG, and MAP.
    """
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation during evaluation
    with torch.no_grad():
        # Forward pass to obtain final user and item embeddings
        _, out = model(train_edge_index)
        final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))

        # Get recommendation metrics
        test_topK_recall, test_topK_precision, test_topK_ndcg, test_topK_map = get_metrics(
            final_user_Embed, final_item_Embed, n_users, n_items, train_df, test_df, K
        )

    # Return rounded values of metrics
    return (
        round(test_topK_recall, 4),
        round(test_topK_precision, 4),
        round(test_topK_ndcg, 4),
        round(test_topK_map, 4)
    )


def preprocess_data(path):
    # Load data
    columns_name = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(path, sep=",").iloc[:, 1:]
    df = df[df['rating'] >= 3]

    # Split data to train and test
    train, test = train_test_split(df.values, test_size=0.2, random_state=SEED)
    train_df = pd.DataFrame(train, columns=df.columns)
    test_df = pd.DataFrame(test, columns=df.columns)

    # Label encode user and item IDs
    le_user = pp.LabelEncoder()
    le_item = pp.LabelEncoder()
    train_df['user_id_idx'] = le_user.fit_transform(train_df['user_id'].values)
    train_df['item_id_idx'] = le_item.fit_transform(train_df['item_id'].values)

    # Process test data
    train_user_ids = train_df['user_id'].unique()
    train_item_ids = train_df['item_id'].unique()

    test_df = test_df[
        (test_df['user_id'].isin(train_user_ids)) & (test_df['item_id'].isin(train_item_ids))
    ]

    test_df['user_id_idx'] = le_user.transform(test_df['user_id'].values)
    test_df['item_id_idx'] = le_item.transform(test_df['item_id'].values)

    n_users = train_df['user_id_idx'].nunique()
    n_items = train_df['item_id_idx'].nunique()

    # Define and check train_edge_index
    u_t = torch.LongTensor(train_df.user_id_idx)
    i_t = torch.LongTensor(train_df.item_id_idx) + n_users

    train_edge_index = torch.stack((
        torch.cat([u_t, i_t]),
        torch.cat([i_t, u_t])
    )).to(device)

    # Assert statements for train_edge_index
    assert train_edge_index[:, -1].cpu().numpy()[::-1].all() == train_edge_index[:, len(train) - 1].cpu().numpy().all()
    assert train_edge_index[:, 0].cpu().numpy()[::-1].all() == train_edge_index[:, len(train)].cpu().numpy().all()

    return train_df, test_df, train_edge_index, n_users, n_items


def main():
    # Parse arguments for training and testing
    args = parse_args()

    # Preprocess data 
    train_df, test_df, train_edge_index, n_users, n_items = preprocess_data(args.data_path)

    # Load pre-trained model
    model = RecSysGNN(
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_users=n_users,
        num_items=n_items,
    ).to(device)

    model.load_state_dict(torch.load(args.model_path))

    # Test and evaluate the model
    light_recall, light_precision, light_ndcg, light_map = test_and_eval(model, train_df, test_df, train_edge_index, n_users, n_items, args.K)

    # Print evaluation results
    print(f'Recall@{args.K}:     {light_recall:.2f}')
    print(f'Precision@{args.K}:  {light_precision:.2f}')
    print(f'NDCG@{args.K}:       {light_ndcg:.2f}')
    print(f'MAP@{args.K}:        {light_map:.2f}')


if __name__ == "__main__":
    main()
