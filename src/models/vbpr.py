r"""
VBPR -- Recommended version
################################################
Reference:
VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback -Ruining He, Julian McAuley. AAAI'16
"""
import numpy as np
import os
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_normal_initialization
import torch.nn.functional as F


class VBPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader):
        super(VBPR, self).__init__(config, dataloader)

        # load parameters info
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton

        # define layers and loss
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))
        if self.v_feat is not None and self.t_feat is not None:
            self.item_raw_features = torch.cat((self.t_feat, self.v_feat), -1)
        elif self.v_feat is not None:
            self.item_raw_features = self.v_feat
        else:
            self.item_raw_features = self.t_feat
   
        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

        self.gamma = config['gamma']
        self.gamma_infer = config['gamma_infer']
        self.alpha = config['alpha']
        self.beta = config['beta']

        # user_history_item_feat and user_similarity_preference 
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        aver_image_his_feat_path = os.path.join(dataset_path, config['user_his_aver_image_feat'])
        self.aver_image_history_feat = torch.from_numpy(np.load(aver_image_his_feat_path)).type(torch.FloatTensor).to(
                    self.device)
        aver_text_his_feat_path = os.path.join(dataset_path, config['user_his_aver_text_feat'])
        self.aver_text_history_feat = torch.from_numpy(np.load(aver_text_his_feat_path)).type(torch.FloatTensor).to(
                    self.device)
        self.v_raw_norm_feat  = F.normalize(self.v_feat, p=2, dim=1)
        self.t_raw_norm_feat  = F.normalize(self.t_feat, p=2, dim=1)
      
        user_his_image_pre_path = os.path.join(dataset_path, config['user_his_image_pre'])
        user_his_text_pre_path = os.path.join(dataset_path, config['user_his_text_pre'])
        self.user_like_sim_image = torch.from_numpy(np.load(user_his_image_pre_path)).type(torch.FloatTensor).to(
                    self.device)
        self.user_like_sim_text = torch.from_numpy(np.load(user_his_text_pre_path)).type(torch.FloatTensor).to(
                    self.device)

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.u_embedding[user, :]

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding[item, :]

    def forward(self, dropout=0.0):
        item_feats_emb = self.item_linear(self.item_raw_features)
        item_embeddings = torch.cat((self.i_embedding, item_feats_emb), -1)

        user_e = F.dropout(self.u_embedding, dropout)
        item_e = F.dropout(item_embeddings, dropout)
        return user_e, item_e 


    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]
        
        user_embeddings, item_embeddings = self.forward()

        # user_history_item_feat
        user_his_image_feat_all = self.aver_image_history_feat
        user_his_image_feat = user_his_image_feat_all[user]
        pos_item_image_feat = self.v_raw_norm_feat[pos_item]
        neg_item_image_feat = self.v_raw_norm_feat[neg_item]

        user_his_text_feat_all = self.aver_text_history_feat
        user_his_text_feat = user_his_text_feat_all[user]
        pos_item_text_feat = self.t_raw_norm_feat[pos_item]
        neg_item_text_feat = self.t_raw_norm_feat[neg_item]

        # item-to-history similarity
        user_his_sim_image = self.user_like_sim_image[user]
        pos_v_his = torch.sum(user_his_image_feat * pos_item_image_feat, dim=1)
        neg_v_his = torch.sum(user_his_image_feat * neg_item_image_feat, dim=1)
        pos_v_pre = F.sigmoid(pos_v_his)
        neg_v_pre = F.sigmoid(neg_v_his)

        user_his_sim_text = self.user_like_sim_text[user]
        pos_t_his = torch.sum(user_his_text_feat * pos_item_text_feat, dim=1)
        neg_t_his = torch.sum(user_his_text_feat * neg_item_text_feat, dim=1)
        pos_t_pre = F.sigmoid(pos_t_his)
        neg_t_pre = F.sigmoid(neg_t_his)

        pos_m_pre = (pos_v_pre + pos_t_pre)/2
        neg_m_pre = (neg_v_pre + neg_t_pre)/2

        
        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        neg_e = item_embeddings[neg_item, :]
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)

        pos_score_with_m_pre = torch.mul(F.elu(pos_item_score) + 1, torch.pow(pos_m_pre, self.gamma))
        neg_score_with_m_pre = torch.mul(F.elu(neg_item_score) + 1, torch.pow(neg_m_pre, self.gamma))

        mf_loss = self.loss(pos_score_with_m_pre, neg_score_with_m_pre) # 一个数值
        reg_loss = self.reg_loss(user_e, pos_e, neg_e) # 一个数值
        loss = mf_loss + self.reg_weight * reg_loss 
        return loss
    
 
    # U-ISOLATOR   
    # def full_sort_predict(self, interaction):
    #     user = interaction[0]
    #     user_embeddings, item_embeddings = self.forward()
    #     user_e = user_embeddings[user, :]

    #     # user_history_item_feat
    #     user_his_image_feat_all = self.aver_image_history_feat
    #     user_his_image_feat = user_his_image_feat_all[user]
    #     user_his_text_feat_all = self.aver_text_history_feat
    #     user_his_text_feat = user_his_text_feat_all[user]

    #     # user_similarity_preference
    #     user_his_sim_image = self.user_like_sim_image[user]
    #     user_his_sim_text = self.user_like_sim_text[user]
    #     # item-to-history similarity
    #     match_t = torch.matmul(user_his_text_feat, self.t_raw_norm_feat.T)
    #     match_v = torch.matmul(user_his_image_feat, self.v_raw_norm_feat.T)

    #     d_t = match_t - user_his_sim_text.unsqueeze(1)
    #     d_v = match_v - user_his_sim_image.unsqueeze(1)

    #     user_v_pre = torch.sigmoid(torch.where(d_v > 0, torch.exp(-self.alpha * d_v), 4 / (1 + torch.exp(-self.beta * d_v)) - 1))
    #     user_t_pre = torch.sigmoid(torch.where(d_t > 0, torch.exp(-self.alpha * d_t), 4 / (1 + torch.exp(-self.beta * d_t)) - 1))

    #     user_m_pre = (user_t_pre + user_v_pre) / 2
    #     final_user_m_pre = torch.pow(user_m_pre, self.gamma_infer)

    #     all_item_e = item_embeddings
    #     score = (F.elu(torch.matmul(user_e, all_item_e.transpose(0, 1))) + 1) * final_user_m_pre
    #     return score
    
    # D-ISOLATOR
    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        self.restore_user_e, self.restore_item_e = user_embeddings, item_embeddings
        return score
 

