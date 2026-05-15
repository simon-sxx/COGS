# coding=utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
import copy
from network.prob_graph import build_graph, bernonlli_sample, gumbel_sample, freeze_graph, cumulative_time_graph
from alg.cls_alg.get_nets import get_feat, get_encoder, get_decoder
from network.nets import classifier, NPChangeTransitionPrior, cnn_featurizer
from alg.cls_alg.algs.base import Algorithm
from alg.cls_alg.optimizer import build_optim
from network.loss import build_loss, Causal_Loss
from utils.utils import (
    feature_sel_txt,
    plot_feature_sel,
    score_dict_2_string,
    read_dict_from_csv,
    eval_ood_performance,
)

import torch.distributions as D

class COGS(Algorithm):
    """
    COGS: A Causal Representation Learning Framework for Out-of-Distribution Generalization in Time Series
    """

    def __init__(self, args, log, device="cuda"):
        super(COGS, self).__init__(args)
        self.args = args
        self.device = device
        self.epoch_i = 0
        self.total_step = 0
        self.log = log
        self.rec_loss_weight = args.rec_loss_weight
        
        self.graph_freezed = False
        self.z_dim = self.args.encoder.z_dim
        self.lag = args.lag
        self.kld_loss_weight = args.kld_loss_weight
        # environment inference params
        self.env_prototype_num = args.env_infer.env_prototype_num
        self.env_proto_dim = args.env_infer.env_proto_dim
        self.momentum = args.env_infer.momentum if hasattr(args.env_infer, 'momentum') else 0.8
        temp_start, temp_end = (
            args.env_infer.temperature_start,
            args.env_infer.temperature_end,
        )
        self.temp_gamma = (temp_end / temp_start) ** (1 / args.total_epoch)
        self.env_temperature = temp_start

        self.encoder = get_encoder(args)
        self.decoder = get_decoder(args)
        self.featurizer = get_feat(args)
        self.env_featurizer = cnn_featurizer(args)

        self.momentum_encoder = copy.deepcopy(self.env_featurizer)
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False  
        self.momentum_coefficient = 0.999 
    
        self.classifier = classifier(
            self.featurizer.out_dim, args.num_classes, args.classifier.type)
        
        self.embedding_dim = 8
        self.transition_prior = NPChangeTransitionPrior(lags=self.lag, 
                                                    latent_size=self.z_dim,
                                                    embedding_dim=self.embedding_dim, 
                                                    num_layers=3, 
                                                    hidden_dim=64)
        self.embedding = nn.Embedding(self.env_prototype_num, self.embedding_dim)

        graph_in_dim = self.args.encoder.z_dim
        graph_out_dim = self.args.num_classes
        self.causal_items = [str(i) for i in range(graph_in_dim)]
        self.graph = build_graph(self.args.graph_discov, graph_in_dim, graph_out_dim, self.device, mode="unify_pred")

        self.graph_discov_epoch = self.args.total_epoch if self.args.graph_discov.freeze_epoch > self.args.total_epoch else self.args.graph_discov.freeze_epoch
        end_tau, start_tau = (
            self.args.graph_discov.end_tau,
            self.args.graph_discov.start_tau,
        )
        self.gumbel_tau_gamma = (end_tau / start_tau) ** (1 / self.graph_discov_epoch)
        self.gumbel_tau = start_tau

        end_lmd, start_lmd = (
            self.args.graph_discov.lambda_s_end,
            self.args.graph_discov.lambda_s_start,
        )
        self.lambda_gamma = (end_lmd / start_lmd) ** (1 / self.graph_discov_epoch)
        self.lambda_s = start_lmd

        self.rec_loss = build_loss(self.args.rec_loss)
        self.cls_loss = build_loss(self.args.cls_loss)
        self.graph_loss = Causal_Loss(data_loss=self.cls_loss, norm_by_shape=self.args.graph_discov.norm_by_shape)

        vae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        cls_params = list(self.featurizer.parameters()) + list(self.classifier.parameters()) + list(self.transition_prior.parameters()) + list(self.embedding.parameters())
        self.optimizer_vae, self.scheduler_vae = build_optim(vae_params, self.args.vae_optimizer, self.args.total_epoch)
        self.optimizer_cls, self.scheduler_cls = build_optim(cls_params, self.args.cls_optimizer, self.args.total_epoch)
        self.optimizer_graph, self.scheduler_graph = build_optim(self.graph, self.args.graph_discov, self.graph_discov_epoch)
        self.optimizer_ei, self.scheduler_ei = build_optim(self.env_featurizer, self.args.env_optimizer, self.args.total_epoch)

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim))

        self.proto_initialized = False
        self.register_buffer("env_prototypes", torch.zeros(self.env_prototype_num, self.env_proto_dim))
        self.register_buffer("env_prototype_counts", torch.zeros(self.env_prototype_num))

    @property
    def base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z

    def _kld_loss(self, mu, logvar, z, domain_labels):
        batch_size, length, _ = mu.shape
        std = torch.exp(0.5 * logvar)
        # todo
        std = torch.clamp(std, min=1e-2, max=1e2)
        q_dist = D.Normal(mu, std)
        log_qz = q_dist.log_prob(z)
        mask_nan = torch.isnan(log_qz)
        mask_inf = torch.isinf(log_qz)
        if mask_nan.any() or mask_inf.any():
            log_qz = torch.where(mask_nan | mask_inf, torch.tensor(-1e10, device=log_qz.device), log_qz)
        # Past KLD
        p_dist = D.Normal(torch.zeros_like(mu[:, :self.lag]), torch.ones_like(logvar[:, :self.lag]))
        log_qz_normal = torch.sum(torch.sum(log_qz[:, :self.lag], dim=-1), dim=-1)
        log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(z[:, :self.lag]), dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()

        # Future KLD
        embeddings = self.embedding(domain_labels)
        log_qz_laplace = log_qz[:, self.lag:]
        residuals, logabsdet, hist_jac = self.transition_prior(z, embeddings)
        log_probs = self.base_dist.log_prob(residuals)
        mask_nan = torch.isnan(log_probs)
        mask_inf = torch.isinf(log_probs)
        if mask_nan.any() or mask_inf.any():
            log_probs = torch.where(mask_nan | mask_inf, torch.tensor(-1e10, device=log_probs.device), log_probs)
        result = torch.sum(log_probs, dim=1)
        log_pz_laplace = result + logabsdet
        log_qz_sum = torch.sum(log_qz_laplace, dim=[-2, -1])
    
        kld_laplace = (log_qz_sum - log_pz_laplace) / (length-self.lag)
        kld_laplace = kld_laplace.mean()

        kld_normal = torch.clamp(kld_normal, min=0.0) 
        kld_laplace = torch.clamp(kld_laplace, min=0.0)  


        kld_loss = kld_normal + kld_laplace

        return kld_loss


    def causal_masking(self, x, mask, ref):
        if isinstance(ref, str):
            if ref == "absent_feat":
                x_ref = torch.zeros_like(x)
                x_ref[:, :, 1] = 1
            elif ref == "zero":
                x_ref = torch.zeros_like(x)
            elif ref == "noise":
                x_ref = torch.randn_like(x)
        else:
            raise ValueError("ref should be absent_feat or zero or noise")
        if len(mask.shape) == 2:
            x_causal = torch.einsum("btn,bn->btn", x, mask) + torch.einsum("btn,bn->btn", x_ref, 1 - mask)
            x_non_causal = torch.einsum("btn,bn->btn", x, 1 - mask) + torch.einsum("btn,bn->btn", x_ref, mask)
        elif len(mask.shape) == 3:
            x_causal = torch.einsum("btn,btn->btn", x, mask) + torch.einsum("btn,btn->btn", x_ref, 1 - mask)
            x_non_causal = torch.einsum("btn,btn->btn", x, 1 - mask) + torch.einsum("btn,btn->btn", x_ref, mask)
        
        return x_causal, x_non_causal

    def vrex_loss_simple(self, outputs, labels, domain_labels, vrex_weight=1.0):
        """
        Variance penality
        """
        unique_domains = torch.unique(domain_labels)
        
        if len(unique_domains) < 2:
            return self.cls_loss(outputs, labels)

        domain_losses = []
        for domain in unique_domains:
            domain_mask = (domain_labels == domain)
            if domain_mask.sum() > 0:
                domain_outputs = outputs[domain_mask]
                domain_labels_true = labels[domain_mask]
                domain_loss = self.cls_loss(domain_outputs, domain_labels_true)
                domain_losses.append(domain_loss)
        
        if len(domain_losses) < 2:
            return domain_losses[0] if domain_losses else torch.tensor(0.0, device=self.device)
        
        domain_losses_tensor = torch.stack(domain_losses)
        average_risk = domain_losses_tensor.mean()
        risk_variance = domain_losses_tensor.var()
        
        vrex_loss = average_risk + vrex_weight * risk_variance
        
        return vrex_loss, average_risk, risk_variance


    def update(self, x, labels, domain_labels, **kwargs):
        self.encoder.train()
        self.decoder.train()
        self.featurizer.train()
        self.classifier.train()
        self.transition_prior.train()
        self.embedding.train()
        self.env_featurizer.eval()
        loss_dict = {}
        bs = x.shape[0]

        mu_logvar = self.encoder(x)
        mu = mu_logvar[:, :, :self.args.encoder.z_dim]
        logvar = mu_logvar[:, :, self.args.encoder.z_dim:]
        z = self.reparametrize(mu, logvar)
        x_recon = self.decoder(z)

        sampled_graph, prob_graph = gumbel_sample(
            self.graph, 
            bs, 
            tau=self.gumbel_tau, 
            t_length=self.args.t_length, 
            time_cumu_type=self.args.graph_discov.time_graph.time_cumu_type, 
            time_cumulative=self.args.graph_discov.time_graph.enable, 
            time_dim=self.args.graph_discov.time_graph.enable
        )
        z_causal, z_non_causal = self.causal_masking(mu, sampled_graph, "noise")

        zc = self.featurizer(z_causal.permute(0, 2, 1))
        out = self.classifier(zc)

        rec_loss = self.rec_loss(x, x_recon)
        vrex_results = self.vrex_loss_simple(out, labels, domain_labels, 
                                        vrex_weight=self.args.vrex_weight)
        
        if isinstance(vrex_results, tuple):
            cls_loss, average_risk, risk_variance = vrex_results
            loss_dict['average_risk'] = average_risk.item()
            loss_dict['risk_variance'] = risk_variance.item()
        else:
            cls_loss = vrex_results

        _, loss_sparsity = self.graph_loss(out, labels, prob_graph[0, ..., 0])
        kld_loss = self._kld_loss(mu, logvar, z, domain_labels)
        loss = cls_loss + self.lambda_s * loss_sparsity + rec_loss * self.rec_loss_weight + self.kld_loss_weight * kld_loss 

        if kld_loss.item() > 50.0 and self.epoch_i > 20:      
            loss_dict['total_loss'] = loss.item()
            loss_dict['class_loss'] = cls_loss.item()
            loss_dict['rec_loss'] = rec_loss.item()
            loss_dict['kld_loss'] = kld_loss.item()
            loss_dict['graph_sparsity_loss'] = loss_sparsity.item()
            return out, labels, loss_dict

        self.optimizer_vae.zero_grad()
        self.optimizer_cls.zero_grad()
        self.optimizer_graph.zero_grad()
        loss.backward()
        if hasattr(self.args, 'use_grad_clip') and self.args.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + 
                                        list(self.decoder.parameters()), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(list(self.featurizer.parameters()) + 
                                        list(self.classifier.parameters()) + 
                                        list(self.transition_prior.parameters()), max_norm=1.0)
    
        self.optimizer_vae.step()
        self.optimizer_cls.step()
        self.optimizer_graph.step()
        loss_dict['total_loss'] = loss.item()
        loss_dict['class_loss'] = cls_loss.item()
        loss_dict['rec_loss'] = rec_loss.item()
        loss_dict['kld_loss'] = kld_loss.item()
        loss_dict['graph_sparsity_loss'] = loss_sparsity.item()
        
        return out, labels, loss_dict

    def update_freeze_graph(self, x, labels, domain_labels, **kwargs):
        self.encoder.train()
        self.decoder.train()
        self.featurizer.train()
        self.classifier.train()
        self.transition_prior.train()
        self.embedding.train()
        self.env_featurizer.eval()
        loss_dict = {}
        bs = x.shape[0]

        mu_logvar = self.encoder(x)
        mu = mu_logvar[:, :, :self.args.encoder.z_dim]
        logvar = mu_logvar[:, :, self.args.encoder.z_dim:]
        z = self.reparametrize(mu, logvar)
        x_recon = self.decoder(z)

        sampled_graph = bernonlli_sample(
            self.graph, 
            bs, 
            prob=False, 
            hard_mask=True, 
            t_length=self.args.t_length, 
            threshold=self.args.graph_discov.graph_thresh, 
            time_cumu_type=self.args.graph_discov.time_graph.time_cumu_type, 
            time_cumulative=self.args.graph_discov.time_graph.enable
        )  
        z_causal, z_non_causal = self.causal_masking(mu, sampled_graph, "noise")

        zc = self.featurizer(z_causal.permute(0, 2, 1))
        out = self.classifier(zc)

        rec_loss = self.rec_loss(x, x_recon)
        vrex_results = self.vrex_loss_simple(out, labels, domain_labels, 
                                        vrex_weight=getattr(self.args, 'vrex_weight', 1.0))
        
        if isinstance(vrex_results, tuple):
            cls_loss, average_risk, risk_variance = vrex_results
            loss_dict['average_risk'] = average_risk.item()
            loss_dict['risk_variance'] = risk_variance.item()
        else:
            cls_loss = vrex_results

        kld_loss = self._kld_loss(mu, logvar, z, domain_labels)
        loss = cls_loss + rec_loss * self.rec_loss_weight + self.kld_loss_weight * kld_loss 

        self.optimizer_vae.zero_grad()
        self.optimizer_cls.zero_grad()
        loss.backward()
        self.optimizer_vae.step()
        self.optimizer_cls.step()
        loss_dict['total_loss'] = loss.item()
        loss_dict['class_loss'] = cls_loss.item()
        loss_dict['rec_loss'] = rec_loss.item()
        loss_dict['kld_loss'] = kld_loss.item()
        
        return out, labels, loss_dict

    def update_momentum_encoder(self):
        with torch.no_grad():
            for param_q, param_k in zip(self.env_featurizer.parameters(), 
                                    self.momentum_encoder.parameters()):
                param_k.data = self.momentum_coefficient * param_k.data + (1 - self.momentum_coefficient) * param_q.data


    def contrastive_loss(self, features, env_labels):
        batch_size = features.shape[0]
        similarity_matrix = torch.matmul(features, features.T) / self.env_temperature
        env_mask = (env_labels.unsqueeze(1) == env_labels.unsqueeze(0))
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        env_mask = env_mask & (~self_mask)

        positive_sim = similarity_matrix.masked_fill(~env_mask, -float('inf'))
        all_sim = similarity_matrix.masked_fill(self_mask, -float('inf'))
        
        valid_mask = env_mask.sum(dim=1) > 0
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        loss = -torch.logsumexp(positive_sim[valid_mask], dim=1) + torch.logsumexp(all_sim[valid_mask], dim=1)
        
        return loss.mean()

    def prototype_loss(self, features, zo_momentum, env_labels):
        batch_size = features.shape[0]
        
        all_similarities = torch.matmul(features, self.env_prototypes.T) / self.env_temperature
        momentum_similarities = torch.matmul(features, zo_momentum.T) / self.env_temperature

        labels_mask = F.one_hot(env_labels, num_classes=self.env_prototype_num).bool()
        
        positive_similarities = all_similarities.masked_select(labels_mask).view(batch_size, 1)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        positive_momentum_similarities = momentum_similarities.masked_select(self_mask).view(batch_size, 1)
        
        loss1 = -positive_similarities + torch.logsumexp(all_similarities, dim=1, keepdim=True)
        loss2 = - positive_momentum_similarities + torch.logsumexp(momentum_similarities, dim=1, keepdim=True)
        loss = loss1 + loss2

        return loss.mean()

    def pytorch_kmeans(self, features, k, max_iters=300, tol=1e-4):
        device = features.device
        n_samples, n_features = features.shape
        
        torch.manual_seed(42)  
        centroids_idx = self.kmeans_plus_plus_initialization(features, k)
        centroids = features[centroids_idx].clone()
        
        for iteration in range(max_iters):
            distances = torch.cdist(features, centroids)
            assignments = torch.argmin(distances, dim=1)

            new_centroids = torch.zeros_like(centroids)
            for i in range(k):
                mask = assignments == i
                if mask.sum() > 0:
                    new_centroids[i] = features[mask].mean(dim=0)
                else:
                    new_centroids[i] = centroids[i]

            if torch.norm(centroids - new_centroids) < tol:
                break
                
            centroids = new_centroids
        
        return centroids.detach()

    def global_em_update(self, train_loader_list):
        self.eval()
        dataset_features = []
        dataset_indices = []
        
        with torch.no_grad():
            for loader_idx, loader in enumerate(train_loader_list):
                loader_features = []
                loader_indices = []

                for batch in loader:
                    x = batch[0].transpose(1, 2).to(self.device).float()
                    indices = batch[-1].to(self.device).long()
                    mu_logvar = self.encoder(x)
                    mu = mu_logvar[:, :, :self.args.encoder.z_dim]
                    sampled_graph = bernonlli_sample(
                        self.graph, 
                        x.shape[0], 
                        prob=True, 
                        hard_mask=False, 
                        t_length=self.args.t_length, 
                        threshold=self.args.graph_discov.graph_thresh,
                        time_cumu_type=self.args.graph_discov.time_graph.time_cumu_type,
                        time_cumulative=self.args.graph_discov.time_graph.enable
                    )
                    _, z_non_causal = self.causal_masking(mu, sampled_graph, "zero")
                    
                    zo, _ = self.momentum_encoder(z_non_causal.permute(0, 2, 1))
                    projections = F.normalize(zo, dim=1)

                    loader_features.append(projections)
                    loader_indices.append(indices)

                dataset_features.append(torch.cat(loader_features, dim=0))
                dataset_indices.append(torch.cat(loader_indices, dim=0))

            all_features = torch.cat(dataset_features, dim=0)
            if all_features.shape[0] > 10000:
                indices = torch.randperm(all_features.shape[0])[:10000]
                features_subset = all_features[indices]
            else:
                features_subset = all_features

            old_prototypes = self.env_prototypes.clone()
            new_prototypes = self.pytorch_kmeans(features_subset, self.env_prototype_num)
            new_prototypes = F.normalize(new_prototypes, dim=1)

            self.env_prototypes.copy_(self.momentum * old_prototypes + (1 - self.momentum) * new_prototypes)
            self.env_prototypes = F.normalize(self.env_prototypes, dim=1)

            for loader_idx, (features, indices) in enumerate(zip(dataset_features, dataset_indices)):
                distances = torch.cdist(features, new_prototypes)
                env_labels = torch.argmin(distances, dim=1)

                env_labels_np = env_labels.cpu().numpy()
                indices_np = indices.cpu().numpy()
                
                train_loader_list[loader_idx].dataset.set_labels_by_index(env_labels_np, indices_np, 'domain_label')

    def independence_constraint(self, zc, zo):
        zc_centered = zc - zc.mean(dim=0, keepdim=True)
        zo_centered = zo - zo.mean(dim=0, keepdim=True)
        cross_corr = torch.mm(zc_centered.T, zo_centered) / (zc.shape[0] - 1)

        return torch.sum(cross_corr ** 2)

    def update_env_infer(self, x, domain_labels):
        self.env_featurizer.train()
        self.encoder.eval()
        self.decoder.eval()
        self.featurizer.eval()
        self.classifier.eval()
        self.transition_prior.eval()
        self.embedding.eval()

        loss_dict = {}
        bs = x.shape[0]
        
        with torch.no_grad():
            mu_logvar = self.encoder(x)
            mu = mu_logvar[:, :, :self.args.encoder.z_dim]
        sampled_graph, prob_graph = gumbel_sample(
            self.graph, 
            bs, 
            tau=self.gumbel_tau, 
            t_length=self.args.t_length, 
            time_cumu_type=self.args.graph_discov.time_graph.time_cumu_type, 
            time_cumulative=self.args.graph_discov.time_graph.enable, 
            time_dim=self.args.graph_discov.time_graph.enable
        )
        z_causal, z_non_causal = self.causal_masking(mu, sampled_graph, "noise")
        
        zo, _ = self.env_featurizer(z_non_causal.permute(0, 2, 1))
        with torch.no_grad():
            zc = self.featurizer(z_causal.permute(0, 2, 1))
        projections = F.normalize(zo, dim=1)
        with torch.no_grad():
            zo_momentum, _ = self.momentum_encoder(z_non_causal.permute(0, 2, 1))
            momentum_features = F.normalize(zo_momentum, dim=1)

        loss_sparsity = torch.norm(prob_graph[0, ..., 0], p=1)
        con_loss = self.contrastive_loss(projections, domain_labels)
        independence_loss = self.independence_constraint(zc, zo)
        proto_loss = self.prototype_loss(projections, momentum_features, domain_labels)
        loss = self.args.contras_loss_weight * con_loss + independence_loss * self.args.independence_weight + proto_loss + self.lambda_s * loss_sparsity

        self.optimizer_ei.zero_grad()
        self.optimizer_graph.zero_grad()
        loss.backward()
        if hasattr(self.args, 'use_grad_clip') and self.args.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.env_featurizer.parameters(), max_norm=1.0)
        
        self.optimizer_ei.step()
        self.optimizer_graph.step()
        self.update_momentum_encoder()

        loss_dict['env_contrastive_loss'] = con_loss.item()
        loss_dict['env_prototype_loss'] = proto_loss.item()
        loss_dict['independence_loss'] = independence_loss.item()
        loss_dict['env_total_loss'] = loss.item()
        
        return loss_dict

    def initialize_prototypes_from_data(self, train_loader_list):
        self.eval()
        all_features = []
        
        with torch.no_grad():
            for loader in train_loader_list:
                for batch in loader:
                    x = batch[0].transpose(1, 2).to(self.device).float()
                    mu_logvar = self.encoder(x)
                    mu = mu_logvar[:, :, :self.args.encoder.z_dim]
                    sampled_graph = bernonlli_sample(self.graph, x.shape[0], prob=True, hard_mask=False,
                                                t_length=self.args.t_length, threshold=self.args.graph_discov.graph_thresh,
                                                time_cumu_type=self.args.graph_discov.time_graph.time_cumu_type,
                                                time_cumulative=self.args.graph_discov.time_graph.enable)
                    _, z_non_causal = self.causal_masking(mu, sampled_graph, "zero")
                    features, _ = self.env_featurizer(z_non_causal.permute(0, 2, 1))
                    features = F.normalize(features, dim=1)
                    all_features.append(features)

        all_features = torch.cat(all_features, dim=0)
        indices = self.kmeans_plus_plus_initialization(all_features, self.env_prototype_num)
        protos = all_features[indices]
        self.env_prototypes.copy_(protos)

    def kmeans_plus_plus_initialization(self, features, k):
        n_samples = features.shape[0]
        device = features.device

        indices = torch.zeros(k, dtype=torch.long, device=device)
        indices[0] = torch.randint(0, n_samples, (1,), device=device)

        for i in range(1, k):
            cos_sim = torch.mm(features, features[indices[:i]].t())
            distances = 1.0 - cos_sim
            closest_dist = distances.min(dim=1)[0]

            weights = closest_dist ** 2
            if torch.sum(weights) < 1e-8:
                indices[i] = torch.randint(0, n_samples, (1,), device=device)
            else:
                weights = weights / torch.sum(weights)
                indices[i] = torch.multinomial(weights, 1)
        
        return indices

    def initialize_balanced_environments(self, train_loader_list):
        for loader_idx, loader in enumerate(train_loader_list):
            dataset = loader.dataset
            num_samples = len(dataset)

            samples_per_env = num_samples // self.env_prototype_num
            remainder = num_samples % self.env_prototype_num

            balanced_env_labels = []
            for env_id in range(self.env_prototype_num):
                env_count = samples_per_env
                if env_id < remainder:
                    env_count += 1
                balanced_env_labels.extend([env_id] * env_count)

            balanced_env_labels = np.array(balanced_env_labels)
            np.random.shuffle(balanced_env_labels)
            all_indices = np.arange(num_samples)
            dataset.set_labels_by_index(balanced_env_labels, all_indices, 'domain_label')

    def update_epoch(self, train_loader_list, epoch):
        self.train()
        self.epoch_i = epoch
        pred_epoch, label_epoch = [], []
        env_labels_epoch = []

        len_loader = np.inf
        for loader in train_loader_list:
            if len(loader) < len_loader:
                len_loader = len(loader)

        pbar = tqdm.tqdm(desc="train", total=len_loader)
        for batch_i, data_all in enumerate(zip(*train_loader_list)):   
            self.total_step += 1     
            x = torch.cat([data[0].transpose(1, 2).to(self.device).float() for data in data_all])     # [bs, ts_len, feat_num]
            labels = torch.cat([data[1].to(self.device).long() for data in data_all])            
            domain_labels = torch.cat([data[2].to(self.device).long() for data in data_all])  # [bs,]
            if epoch <= self.graph_discov_epoch:
                pred_batch, labels_batch, loss_dict = self.update(x, labels, domain_labels)
            else:
                self.graph_freezed = True
                pred_batch, labels_batch, loss_dict = self.update_freeze_graph(x, labels, domain_labels)
            pred_epoch.append(pred_batch.detach().cpu())
            label_epoch.append(labels_batch.detach().cpu())
            pbar.update(1)
        pbar.close()


        if not self.proto_initialized:
            self.initialize_prototypes_from_data(train_loader_list)
            self.proto_initialized = True
        pbar = tqdm.tqdm(desc="env inference", total=len_loader)
        for batch_i, data_all in enumerate(zip(*train_loader_list)):
            x = torch.cat([data[0].transpose(1, 2).to(self.device).float() for data in data_all])
            domain_labels = torch.cat([data[2].to(self.device).long() for data in data_all])
            env_loss_dict = self.update_env_infer(x, domain_labels)
            pbar.update(1)
        pbar.close()
        self.global_em_update(train_loader_list)
            

        self.scheduler_vae.step()
        self.scheduler_cls.step()
        self.scheduler_graph.step()
        self.scheduler_ei.step()
        self.gumbel_tau *= self.gumbel_tau_gamma
        self.lambda_s *= self.lambda_gamma
        self.env_temperature *= self.temp_gamma
        graph_lr = self.optimizer_graph.param_groups[0]["lr"]

        return pred_epoch, label_epoch
    

    def predict(self, x, mode="valid/", return_feat=False):
        self.eval()
        bs = x.shape[0]
        loss_dict = {}
        mu_logvar = self.encoder(x)
        mu = mu_logvar[:, :, :self.args.encoder.z_dim]
        if self.graph_freezed:
            prob = False
            hard_mask = True
        else:
            prob = False
            hard_mask = True

        sampled_graph = bernonlli_sample(
            self.graph, 
            bs, 
            prob=prob, 
            hard_mask=hard_mask, 
            t_length=self.args.t_length, 
            threshold=self.args.graph_discov.graph_thresh, 
            time_cumu_type=self.args.graph_discov.time_graph.time_cumu_type, 
            time_cumulative=self.args.graph_discov.time_graph.enable
        )         
        z_causal, z_non_causal = self.causal_masking(mu, sampled_graph, "zero") 
        feat = self.featurizer(z_causal.permute(0, 2, 1))
        out = self.classifier(feat)

        if return_feat:
            return out, feat, loss_dict
        else:
            return out, loss_dict
    
    def save_graph(self):
        torch.save(self.graph, os.path.join(self.log.log_dir, f"iter_{self.epoch_i:d}", "graph.pt"))

    def load_model(self, load_dir):
        model_dir = os.path.join(load_dir, "model.pt")
        self.load_state_dict(torch.load(model_dir, map_location=self.device))

    def load_graph(self, load_dir):
        self.graph = torch.load(os.path.join(load_dir, "graph.pt")).to(self.device)