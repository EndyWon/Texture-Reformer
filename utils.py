import torch
import torch.nn as nn

# original vgg-19 model
from model import Encoder1, Encoder2, Encoder3, Encoder4, Encoder5
from model import Decoder1, Decoder2, Decoder3, Decoder4, Decoder5

# 16x compressed model
from small_model_16x import SmallEncoder1_16x_aux, SmallEncoder2_16x_aux, SmallEncoder3_16x_aux, SmallEncoder4_16x_aux, SmallEncoder5_16x_aux
from small_model_16x import SmallDecoder1_16x, SmallDecoder2_16x, SmallDecoder3_16x, SmallDecoder4_16x, SmallDecoder5_16x


class Reformer(nn.Module):
    def __init__(self, args):
        super(Reformer, self).__init__()
        self.args = args

        # load pre-trained models
        if args.compress:
            self.e1 = SmallEncoder1_16x_aux(args.e1); self.d1 = SmallDecoder1_16x(args.d1)
            self.e2 = SmallEncoder2_16x_aux(args.e2); self.d2 = SmallDecoder2_16x(args.d2)
            self.e3 = SmallEncoder3_16x_aux(args.e3); self.d3 = SmallDecoder3_16x(args.d3)
            self.e4 = SmallEncoder4_16x_aux(args.e4); self.d4 = SmallDecoder4_16x(args.d4)
            self.e5 = SmallEncoder5_16x_aux(args.e5); self.d5 = SmallDecoder5_16x(args.d5)
        else:
            self.e1 = Encoder1(args.e1); self.d1 = Decoder1(args.d1)
            self.e2 = Encoder2(args.e2); self.d2 = Decoder2(args.d2)
            self.e3 = Encoder3(args.e3); self.d3 = Decoder3(args.d3)
            self.e4 = Encoder4(args.e4); self.d4 = Decoder4(args.d4)
            self.e5 = Encoder5(args.e5); self.d5 = Decoder5(args.d5)
    
    # calculate channel-wise mean and standard deviation
    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std
    
    # project to standardize the data and dispel the domain gap
    def project(self, feat):
        size = feat.size()
        mean, std = self.calc_mean_std(feat)
        projected_feat = (feat - mean.expand(size)) / std.expand(size)
        return projected_feat

    # AdaIN
    def adaptive_instance_normalization(self, cF, sF):
        assert (cF.size()[:2] == sF.size()[:2])
        size = cF.size()
        style_mean, style_std = self.calc_mean_std(sF)
        content_mean, content_std = self.calc_mean_std(cF)

        normalized_feat = (cF - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    # WCT
    def whitening_and_coloring(self, cF, sF):
        cFSize = cF.size() # size: c * hw
        c_mean = torch.mean(cF, 1).unsqueeze(1).expand_as(cF)
        cF = cF - c_mean
        contentConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1)
        
        c_u, c_e, c_v = torch.svd(contentConv, some=False)

        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < 1e-5:
                k_c = i
                break

        sFSize = sF.size()
        s_mean = torch.mean(sF, 1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        styleConv = torch.mm(sF,sF.t()).div(sFSize[1] - 1)   
      
        s_u, s_e, s_v = torch.svd(styleConv, some=False)

        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < 1e-5:
                k_s = i
                break

        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        whiten_cF = torch.mm(step2, cF)

        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
        
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)

        return targetFeature


    # Semantic-Guided Texture Warping (SGTW) module
    def SGTW(self, sF, sF_fused, cF_fused, patch_size):
        # if patch_size = 0, set global view
        if patch_size == 0:
            patch_size = min([cF_fused.shape[2], cF_fused.shape[3], sF_fused.shape[2], sF_fused.shape[3]]) - 1

        # extract original style patches
        style_patches = sF.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
        style_patches = style_patches.permute(0, 2, 3, 1, 4, 5)
        style_patches = style_patches.reshape(-1, *style_patches.shape[-3:])

        # extract fused style patches
        fused_style_patches = sF_fused.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
        fused_style_patches = fused_style_patches.permute(0, 2, 3, 1, 4, 5)
        fused_style_patches = fused_style_patches.reshape(-1, *fused_style_patches.shape[-3:])

        # normalize fused style patches
        norm = torch.norm(fused_style_patches.reshape(fused_style_patches.shape[0], -1), dim=1).reshape(-1, 1, 1, 1)
        normalized_fused_style_patches = fused_style_patches/(norm + 1e-7) 
    
        # determine the closest-matching fused style patch for each fused content patch
        coordinate = torch.nn.functional.conv2d(cF_fused, normalized_fused_style_patches)
        
        # binarize the scores
        one_hots = torch.zeros_like(coordinate)
        one_hots.scatter_(1, coordinate.argmax(dim=1, keepdim=True), 1)

        # use the original style patches to reconstruct transformed feature
        deconv_out = torch.nn.functional.conv_transpose2d(one_hots, style_patches)

        # average the overlapped patches
        overlap = torch.nn.functional.conv_transpose2d(one_hots, torch.ones_like(style_patches))
        deconv_out = deconv_out / overlap

        return deconv_out
    

    # View-Specific Texture Reformation (VSTR) operation with *added* semantic guidance
    def VSTR_add(self, cF, sF, cF_sem, sF_sem, patch_size, alpha, semantic_weight):
        # project style feature and content feature
        sF1 = self.project(sF)
        cF1 = self.project(cF)

        # fuse with semantic maps
        sF_fused = (1-semantic_weight)*sF1 + semantic_weight*sF_sem
        cF_fused = (1-semantic_weight)*cF1 + semantic_weight*cF_sem
        
        # Semantic-Guided Texture Warping (SGTW) module
        targetFeature = self.SGTW(sF, sF_fused, cF_fused, patch_size)

        # blend transformed feature with the content feature
        targetFeature = targetFeature.view_as(cF)
        csF = alpha * targetFeature + (1.0 - alpha) * cF

        return csF


    # View-Specific Texture Reformation (VSTR) operation with *concatenated* semantic guidance
    def VSTR_concat(self, cF, sF, cF_sem, sF_sem, patch_size, alpha, semantic_weight):
        # project style feature and content feature
        sF1 = self.project(sF)
        cF1 = self.project(cF)

        # fuse with semantic maps
        sF_fused = torch.cat((sF1, semantic_weight*sF_sem), 1)
        cF_fused = torch.cat((cF1, semantic_weight*cF_sem), 1)
        
        # Semantic-Guided Texture Warping (SGTW) module
        targetFeature = self.SGTW(sF, sF_fused, cF_fused, patch_size)

        # blend transformed feature with the content feature
        targetFeature = targetFeature.view_as(cF)
        csF = alpha * targetFeature + (1.0 - alpha) * cF

        return csF
 

    # AdaIN transformation
    def adain(self, cF, sF, alpha):
        targetFeature = self.adaptive_instance_normalization(cF, sF)
        targetFeature = targetFeature.view_as(cF)
        csF = alpha * targetFeature + (1.0 - alpha) * cF
        return csF

    # WCT transformation
    def wct(self, cF, sF, alpha):
        cF = cF.double()
        sF = sF.double()
        C, W,  H  = cF.size(0), cF.size(1), cF.size(2)
        _, W1, H1 = sF.size(0), sF.size(1), sF.size(2)
        cFView = cF.view(C, -1)
        sFView = sF.view(C, -1)
        targetFeature = self.whitening_and_coloring(cFView, sFView)
        targetFeature = targetFeature.view_as(cF)
        csF = alpha * targetFeature + (1.0 - alpha) * cF
        csF = csF.float().unsqueeze(0)
        return csF
