from utils.data import *
from utils.metric import *
from argparse import ArgumentParser
import torch
import torch.utils.data as Data
from model.MSHNet import *
from model.loss import SLSIoULoss, L2IoULoss, L1IoULoss, L3IoULoss, L4IoULoss, IRSOIoULoss, L3WithDlossIoULoss, LLossOnlyLoss, SoftIoULossModule, FocalIoULoss, AverageMeter

from torch.optim import Adagrad
from tqdm import tqdm
import os
import os.path as osp
import time

os.environ['CUDA_VISIBLE_DEVICES']="0"

def parse_args():

    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of model')

    parser.add_argument('--dataset-dir', type=str, default=osp.join(osp.dirname(osp.abspath(__file__)), 'dataset', 'IRSTD-1k'))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--warm-epoch', type=int, default=5)

    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--multi-gpus', type=bool, default=False)
    parser.add_argument('--if-checkpoint', action='store_true',
                        help='从已有 checkpoint 恢复训练，需同时指定 --checkpoint-dir')
    parser.add_argument('--checkpoint-dir', type=str, default='',
                        help='恢复训练时权重目录，如 weight/MSHNet-dinov3-convnext-tiny-decoder-skip-L3-2026-02-22-07-48-01')

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--weight-path', type=str, default='weight/IRSTD-1k_weight.tar')
    parser.add_argument('--loss-type', type=str, default='L2', 
                        choices=['L1', 'L1-ONLY', 'L1_ONLY', 'L2', 'L2-ONLY', 'L2_ONLY', 'L3', 'L3_NO_LLOSS', 'L3-ONLY', 'L3_ONLY', 'L3D', 'L3+D', 'L3_D', 'L4', 'L4-ONLY', 'L4_ONLY', 'LLOSS-ONLY', 'LLOSS_ONLY', 'SOFTIOU', 'SOFT_IOU', 'IRSOIOU', 'IR-SOIOU', 'RSOIOU', 'IRSOIOU_LLOSS', 'IRSOIOU-LLOSS', 'IRSOIOU_LL', 'L1-FOCAL', 'L1_FOCAL', 'L2-FOCAL', 'L2_FOCAL', 'L3-FOCAL', 'L3_FOCAL', 'L4-FOCAL', 'L4_FOCAL'],
                        help='Loss function type: L1, L2, L3, L4, *-ONLY, LLOSS-ONLY, SOFTIOU, L3D, IRSOIOU, IRSOIOU_LLOSS')

    # Focal Loss 参数（配合 --loss-type focal 使用）
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='FocalIoU: focal 衰减指数 γ（越大越聚焦难样本，默认 2.0）')
    parser.add_argument('--focal-alpha', type=float, default=0.75,
                        help='FocalIoU: 正样本权重 α（目标像素权重，默认 0.75）')
    parser.add_argument('--focal-w', type=float, default=1.0,
                        help='FocalIoU: Focal BCE 相对于 L1-IoU 的权重系数（默认 1.0）')

    args = parser.parse_args()
    return args

class Trainer(object):
    def __init__(self, args):
        assert args.mode == 'train' or args.mode == 'test'

        self.args = args
        self.start_epoch = 0   
        self.mode = args.mode

        trainset = IRSTD_Dataset(args, mode='train')
        valset = IRSTD_Dataset(args, mode='val')
        print(f'Train samples: {len(trainset)} (trainval.txt)  |  Val samples: {len(valset)} (test.txt)')

        self.train_loader = Data.DataLoader(trainset, args.batch_size, shuffle=True, drop_last=False)
        self.val_loader = Data.DataLoader(valset, 1, drop_last=False)

        # 检查CUDA是否可用，如果不可用则使用CPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            try:
                print(f'Using GPU: {torch.cuda.get_device_name(0)}')
            except:
                print('Using GPU: Available')
        else:
            device = torch.device('cpu')
            print('CUDA not available, using CPU')
        self.device = device

        # Base MSHNet only (ResNet + CA + SA, no edge/SMA/LDRB/GP)
        model = MSHNet(3)
        print('Backbone: Base MSHNet')

        if args.multi_gpus:
            if torch.cuda.device_count() > 1:
                print('use '+str(torch.cuda.device_count())+' gpus')
                model = nn.DataParallel(model, device_ids=[0, 1])
        model.to(device)
        self.model = model

        self.optimizer = Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        self.down = nn.MaxPool2d(2, 2)
        self.full_res_deepsup = False

        # 根据参数选择损失函数
        loss_type = args.loss_type.upper()
        # 保存标准化的损失函数名称用于目录命名
        loss_name_for_dir = loss_type
        
        if loss_type in ('L1-FOCAL', 'L1_FOCAL', 'L2-FOCAL', 'L2_FOCAL', 'L3-FOCAL', 'L3_FOCAL', 'L4-FOCAL', 'L4_FOCAL'):
            # 从 loss_type 解析出 variant（L1 / L2 / L3 / L4）
            variant  = loss_type.replace('_', '-').split('-')[0]   # 'L1' / 'L2' / 'L3' / 'L4'
            gamma    = getattr(args, 'focal_gamma', 2.0)
            alpha    = getattr(args, 'focal_alpha', 0.75)
            focal_w  = getattr(args, 'focal_w',     1.0)
            self.loss_fun = FocalIoULoss(variant=variant, gamma=gamma, alpha=alpha, focal_w=focal_w)
            loss_name_for_dir = f'{variant}Focal'
            print(f'Using Loss: {variant}+FocalBCE (γ={gamma}, α={alpha}, w={focal_w})')
        elif loss_type == 'L1':
            self.loss_fun = L1IoULoss()
            print(f'Using Loss: L1 (L1-based IoU Loss)')
        elif loss_type == 'L1-ONLY' or loss_type == 'L1_ONLY':
            self.loss_fun = L1IoULoss()
            self.with_shape = False
            loss_name_for_dir = 'L1-ONLY'
            print(f'Using Loss: L1-ONLY (L1 IoU Loss without LLoss)')
        elif loss_type == 'L2':
            self.loss_fun = L2IoULoss()
            print(f'Using Loss: L2 (L2-based IoU Loss with LLoss)')
        elif loss_type == 'L2-ONLY' or loss_type == 'L2_ONLY':
            self.loss_fun = L2IoULoss()
            self.with_shape = False
            loss_name_for_dir = 'L2-ONLY'
            print(f'Using Loss: L2-ONLY (L2 IoU Loss without LLoss)')
        elif loss_type == 'L3':
            self.loss_fun = L3IoULoss()
            self.use_lloss_for_l3 = True  # 标记使用 LLoss
            loss_name_for_dir = 'L3'  # 默认使用 L3
            print(f'Using Loss: L3 (Mobius IoU Loss with LLoss)')
        elif loss_type == 'L3_NO_LLOSS' or loss_type == 'L3-ONLY' or loss_type == 'L3_ONLY':
            # L3 不使用 LLoss
            self.loss_fun = L3IoULoss()
            self.use_lloss_for_l3 = False  # 标记不使用 LLoss
            loss_name_for_dir = 'L3-ONLY'  # 使用 L3-ONLY 作为目录名
            print(f'Using Loss: L3 (Mobius IoU Loss without LLoss)')
        elif loss_type == 'L3D' or loss_type == 'L3+D' or loss_type == 'L3_D':
            self.loss_fun = L3WithDlossIoULoss()
            loss_name_for_dir = 'L3D'  # 统一使用 L3D 作为目录名
            print(f'Using Loss: L3+D_loss (L3 IoU Loss with D_loss from IR-SOIoU)')
        elif loss_type == 'L4':
            self.loss_fun = L4IoULoss()
            print(f'Using Loss: L4 (Man Fung IoU Loss)')
        elif loss_type == 'L4-ONLY' or loss_type == 'L4_ONLY':
            self.loss_fun = L4IoULoss()
            self.with_shape = False
            loss_name_for_dir = 'L4-ONLY'
            print(f'Using Loss: L4-ONLY (L4 IoU Loss without LLoss)')
        elif loss_type == 'LLOSS-ONLY' or loss_type == 'LLOSS_ONLY':
            self.loss_fun = LLossOnlyLoss()
            loss_name_for_dir = 'LLOSS-ONLY'
            print(f'Using Loss: LLOSS-ONLY (only Location Loss, no IoU — 用于看 location loss 带来的虚警)')
        elif loss_type == 'SOFTIOU' or loss_type == 'SOFT_IOU':
            self.loss_fun = SoftIoULossModule()
            loss_name_for_dir = 'SOFTIOU'
            print(f'Using Loss: SoftIoU (标准 Soft IoU Loss)')
        elif loss_type == 'IRSOIOU' or loss_type == 'IR-SOIOU' or loss_type == 'RSOIOU':
            # 默认使用 D_loss 版本
            self.loss_fun = IRSOIoULoss()
            loss_name_for_dir = 'IRSOIOU'  # 统一使用 IRSOIOU 作为目录名
            print(f'Using Loss: IR-SOIoU (Region Energy-Based Dynamic Loss with D_loss)')
        elif loss_type == 'IRSOIOU_LLOSS' or loss_type == 'IRSOIOU-LLOSS' or loss_type == 'IRSOIOU_LL':
            # 使用 LLoss 版本
            self.loss_fun = IRSOIoULoss()
            self.use_lloss_for_irsoiou = True  # 标记使用 LLoss
            loss_name_for_dir = 'IRSOIOU-LLOSS'  # 使用 IRSOIOU-LLOSS 作为目录名
            print(f'Using Loss: IR-SOIoU with LLoss (Region Energy-Based Dynamic Loss with LLoss)')
        else:
            self.loss_fun = L2IoULoss()
            loss_name_for_dir = 'L2'
            print(f'Unknown loss type {loss_type}, using default L2 (L2 IoU Loss)')
        
        # 保存损失函数名称用于目录命名
        self.loss_name_for_dir = loss_name_for_dir
        
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = mIoU(1)
        self.ROC  = ROCMetric(1, 10)
        self.best_iou = 0
        self.warm_epoch = args.warm_epoch

        if args.mode=='train':
            if args.if_checkpoint and getattr(args, 'checkpoint_dir', '').strip():
                check_folder = args.checkpoint_dir.strip()
                ckpt_path = osp.join(check_folder, 'checkpoint.pkl')
                if not osp.isfile(ckpt_path):
                    raise FileNotFoundError('恢复训练需要 checkpoint.pkl，未找到: %s' % ckpt_path)
                checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_iou = checkpoint.get('iou', 0)
                self.save_folder = check_folder
                print('从 checkpoint 恢复: %s, 从 epoch %d 继续, best_iou %.4f' % (check_folder, self.start_epoch, self.best_iou))
            else:
                timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                save_base = osp.join('weight', f'MSHNet-base-{self.loss_name_for_dir}-{timestamp}')
                self.save_folder = save_base
                if not osp.exists(self.save_folder):
                    os.makedirs(self.save_folder, exist_ok=True)
        if args.mode=='test':
            weight = torch.load(args.weight_path, map_location=self.device, weights_only=True)
            state = weight['state_dict'] if isinstance(weight, dict) and 'state_dict' in weight else \
                    weight['net'] if isinstance(weight, dict) and 'net' in weight else weight
            if not isinstance(state, dict):
                state = weight
            # 兼容训练时 DataParallel 保存的 "module." 前缀
            if isinstance(state, dict) and state and any(k.startswith('module.') for k in state.keys()):
                state = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in state.items()}
            load_ret = self.model.load_state_dict(state, strict=False)
            if load_ret.missing_keys or load_ret.unexpected_keys:
                print('[Test] load_state_dict 非严格匹配:')
                if load_ret.missing_keys:
                    print('  missing_keys (未加载，将用随机初始化):', load_ret.missing_keys[:8], '...' if len(load_ret.missing_keys) > 8 else '')
                if load_ret.unexpected_keys:
                    print('  unexpected_keys (被忽略):', load_ret.unexpected_keys[:8], '...' if len(load_ret.unexpected_keys) > 8 else '')
            '''
                # iou_67.87_weight
                weight = torch.load(args.weight_path)
                self.model.load_state_dict(weight)
            '''
            self.warm_epoch = -1
        

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()
        tag = False
        for i, (data, mask) in enumerate(tbar):
  
            data = data.to(self.device)
            labels = mask.to(self.device)

            if epoch>self.warm_epoch:
                tag = True

            masks, pred = self.model(data, tag)
            loss = 0
            # 论文 Eq.(8): L = (1/5)[ Σ_{i=1}^4 L_SLS(p_i, ↓(p_gt, 2^(4-i))) + L_SLS(p, p_gt) ]
            # 对应: pred↔p, masks[0]↔p_4(full), masks[1]↔p_3(↓2), masks[2]↔p_2(↓4), masks[3]↔p_1(↓8)；GT 用 MaxPool2d(2,2) 逐级下采样
            # 对于 IRSOIOU，根据 use_lloss_for_irsoiou 决定是否使用 LLoss
            # 对于 L3，根据 use_lloss_for_l3 决定是否使用 LLoss
            # 对于其他损失函数，使用默认的 with_shape=True
            if hasattr(self, 'use_lloss_for_irsoiou') and isinstance(self.loss_fun, IRSOIoULoss):
                with_shape = self.use_lloss_for_irsoiou
                loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch, with_shape=with_shape)
                for j in range(len(masks)):
                    if j > 0 and not self.full_res_deepsup:
                        labels = self.down(labels)
                    loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch, with_shape=with_shape)
            elif hasattr(self, 'use_lloss_for_l3') and isinstance(self.loss_fun, L3IoULoss):
                with_shape = self.use_lloss_for_l3
                loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch, with_shape=with_shape)
                for j in range(len(masks)):
                    if j > 0 and not self.full_res_deepsup:
                        labels = self.down(labels)
                    loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch, with_shape=with_shape)
            else:
                # 其他损失函数（L1/L2/L4 及其 *-ONLY）：with_shape 由 getattr 决定，ONLY 时为 False
                with_shape = getattr(self, 'with_shape', True)
                loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch, with_shape=with_shape)
                for j in range(len(masks)):
                    if j > 0 and not self.full_res_deepsup:
                        labels = self.down(labels)
                    loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch, with_shape=with_shape)
                
            loss = loss / (len(masks)+1)
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
       
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))
    
    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        tbar = tqdm(self.val_loader)
        tag = False
        with torch.no_grad():
            for i, (data, mask) in enumerate(tbar):
    
                data = data.to(self.device)
                mask = mask.to(self.device)

                if epoch>self.warm_epoch:
                    tag = True

                loss = 0
                _, pred = self.model(data, tag)
                # loss += self.loss_fun(pred, mask,self.warm_epoch, epoch)

                self.mIoU.update(pred, mask)
                self.PD_FA.update(pred, mask)
                self.ROC.update(pred, mask)
                _, mean_IoU = self.mIoU.get()

                tbar.set_description('Epoch %d, IoU %.4f' % (epoch, mean_IoU))
            FA, PD = self.PD_FA.get(len(self.val_loader))
            _, mean_IoU = self.mIoU.get()
            ture_positive_rate, false_positive_rate, _, _ = self.ROC.get()

            
            if self.mode == 'train':
                if mean_IoU > self.best_iou:
                    self.best_iou = mean_IoU
                
                    torch.save(self.model.state_dict(), self.save_folder+'/weight.pkl')
                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, self.best_iou, PD[0], FA[0] * 1000000))
                        
                all_states = {"net":self.model.state_dict(), "optimizer":self.optimizer.state_dict(), "epoch": epoch, "iou":self.best_iou}
                torch.save(all_states, self.save_folder+'/checkpoint.pkl')
            elif self.mode == 'test':
                print('mIoU: '+str(mean_IoU)+'\n')
                print('Pd: '+str(PD[0])+'\n')
                print('Fa: '+str(FA[0]*1000000)+'\n')


         
if __name__ == '__main__':
    args = parse_args()

    trainer = Trainer(args)
    
    if trainer.mode=='train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            trainer.test(epoch)
    else:
        trainer.test(1)
 