from utils.data import *
from utils.metric import *
from argparse import ArgumentParser
import torch
import torch.utils.data as Data
from model.MSHNet import *
from model.loss import (
    SLSIoULoss, L2IoULoss, L1IoULoss, L3IoULoss, L4IoULoss,
    IRSOIoULoss, L3WithDlossIoULoss,
    LLossOnlyLoss, SoftIoULossModule, AverageMeter,
)
from torch.optim import Adagrad
from tqdm import tqdm
import os.path as osp
import time

os.environ['CUDA_VISIBLE_DEVICES']="0"

def parse_args():

    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of model')

    parser.add_argument('--dataset-dir', type=str, default='dataset/IRSTD-1k',
                        help='Dataset path; IRSTD-1k included in repo')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--warm-epoch', type=int, default=5)

    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--multi-gpus', type=bool, default=False)
    parser.add_argument('--if-checkpoint', type=bool, default=False)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--weight-path', type=str, default='weight/IRSTD-1k_weight.tar')
    _loss_choices = ['L1', 'L1-ONLY', 'L1_ONLY', 'L2', 'L2-ONLY', 'L2_ONLY', 'L3', 'L3-ONLY', 'L3_ONLY',
                     'L3D', 'L3+D', 'L3_D', 'L4', 'L4-ONLY', 'L4_ONLY',
                     'IRSOIOU', 'IR-SOIOU', 'RSOIOU', 'IRSOIOU_LLOSS', 'IRSOIOU-LLOSS', 'IRSOIOU_LL',
                     'LLOSS_ONLY', 'LLOSS-ONLY', 'SOFTIOU']
    parser.add_argument('--loss-type', type=str, default='L2', 
                        choices=_loss_choices,
                        help='Loss: L1, L2, L3, L3D, L4, IRSOIOU, IRSOIOU-LLOSS, LLOSS-ONLY, SOFTIOU (+ -ONLY)')
    parser.add_argument('--use-gaussian-pinwheel', action='store_true', default=False,
                        help='Use 7x7 Gaussian convolution + pinwheel mask in spatial attention to reduce false alarms (σ learnable per layer)')
    parser.add_argument('--use-rotated-pinwheel', action='store_true', default=False,
                        help='Add multi-orientation rotated pinwheel (only when --use-gaussian-pinwheel); fuse by argmax over orientations')
    parser.add_argument('--gp-pipeline', type=str, default=None, choices=['A', 'B'], help='GP: A=argmax, B=line-energy+k-gather')
    parser.add_argument('--n-orientations', type=int, default=4,
                        help='Number of orientations for rotated pinwheel when --use-rotated-pinwheel (default 4)')
    parser.add_argument('--use-learnable-rotated-pinwheel', action='store_true', default=False,
                        help='Use learnable rotated pinwheel mask (init_angle, rotate_angle, tau); only when --use-gaussian-pinwheel, replaces fixed pinwheel')
    parser.add_argument('--use-rot-weight', action='store_true', default=False,
                        help='Three fixed masks (PI4,PI3,PI2_PI3) + learnable weights (only when --use-gaussian-pinwheel); save suffix -RotWeight')
    parser.add_argument('--use-rot-weight-two', action='store_true', default=False,
                        help='Two masks only (PI3, PI2_PI3, no pinwheel) + learnable weights; save suffix -RotWeight2')

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

        self.train_loader = Data.DataLoader(trainset, args.batch_size, shuffle=True, drop_last=True,
                                              num_workers=8, pin_memory=True, persistent_workers=True)
        self.val_loader = Data.DataLoader(valset, 1, drop_last=False,
                                          num_workers=4, pin_memory=True, persistent_workers=True)

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

        use_gp = getattr(args, 'use_gaussian_pinwheel', False)
        use_rot = getattr(args, 'use_rotated_pinwheel', False)
        gp_pipeline = getattr(args, 'gp_pipeline', None)
        use_rot = getattr(args, 'use_rotated_pinwheel', False)
        use_learn_rot = getattr(args, 'use_learnable_rotated_pinwheel', False)
        use_rot_weight = getattr(args, 'use_rot_weight', False)
        use_rot_weight_two = getattr(args, 'use_rot_weight_two', False)
        n_orient = getattr(args, 'n_orientations', 4)
        use_gp_pipeline_b = bool(use_gp and gp_pipeline == 'B')
        if use_gp and gp_pipeline == 'A':
            use_rot = True
        if use_gp and gp_pipeline == 'B':
            use_rot = False
            use_learn_rot = False
            use_rot_weight = False
            use_rot_weight_two = False
        if use_rot and not use_gp:
            use_rot = False
        if use_learn_rot and not use_gp:
            use_learn_rot = False
        if use_rot_weight and not use_gp:
            use_rot_weight = False
        if use_rot_weight_two and not use_gp:
            use_rot_weight_two = False
        if use_rot_weight or use_rot_weight_two:
            use_rot = False
            use_learn_rot = False
        if use_rot_weight_two:
            use_rot_weight = False
        if use_learn_rot and use_rot:
            use_rot = False
        model = MSHNet(3, use_gaussian_pinwheel=use_gp, use_rotated_pinwheel=use_rot, n_orientations=n_orient, use_learnable_rotated_pinwheel=use_learn_rot, use_rot_weight=use_rot_weight, use_rot_weight_two=use_rot_weight_two, gp_pipeline_b=use_gp_pipeline_b)
        if use_gp:
            msg = 'Spatial attention: 7x7 Gaussian + pinwheel (σ learnable)'
            if use_rot_weight_two:
                msg += ' + 2 masks (PI3, PI2_PI3) with learnable weights (-RotWeight2)'
            elif use_rot_weight:
                msg += ' + 3 fixed rotation masks with learnable weights (-RotWeight)'
            elif use_learn_rot:
                msg += ' + learnable rotated mask (init_angle, rotate_angle, tau)'
            elif use_rot:
                msg += ' + rotated pinwheel (n=%d)' % n_orient
            print(msg)

        if args.multi_gpus:
            if torch.cuda.device_count() > 1:
                print('use '+str(torch.cuda.device_count())+' gpus')
                model = nn.DataParallel(model, device_ids=[0, 1])
        model.to(device)
        self.model = model

        self.optimizer = Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)

        self.down = nn.MaxPool2d(2, 2)
        
        # 根据参数选择损失函数
        loss_type = args.loss_type.upper()
        # 保存标准化的损失函数名称用于目录命名
        loss_name_for_dir = loss_type
        
        if loss_type == 'L1':
            self.loss_fun = L1IoULoss()
            self.use_lloss_for_l1 = True
            loss_name_for_dir = 'L1'
            print(f'Using Loss: L1 (L1-based IoU Loss with LLoss)')
        elif loss_type == 'L1-ONLY' or loss_type == 'L1_ONLY':
            self.loss_fun = L1IoULoss()
            self.use_lloss_for_l1 = False
            loss_name_for_dir = 'L1-ONLY'
            print(f'Using Loss: L1-ONLY (L1 IoU Loss without LLoss)')
        elif loss_type == 'L2':
            self.loss_fun = SLSIoULoss()
            loss_name_for_dir = 'L2'
            print(f'Using Loss: L2 (SLS IoU Loss - default)')
        elif loss_type == 'L2-ONLY' or loss_type == 'L2_ONLY':
            self.loss_fun = L2IoULoss()
            self.use_lloss_for_l2 = False
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
            loss_name_for_dir = 'L3D'
            print(f'Using Loss: L3+D_loss (L3 IoU Loss with D_loss from IR-SOIoU)')
        elif loss_type == 'L4':
            self.loss_fun = L4IoULoss()
            self.use_lloss_for_l4 = True
            loss_name_for_dir = 'L4'
            print(f'Using Loss: L4 (Man Fung IoU Loss with LLoss)')
        elif loss_type == 'L4-ONLY' or loss_type == 'L4_ONLY':
            self.loss_fun = L4IoULoss()
            self.use_lloss_for_l4 = False
            loss_name_for_dir = 'L4-ONLY'
            print(f'Using Loss: L4-ONLY (L4 IoU Loss without LLoss)')
        elif loss_type == 'IRSOIOU' or loss_type == 'IR-SOIOU' or loss_type == 'RSOIOU':
            self.loss_fun = IRSOIoULoss()
            loss_name_for_dir = 'IRSOIOU'
            print(f'Using Loss: IR-SOIoU (Region Energy-Based Dynamic Loss with D_loss)')
        elif loss_type == 'IRSOIOU_LLOSS' or loss_type == 'IRSOIOU-LLOSS' or loss_type == 'IRSOIOU_LL':
            self.loss_fun = IRSOIoULoss()
            self.use_lloss_for_irsoiou = True
            loss_name_for_dir = 'IRSOIOU-LLOSS'
            print(f'Using Loss: IR-SOIoU with LLoss')
        elif loss_type == 'LLOSS_ONLY' or loss_type == 'LLOSS-ONLY':
            self.loss_fun = LLossOnlyLoss()
            loss_name_for_dir = 'LLOSS-ONLY'
            print(f'Using Loss: LLoss only (location/shape loss only)')
        elif loss_type == 'SOFTIOU':
            self.loss_fun = SoftIoULossModule()
            loss_name_for_dir = 'SOFTIOU'
            print(f'Using Loss: Soft IoU (standard soft IoU module)')
        else:
            self.loss_fun = SLSIoULoss()
            loss_name_for_dir = 'L2'  # 默认使用 L2
            print(f'Unknown loss type {loss_type}, using default L2 (SLS IoU Loss)')
        
        # 保存损失函数名称用于目录命名
        self.loss_name_for_dir = loss_name_for_dir
        
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = mIoU(1)
        self.ROC  = ROCMetric(1, 10)
        self.best_iou = 0
        self.warm_epoch = args.warm_epoch

        if args.mode=='train':
            if args.if_checkpoint:
                check_folder = ''
                checkpoint = torch.load(check_folder+'/checkpoint.pkl')
                self.model.load_state_dict(checkpoint['net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch']+1
                self.best_iou = checkpoint['iou']
                self.save_folder = check_folder
            else:
                # 使用相对路径，兼容Windows和Linux，包含损失函数类型
                timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                gp_suffix = '-GP' if getattr(self.args, 'use_gaussian_pinwheel', False) else ''
                rot_suffix = '-Rot' if getattr(self.args, 'use_rotated_pinwheel', False) else ''
                learn_rot_suffix = '-LearnRot' if getattr(self.args, 'use_learnable_rotated_pinwheel', False) else ''
                rot_weight_suffix = '-RotWeight' if getattr(self.args, 'use_rot_weight', False) else ''
                rot_weight_two_suffix = '-RotWeight2' if getattr(self.args, 'use_rot_weight_two', False) else ''
                gp_pl = getattr(self.args, 'gp_pipeline', None)
                pipeline_suffix = '-PipelineA' if gp_pl == 'A' else '-PipelineB' if gp_pl == 'B' else ''
                save_base = osp.join('weight', f'MSHNet-{self.loss_name_for_dir}{gp_suffix}{rot_suffix}{learn_rot_suffix}{rot_weight_suffix}{rot_weight_two_suffix}{pipeline_suffix}-{timestamp}')
                self.save_folder = save_base
                if not osp.exists(self.save_folder):
                    os.makedirs(self.save_folder, exist_ok=True)
        if args.mode=='test':
          
            weight = torch.load(args.weight_path)
            # 兼容两种格式：直接state_dict或包含state_dict的字典
            if isinstance(weight, dict):
                if 'state_dict' in weight:
                    self.model.load_state_dict(weight['state_dict'])
                elif 'net' in weight:
                    self.model.load_state_dict(weight['net'])
                else:
                    # 如果字典的键看起来像模型参数，直接使用
                    self.model.load_state_dict(weight)
            else:
                # 直接是state_dict
                self.model.load_state_dict(weight)
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

            if hasattr(self, 'use_lloss_for_irsoiou') and isinstance(self.loss_fun, IRSOIoULoss):
                with_shape = self.use_lloss_for_irsoiou
                loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch, with_shape=with_shape)
                for j in range(len(masks)):
                    if j>0:
                        labels = self.down(labels)
                    loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch, with_shape=with_shape)
            elif hasattr(self, 'use_lloss_for_l3') and isinstance(self.loss_fun, L3IoULoss):
                with_shape = self.use_lloss_for_l3
                loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch, with_shape=with_shape)
                for j in range(len(masks)):
                    if j>0:
                        labels = self.down(labels)
                    loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch, with_shape=with_shape)
            elif hasattr(self, 'use_lloss_for_l1') and isinstance(self.loss_fun, L1IoULoss):
                with_shape = self.use_lloss_for_l1
                loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch, with_shape=with_shape)
                for j in range(len(masks)):
                    if j>0:
                        labels = self.down(labels)
                    loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch, with_shape=with_shape)
            elif hasattr(self, 'use_lloss_for_l2') and isinstance(self.loss_fun, L2IoULoss):
                with_shape = self.use_lloss_for_l2
                loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch, with_shape=with_shape)
                for j in range(len(masks)):
                    if j>0:
                        labels = self.down(labels)
                    loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch, with_shape=with_shape)
            elif hasattr(self, 'use_lloss_for_l4') and isinstance(self.loss_fun, L4IoULoss):
                with_shape = self.use_lloss_for_l4
                loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch, with_shape=with_shape)
                for j in range(len(masks)):
                    if j>0:
                        labels = self.down(labels)
                    loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch, with_shape=with_shape)
            else:
                # 其他损失函数使用默认参数（with_shape=True 是默认值）
                loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch)
                for j in range(len(masks)):
                    if j>0:
                        labels = self.down(labels)
                    loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch)
                
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