import jittor as jt
from jittor import init
import os, time, pickle, argparse, networks, utils
from jittor import nn
import matplotlib.pyplot as plt
from edge_promoting import edge_promoting
import jittor.transform as transforms
import jittor.optim as optim
parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='project_name', help='')
parser.add_argument('--src_data', required=False, default='src_data', help='sec data path')
parser.add_argument('--tgt_data', required=False, default='tgt_data', help='tgt data path')
parser.add_argument('--vgg_model', required=False, default='vgg19.pth', help='pre-trained VGG19 model path')
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--in_ndc', type=int, default=3, help='input channel for discriminator')
parser.add_argument('--out_ndc', type=int, default=1, help='output channel for discriminator')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--nb', type=int, default=8, help='the number of resnet block layer for generator')
parser.add_argument('--input_size', type=int, default=128, help='input size')
parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--pre_train_epoch', type=int, default=10)
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--con_lambda', type=float, default=10, help='lambda for content loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--latest_generator_model', required=False, default='', help='the latest trained model path')
parser.add_argument('--latest_discriminator_model', required=False, default='', help='the latest trained model path')
args = parser.parse_args()
print('------------ Options -------------')
for (k, v) in sorted(vars(args).items()):
    print(('%s: %s' % (str(k), str(v))))
print('-------------- End ----------------')
jt.flags.use_cuda = 0
device=0
if (not os.path.isdir(os.path.join((args.name + '_results'), 'Reconstruction'))):
    os.makedirs(os.path.join((args.name + '_results'), 'Reconstruction'))
if (not os.path.isdir(os.path.join((args.name + '_results'), 'Transfer'))):
    os.makedirs(os.path.join((args.name + '_results'), 'Transfer'))
if (not os.path.isdir(os.path.join('data', args.tgt_data, 'pair'))):
    print('edge-promoting start!!')
    edge_promoting(os.path.join('data', args.tgt_data, 'train'), os.path.join('data', args.tgt_data, 'pair'))
else:
    print('edge-promoting already done')
src_transform = transforms.Compose([transforms.Resize((args.input_size, args.input_size)), transforms.ToTensor(), transforms.ImageNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
tgt_transform = transforms.Compose([transforms.ToTensor(), transforms.ImageNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
train_loader_src = utils.data_load(os.path.join('data', args.src_data), 'train', src_transform, args.batch_size, shuffle=True, drop_last=True)
train_loader_tgt = utils.data_load(os.path.join('data', args.tgt_data), 'pair', tgt_transform, args.batch_size, shuffle=True, drop_last=True)
test_loader_src = utils.data_load(os.path.join('data', args.src_data), 'test', src_transform, 1, shuffle=True, drop_last=True)
G = networks.generator(args.in_ngc, args.out_ngc, args.ngf, args.nb)
D = networks.discriminator(args.in_ndc, args.out_ndc, args.ndf)
VGG = networks.VGG19(init_weights=args.vgg_model, feature_mode=True)
G.train()
D.train()
VGG.eval()
print('---------- Networks initialized -------------')
utils.print_network(G)
utils.print_network(D)
utils.print_network(VGG)
print('-----------------------------------------------')
BCE_loss = nn.BCELoss()
L1_loss = nn.L1Loss()
G_optimizer = optim.Adam(G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
D_optimizer = optim.Adam(D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
G_scheduler = jt.lr_scheduler.MultiStepLR(optimizer=G_optimizer, milestones=[(args.train_epoch // 2), ((args.train_epoch // 4) * 3)], gamma=0.1)
D_scheduler = jt.lr_scheduler.MultiStepLR(optimizer=D_optimizer, milestones=[(args.train_epoch // 2), ((args.train_epoch // 4) * 3)], gamma=0.1)
pre_train_hist = {}
pre_train_hist['Recon_loss'] = []
pre_train_hist['per_epoch_time'] = []
pre_train_hist['total_time'] = []
if (args.latest_generator_model == ''):
    print('Pre-training start!')
    start_time = time.time()
    for epoch in range(args.pre_train_epoch):
        epoch_start_time = time.time()
        Recon_losses = []
        for (x, _) in train_loader_src:
            G_optimizer.zero_grad()
            x_feature = VGG(((x + 1) / 2))
            G_ = G(x)
            G_feature = VGG(((G_ + 1) / 2))
            Recon_loss = (10 * L1_loss(G_feature, x_feature.detach()))
            Recon_losses.append(Recon_loss.item())
            pre_train_hist['Recon_loss'].append(Recon_loss.item())
            G_optimizer.backward(Recon_loss)
            G_optimizer.step()
        per_epoch_time = (time.time() - epoch_start_time)
        pre_train_hist['per_epoch_time'].append(per_epoch_time)
        print(('[%d/%d] - time: %.2f, Recon loss: %.3f' % ((epoch + 1), args.pre_train_epoch, per_epoch_time, jt.mean(jt.array(Recon_losses)))))
    total_time = (time.time() - start_time)
    pre_train_hist['total_time'].append(total_time)
    with open(os.path.join((args.name + '_results'), 'pre_train_hist.pkl'), 'wb') as f:
        pickle.dump(pre_train_hist, f)
    with jt.no_grad():
        G.eval()
        for (n, (x, _)) in enumerate(train_loader_src):
            G_recon = G(x)
            result = jt.contrib.concat((x[0], G_recon[0]), dim=2)
            path = os.path.join((args.name + '_results'), 'Reconstruction', (((args.name + '_train_recon_') + str((n + 1))) + '.png'))
            plt.imsave(path, ((result.transpose(1, 2, 0) + 1) / 2))
            if (n == 4):
                break
        for (n, (x, _)) in enumerate(test_loader_src):
            G_recon = G(x)
            result = jt.contrib.concat((x[0], G_recon[0]), dim=2)
            path = os.path.join((args.name + '_results'), 'Reconstruction', (((args.name + '_test_recon_') + str((n + 1))) + '.png'))
            plt.imsave(path, ((result.transpose(1, 2, 0) + 1) / 2))
            if (n == 4):
                break
else:
    print('Load the latest generator model, no need to pre-train')
train_hist = {}
train_hist['Disc_loss'] = []
train_hist['Gen_loss'] = []
train_hist['Con_loss'] = []
train_hist['per_epoch_time'] = []
train_hist['total_time'] = []
print('training start!')
start_time = time.time()
real = jt.ones([args.batch_size, 1, (args.input_size // 4), (args.input_size // 4)])
fake = jt.zeros([args.batch_size, 1, (args.input_size // 4), (args.input_size // 4)])
for epoch in range(args.train_epoch):
    epoch_start_time = time.time()
    G.train()
    G_scheduler.step()
    D_scheduler.step()
    Disc_losses = []
    Gen_losses = []
    Con_losses = []
    for ((x, _), (y, _)) in zip(train_loader_src, train_loader_tgt):
        e = y[:, :, :, args.input_size:]
        y = y[:, :, :, :args.input_size]
        D_optimizer.zero_grad()
        y=jt.array(y)
        D_real = D(y)
        D_real_loss = BCE_loss(D_real, real)
        G_ = G(x)
        D_fake = D(G_)
        D_fake_loss = BCE_loss(D_fake, fake)
        D_edge = D(e)
        D_edge_loss = BCE_loss(D_edge, fake)
        Disc_loss = ((D_real_loss + D_fake_loss) + D_edge_loss)
        Disc_losses.append(Disc_loss.item())
        train_hist['Disc_loss'].append(Disc_loss.item())
        D_optimizer.backward(Disc_loss)
        D_optimizer.step()
        G_optimizer.zero_grad()
        G_ = G(x)
        D_fake = D(G_)
        D_fake_loss = BCE_loss(D_fake, real)
        x_feature = VGG(((x + 1) / 2))
        G_feature = VGG(((G_ + 1) / 2))
        Con_loss = (args.con_lambda * L1_loss(G_feature, x_feature.detach()))
        Gen_loss = (D_fake_loss + Con_loss)
        Gen_losses.append(D_fake_loss.item())
        train_hist['Gen_loss'].append(D_fake_loss.item())
        Con_losses.append(Con_loss.item())
        train_hist['Con_loss'].append(Con_loss.item())
        G_optimizer.backward(Gen_loss)
        G_optimizer.step()
    
    per_epoch_time = (time.time() - epoch_start_time)
    train_hist['per_epoch_time'].append(per_epoch_time)
    print(('[%d/%d] - time: %.2f, Disc loss: %.3f, Gen loss: %.3f, Con loss: %.3f' % ((epoch + 1), args.train_epoch, per_epoch_time, jt.mean(jt.array(Disc_losses)), jt.mean(jt.array(Gen_losses)), jt.mean(jt.array(Con_losses)))))
    if (((epoch % 2) == 1) or (epoch == (args.train_epoch - 1))):
        with jt.no_grad():
            G.eval()
            for (n, (x, _)) in enumerate(train_loader_src):
                G_recon = G(x)
                result = jt.contrib.concat((x[0], G_recon[0]), dim=2)
                path = os.path.join((args.name + '_results'), 'Transfer', (((((str((epoch + 1)) + '_epoch_') + args.name) + '_train_') + str((n + 1))) + '.png'))
                plt.imsave(path, ((result.transpose(1, 2, 0) + 1) / 2))
                if (n == 4):
                    break
            for (n, (x, _)) in enumerate(test_loader_src):
                G_recon = G(x)
                result = jt.contrib.concat((x[0], G_recon[0]), dim=2)
                path = os.path.join((args.name + '_results'), 'Transfer', (((((str((epoch + 1)) + '_epoch_') + args.name) + '_test_') + str((n + 1))) + '.png'))
                plt.imsave(path, ((result.transpose(1, 2, 0) + 1) / 2))
                if (n == 4):
                    break
            jt.save(G.state_dict(), os.path.join((args.name + '_results'), 'generator_latest.pkl'))
            jt.save(D.state_dict(), os.path.join((args.name + '_results'), 'discriminator_latest.pkl'))
total_time = (time.time() - start_time)
train_hist['total_time'].append(total_time)
print(('Avg one epoch time: %.2f, total %d epochs time: %.2f' % (jt.mean(jt.array(train_hist['per_epoch_time'])), args.train_epoch, total_time)))
print('Training finish!... save training results')
jt.save(G.state_dict(), os.path.join((args.name + '_results'), 'generator_param.pkl'))
jt.save(D.state_dict(), os.path.join((args.name + '_results'), 'discriminator_param.pkl'))
with open(os.path.join((args.name + '_results'), 'train_hist.pkl'), 'wb') as f:
    pickle.dump(train_hist, f)
