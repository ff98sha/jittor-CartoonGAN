import jittor as jt
from jittor import init
from jittor import nn
import utils

class resnet_block(nn.Module):

    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv(channel, channel, kernel, stride=stride, padding=padding)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv(channel, channel, kernel, stride=stride, padding=padding)
        self.conv2_norm = nn.InstanceNorm2d(channel)
        utils.initialize_weights(self)

    def execute(self, input):
        x = nn.relu(self.conv1_norm(self.conv1(input)))
        x = self.conv2_norm(self.conv2(x))
        return (input + x)

class generator(nn.Module):

    def __init__(self, in_nc, out_nc, nf=32, nb=6):
        super(generator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.nb = nb
        self.down_convs = nn.Sequential(nn.Conv(in_nc, nf, 7, stride=1, padding=3), nn.InstanceNorm2d(nf), nn.ReLU(), nn.Conv(nf, (nf * 2), 3, stride=2, padding=1), nn.Conv((nf * 2), (nf * 2), 3, stride=1, padding=1), nn.InstanceNorm2d((nf * 2)), nn.ReLU(), nn.Conv((nf * 2), (nf * 4), 3, stride=2, padding=1), nn.Conv((nf * 4), (nf * 4), 3, stride=1, padding=1), nn.InstanceNorm2d((nf * 4)), nn.ReLU())
        self.resnet_blocks = []
        for i in range(nb):
            self.resnet_blocks.append(resnet_block((nf * 4), 3, 1, 1))
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
        self.up_convs = nn.Sequential(nn.ConvTranspose((nf * 4), (nf * 2), 3, stride=2, padding=1, output_padding=1), nn.Conv((nf * 2), (nf * 2), 3, stride=1, padding=1), nn.InstanceNorm2d((nf * 2)), nn.ReLU(), nn.ConvTranspose((nf * 2), nf, 3, stride=2, padding=1, output_padding=1), nn.Conv(nf, nf, 3, stride=1, padding=1), nn.InstanceNorm2d(nf), nn.ReLU(), nn.Conv(nf, out_nc, 7, stride=1, padding=3), nn.Tanh())
        utils.initialize_weights(self)

    def execute(self, input):
        x = self.down_convs(input)
        x = self.resnet_blocks(x)
        output = self.up_convs(x)
        return output

class discriminator(nn.Module):

    def __init__(self, in_nc, out_nc, nf=32):
        super(discriminator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.convs = nn.Sequential(nn.Conv(in_nc, nf, 3, stride=1, padding=1), nn.LeakyReLU(scale=0.2), nn.Conv(nf, (nf * 2), 3, stride=2, padding=1), nn.LeakyReLU(scale=0.2), nn.Conv((nf * 2), (nf * 4), 3, stride=1, padding=1), nn.InstanceNorm2d((nf * 4)), nn.LeakyReLU(scale=0.2), nn.Conv((nf * 4), (nf * 4), 3, stride=2, padding=1), nn.LeakyReLU(scale=0.2), nn.Conv((nf * 4), (nf * 8), 3, stride=1, padding=1), nn.InstanceNorm2d((nf * 8)), nn.LeakyReLU(scale=0.2), nn.Conv((nf * 8), (nf * 8), 3, stride=1, padding=1), nn.InstanceNorm2d((nf * 8)), nn.LeakyReLU(scale=0.2), nn.Conv((nf * 8), out_nc, 3, stride=1, padding=1), nn.Sigmoid())
        utils.initialize_weights(self)

    def execute(self, input):
        output = self.convs(input)
        return output

class VGG19(nn.Module):

    def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=1000):
        super(VGG19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.Sequential(nn.Linear(((512 * 7) * 7), 4096), nn.ReLU(), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(), nn.Linear(4096, num_classes))
        if (not (init_weights == None)):
            self.load_parameters(jt.load(init_weights))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if (v == 'M'):
                layers += [nn.Pool(2, stride=2, op='maximum')]
            else:
                conv2d = nn.Conv(in_channels, v, 3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm(v), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = v
        return nn.Sequential(*layers)

    def execute(self, x):
        if self.feature_mode:
            module_list = list(self.features.modules())
            for l in module_list[1:27]:
                x = l(x)
        if (not self.feature_mode):
            x = x.view((x.shape[0], (- 1)))
            x = self.classifier(x)
        return x