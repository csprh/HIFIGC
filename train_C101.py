"""
Learnable generative compression model modified from [1],
implemented in Pytorch.

Example usage:
python3 train.py -h

[1] Mentzer et. al., "High-Fidelity Generative Image Compression",
    arXiv:2006.09965 (2020).
"""
from pprint import pprint
import pandas as pd
import numpy as np
import os, glob, time, datetime
import logging, pickle, argparse
import functools, itertools

from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from src.compression import compression_utils
from src.loss.perceptual_similarity import perceptual_loss as ps
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
# Custom modules
from src.model import Model
from src.helpers import utils, datasets, metrics
from default_config import classi_only, hific_args, mse_lpips_args, directories, ModelModes, ModelTypes

# go fast boi!!
torch.backends.cudnn.benchmark = True

def create_model(args, device, logger, storage, storage_test):

    start_time = time.time()
    model = Model(args, logger, storage, storage_test, model_type=args.model_type)
    logger.info(model)
    logger.info('Trainable parameters:')

    for n, p in model.named_parameters():
        logger.info('{} - {}'.format(n, p.shape))

    logger.info("Number of trainable parameters: {}".format(utils.count_parameters(model)))
    logger.info("Estimated size (under fp32): {:.3f} MB".format(utils.count_parameters(model) * 4. / 10**6))
    logger.info('Model init {:.3f}s'.format(time.time() - start_time))

    return model

def optimize_loss(loss, opt, retain_graph=False):
    loss.backward(retain_graph=retain_graph)
    opt.step()
    opt.zero_grad()

def optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt):
    compression_loss.backward()
    amortization_opt.step()
    hyperlatent_likelihood_opt.step()
    amortization_opt.zero_grad()
    hyperlatent_likelihood_opt.zero_grad()


def make_deterministic(seed=42):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Don't go fast boi :(

    np.random.seed(seed)


def end_of_epoch_metrics(args, model, data_loader, device, logger):

    model.eval()
    old_mode = model.model_mode
    #model.set_model_mode(ModelModes.EVALUATION)
    model.training = False
    classi_acc_total = []
    n, N = 0, len(data_loader.dataset)
    input_filenames_total = list()
    output_filenames_total = list()
    q_bpp_total, q_bpp_total_attained, LPIPS_total = torch.Tensor(N), torch.Tensor(N), torch.Tensor(N)
    MS_SSIM_total, PSNR_total = torch.Tensor(N), torch.Tensor(N)
    comp_loss_total, classi_loss_total, classi_acc_total1 = torch.Tensor(N), torch.Tensor(N),torch.Tensor(N)
    with torch.no_grad():
        thisIndx =  0
        for idx1, (dataAll, yAll) in enumerate(tqdm(data_loader), 0):
          dataAll = dataAll.to(device, dtype=torch.float)
          yAll = yAll.to(device)
          losses, intermediates = model(dataAll, yAll, return_intermediates=True, writeout=True)
          classi_acc = losses['classi_acc']
          classi_acc_total.append(classi_acc.item())



    # Reproducibility
    make_deterministic()
    perceptual_loss_fn = ps.PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available())

    # Build probability tables
    logger.info('Building hyperprior probability tables...')
    model.Hyperprior.hyperprior_entropy_model.build_tables()
    logger.info('All tables built.')


    max_value = 255.
    MS_SSIM_func = metrics.MS_SSIM(data_range=max_value)
    utils.makedirs(args.output_dir)

    logger.info('Starting compression...')
    start_time = time.time()



    with torch.no_grad():
        thisIndx =  0
        for idx1, (dataAll, yAll) in enumerate(tqdm(data_loader), 0):
          dataAll = dataAll.to(device, dtype=torch.float)
          yAll = yAll.to(device)
          #if idx1 > 2:
          #    break
          B = dataAll.size(0)
          for idxB in range(B):
            data = dataAll[idxB,:,:,:]
            data = data.unsqueeze(0)
            y = yAll[idxB]
            y = y.unsqueeze(0)
            model.set_model_mode(old_mode)
            model.training = False
            losses = model(data, y, train_generator=False)
            compression_loss = losses['compression']

            if model.use_classiOnly is True:
                classi_loss = losses['classi']
                classi_acc = losses['classi_acc']

            model.set_model_mode(ModelModes.EVALUATION)
            model.training = False
            # Perform entropy coding
            q_bpp_attained, compressed_output = model.compress(data, silent = True)

            if args.save is True:
                    compression_utils.save_compressed_format(compressed_output, out_path=os.path.join(args.output_dir, "compressed.hfc"))

            reconstruction = model.decompress(compressed_output)
            q_bpp = compressed_output.total_bpp


            if args.normalize_input_image is True:
                # [-1., 1.] -> [0., 1.]
                data = (data + 1.) / 2.

            perceptual_loss = perceptual_loss_fn.forward(reconstruction, data, normalize=True)

            # [0., 1.] -> [0., 255.]
            psnr = metrics.psnr(reconstruction.cpu().numpy() * max_value, data.cpu().numpy() * max_value, max_value)
            ms_ssim = MS_SSIM_func(reconstruction * max_value, data * max_value)
            PSNR_total[thisIndx] = torch.Tensor(psnr)
            MS_SSIM_total[thisIndx] = ms_ssim.data

            q_bpp_per_im = float(q_bpp.item()) if type(q_bpp) == torch.Tensor else float(q_bpp)

            fname = os.path.join(args.output_dir, "{}_RECON_{:.3f}bpp.png".format(thisIndx, q_bpp_per_im))
            torchvision.utils.save_image(reconstruction, fname, normalize=True)
            output_filenames_total.append(fname)

            q_bpp_total[thisIndx] = q_bpp.data if type(q_bpp) == torch.Tensor else q_bpp
            q_bpp_total_attained[thisIndx] = q_bpp_attained.data if type(q_bpp_attained) == torch.Tensor else q_bpp_attained
            LPIPS_total[thisIndx] = perceptual_loss.data
            comp_loss_total[thisIndx] = compression_loss.data
            if model.use_classiOnly is True:
                classi_loss_total[thisIndx] = classi_loss.data
                classi_acc_total1[thisIndx] = classi_acc.data
            thisIndx = thisIndx + 1


    logger.info(f'BPP: mean={q_bpp_total.mean(dim=0):.3f}, std={q_bpp_total.std(dim=0):.3f}')
    logger.info(f'BPPA: mean={q_bpp_total_attained.mean(dim=0):.3f}, std={q_bpp_total_attained.std(dim=0):.3f}')
    logger.info(f'LPIPS: mean={LPIPS_total.mean(dim=0):.3f}, std={LPIPS_total.std(dim=0):.3f}')
    logger.info(f'PSNR: mean={PSNR_total.mean(dim=0):.3f}, std={PSNR_total.std(dim=0):.3f}')
    logger.info(f'MS_SSIM: mean={MS_SSIM_total.mean(dim=0):.3f}, std={MS_SSIM_total.std(dim=0):.3f}')
    logger.info(f'CompLoss: mean={comp_loss_total.mean(dim=0):.3f}, std={comp_loss_total.std(dim=0):.3f}')
    logger.info(f'ClassiLoss: mean={classi_loss_total.mean(dim=0):.3f}, std={classi_loss_total.std(dim=0):.3f}')
    logger.info(f'ClassiAcc1: mean={classi_acc_total1.mean(dim=0):.3f}, std={classi_acc_total1.std(dim=0):.3f}')
    logger.info(f'ClassiAcc2: mean={np.mean(classi_acc_total):.3f}')
    #df = pd.DataFrame([input_filenames_total, output_filenames_total]).T
    #df.columns = ['input_filename', 'output_filename']
    #df['bpp_original'] = bpp_total.cpu().numpy()
    #df['q_bpp'] = q_bpp_total.cpu().numpy()
    #df['LPIPS'] = LPIPS_total.cpu().numpy()

    #df['PSNR'] = PSNR_total.cpu().numpy()
    #df['MS_SSIM'] = MS_SSIM_total.cpu().numpy()

    #df_path = os.path.join(args.output_dir, 'compression_metrics.h5')
    #df.to_hdf(df_path, key='df')

    #pprint(df)

    #logger.info('Complete. Reconstructions saved to {}. Output statistics saved to {}'.format(args.output_dir, df_path))
    delta_t = time.time() - start_time
    logger.info('Time elapsed: {:.3f} s'.format(delta_t))
    logger.info('Rate: {:.3f} Images / s:'.format(float(N) / delta_t))

    model.set_model_mode(old_mode)



def test(args, model, epoch, idx, data, y, test_data, ytest, device, epoch_test_loss, storage, best_test_loss,
         start_time, epoch_start_time, logger, train_writer, test_writer):

    model.eval()
    with torch.no_grad():
        data = data.to(device, dtype=torch.float)

        #losses, intermediates = model(data, y, return_intermediates=True, writeout=False)
        #utils.save_images(train_writer, model.step_counter, intermediates.input_image, intermediates.reconstruction,
        #    fname=os.path.join(args.figures_save, 'recon_epoch{}_idx{}_TRAIN_{:%Y_%m_%d_%H:%M}.jpg'.format(epoch, idx, datetime.datetime.now())))

        test_data = test_data.to(device, dtype=torch.float)
        losses, intermediates = model(test_data, ytest, return_intermediates=True, writeout=True)
        #utils.save_images(test_writer, model.step_counter, intermediates.input_image, intermediates.reconstruction,
        #    fname=os.path.join(args.figures_save, 'recon_epoch{}_idx{}_TEST_{:%Y_%m_%d_%H:%M}.jpg'.format(epoch, idx, datetime.datetime.now())))

        compression_loss = losses['compression']

        classi_loss = losses['classi']
        classi_acc  = losses['classi_acc']
        epoch_test_loss.append(compression_loss.item())
        mean_test_loss = np.mean(epoch_test_loss)
        mean_test_acc = np.mean(classi_acc.item())
        mean_test_classi_loss = np.mean(classi_loss.item())

        #best_test_loss = utils.log(model, storage, epoch, idx, mean_test_loss, compression_loss.item(),
        #                             best_test_loss, start_time, epoch_start_time,
        #                             batch_size=data.shape[0],avg_bpp=0 ,header='[TEST]',
        #                             logger=logger, writer=test_writer)

    return best_test_loss, epoch_test_loss, mean_test_acc, mean_test_classi_loss

def train_test_val_dataset(dataset, test_split=0.1, val_split=0.1, random_state=1):
    train_init_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split, random_state=random_state)
    train_idx, val_idx = train_test_split(list(range(len(train_init_idx))), test_size=val_split, random_state=random_state)
    trainset = Subset(dataset, train_idx)
    valset = Subset(dataset, val_idx)
    testset = Subset(dataset, test_idx)
    return trainset, valset, testset


def train(args, model, train_loader, test_loader, device, logger, optimizers, bpp):

    start_time = time.time()
    test_loader_iter = iter(test_loader)
    current_D_steps, train_generator = 0, True
    best_loss, best_test_loss, mean_epoch_loss = np.inf, np.inf, np.inf
    train_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'train'))
    test_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'test'))
    storage, storage_test = model.storage_train, model.storage_test

    Ntrain = len(train_loader)
    classi_loss_total_train, classi_acc_total_train = torch.Tensor(Ntrain), torch.Tensor(Ntrain)

    classi_opt, amortization_opt, hyperlatent_likelihood_opt = optimizers['classi'], optimizers['amort'], optimizers['hyper']
    #end_of_epoch_metrics(args, model, train_loader, device, logger)
    if model.use_discriminator is True:
        disc_opt = optimizers['disc']


    for epoch in trange(args.n_epochs, desc='Epoch'):

        epoch_loss, epoch_test_loss = [], []
        epoch_start_time = time.time()

        if epoch > 0:
            ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)

        model.train()
        test_index = 0
        test_acc_total= 0
        mean_test_acc_total = 0
        test_classi_loss_total = 0
        best_mean_test_classi_loss_total = 10000000000
        for idx, (data, y) in enumerate(tqdm(train_loader, desc='Train'), 0):

            #if idx == 10:
            #    break
            data = data.to(device, dtype=torch.float)
            y = y.to(device)
            try:
                if model.use_classiOnly is True:
                    losses = model(data, y, train_generator=False)
                    classi_loss = losses['classi']
                    classi_acc  = losses['classi_acc']
                    compression_loss = losses['compression']

                    optimize_loss(classi_loss, classi_opt)
                    classi_loss_total_train[idx] = classi_loss.data
                    classi_acc_total_train[idx] = classi_acc.data
                    model.step_counter += 1
                else:
                  if model.use_discriminator is True:
                    # Train D for D_steps, then G, using distinct batches
                    losses = model(data, y, train_generator=train_generator)
                    compression_loss = losses['compression']
                    disc_loss = losses['disc']

                    if train_generator is True:
                        optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt)
                        train_generator = False
                    else:
                        optimize_loss(disc_loss, disc_opt)
                        current_D_steps += 1

                        if current_D_steps == args.discriminator_steps:
                            current_D_steps = 0
                            train_generator = True

                        continue
                  else:
                    # Rate, distortion, perceptual only
                    losses = model(data, y, train_generator=True)
                    compression_loss = losses['compression']
                    optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt)

            except KeyboardInterrupt:
                # Note: saving not guaranteed!
                if model.step_counter > args.log_interval+1:
                    logger.warning('Exiting, saving ...')
                    ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)
                    return model, ckpt_path
                else:
                    return model, None

            if model.step_counter % args.log_interval == 0:
                epoch_loss.append(compression_loss.item())
                mean_epoch_loss = np.mean(epoch_loss)

                #best_loss = utils.log(model, storage, epoch, idx, mean_epoch_loss, compression_loss.item(),
                #                best_loss, start_time, epoch_start_time, batch_size=data.shape[0],
                #                avg_bpp=bpp, logger=logger, writer=train_writer)
                try:
                    test_data, ytest = test_loader_iter.next()

                except StopIteration:
                    test_loader_iter = iter(test_loader)
                    test_data, ytest = test_loader_iter.next()

                ytest = ytest.to(device)
                best_test_loss, epoch_test_loss, mean_test_acc, mean_test_classi_loss = test(args, model, epoch, idx, data, y, test_data, ytest, device, epoch_test_loss, storage_test,
                     best_test_loss, start_time, epoch_start_time, logger, train_writer, test_writer)

                test_index = test_index + 1
                test_classi_loss_total = test_classi_loss_total + mean_test_classi_loss
                mean_test_classi_loss_total = test_classi_loss_total/test_index

                test_acc_total = test_acc_total  + mean_test_acc
                mean_test_acc_total = test_acc_total/test_index

                with open(os.path.join(args.storage_save, 'storage_{}_tmp.pkl'.format(args.name)), 'wb') as handle:
                    pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)

                model.train()



                if model.step_counter > args.n_steps:
                    logger.info('Reached step limit [args.n_steps = {}]'.format(args.n_steps))
                    break


            # LR scheduling
            if model.use_classiOnly is True:
                utils.update_lr(args, classi_opt, model.step_counter, logger)
            utils.update_lr(args, amortization_opt, model.step_counter, logger)
            utils.update_lr(args, hyperlatent_likelihood_opt, model.step_counter, logger)
            if model.use_discriminator is True:
                utils.update_lr(args, disc_opt, model.step_counter, logger)
        if mean_test_classi_loss_total < best_mean_test_classi_loss_total:
            logger.info(f'Classi_loss decreased to : {mean_test_classi_loss:.3f}.  Saving Model')
            best_mean_test_classi_loss_total = mean_test_classi_loss_total
            ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)
        # End epoch
        mean_epoch_loss = np.mean(epoch_loss)
        mean_epoch_test_loss = np.mean(epoch_test_loss)

        logger.info('===>> Epoch {} | Mean train loss: {:.3f} | Mean test loss: {:.3f}  | Mean test classi acc: {:.3f}'.format(epoch,
            mean_epoch_loss, mean_epoch_test_loss, mean_test_acc_total))
        logger.info(f'ClassiLossTrain: mean={classi_loss_total_train.mean(dim=0):.3f}, std={classi_loss_total_train.std(dim=0):.3f}')
        logger.info(f'ClassiAccTrain: mean={classi_acc_total_train.mean(dim=0):.3f}, std={classi_acc_total_train.std(dim=0):.3f}')

        #end_of_epoch_metrics(args, model, train_loader, device, logger)
        #end_of_epoch_metrics(args, model, test_loader, device, logger)


        if model.step_counter > args.n_steps:
            break

    with open(os.path.join(args.storage_save, 'storage_{}_{:%Y_%m_%d_%H:%M:%S}.pkl'.format(args.name, datetime.datetime.now())), 'wb') as handle:
        pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)
    args.ckpt = ckpt_path
    logger.info("Training complete. Time elapsed: {:.3f} s. Number of steps: {}".format((time.time()-start_time), model.step_counter))

    return model, ckpt_path


if __name__ == '__main__':

    description = "Learnable generative compression."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General options - see `default_config.py` for full options
    general = parser.add_argument_group('General options')
    general.add_argument("-n", "--name", default=None, help="Identifier for checkpoints and metrics.")
    general.add_argument("-mt", "--model_type", required=True, choices=(ModelTypes.COMPRESSION, ModelTypes.COMPRESSION_GAN, ModelTypes.CLASSI_ONLY),
        help="Type of model - with or without GAN component")
    general.add_argument("-regime", "--regime", choices=('low','med','high'), default='low', help="Set target bit rate - Low (0.14), Med (0.30), High (0.45)")
    general.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU ID.")
    general.add_argument("-log_intv", "--log_interval", type=int, default=10, help="Number of steps between logs.")
    general.add_argument("-save_intv", "--save_interval", type=int, default=50000, help="Number of steps between checkpoints.")
    general.add_argument("-multigpu", "--multigpu", help="Toggle data parallel capability using torch DataParallel", action="store_true")
    general.add_argument("-norm", "--normalize_input_image", default = True, help="Normalize input images to [-1,1]", action="store_true")
    general.add_argument('-bs', '--batch_size', type=int, default=8, help='input batch size for training')
    general.add_argument('--save', type=str, default='experiments', help='Parent directory for stored information (checkpoints, logs, etc.)')
    general.add_argument("-lt", "--likelihood_type", choices=('gaussian', 'logistic'), default='gaussian', help="Likelihood model for latents.")
    general.add_argument("-force_gpu", "--force_set_gpu", help="Set GPU to given ID", action="store_true")
    general.add_argument("-LMM", "--use_latent_mixture_model", help="Use latent mixture model as latent entropy model.", action="store_true")
    general.add_argument("-o", "--output_dir", type=str, default='data/reconstructions',
        help="Path to directory to store output images")
    # Optimization-related options
    optim_args = parser.add_argument_group("Optimization-related options")
    optim_args.add_argument('-steps', '--n_steps', type=float, default=1e6,
        help="Number of gradient steps. Optimization stops at the earlier of n_steps/n_epochs.")
    optim_args.add_argument('-epochs', '--n_epochs', type=int, default=100,
        help="Number of passes over training dataset. Optimization stops at the earlier of n_steps/n_epochs.")
    optim_args.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Optimizer learning rate.")
    optim_args.add_argument("-wd", "--weight_decay", type=float, default=1e-6, help="Coefficient of L2 regularization.")

    # Architecture-related options
    arch_args = parser.add_argument_group("Architecture-related options")
    arch_args.add_argument('-nrb', '--n_residual_blocks', type=int, default=7,
        help="Number of residual blocks to use in Generator.")

    # Warmstart adversarial training from autoencoder/hyperprior
    warmstart_args = parser.add_argument_group("Warmstart options")
    warmstart_args.add_argument("-warmstart", "--warmstart", help="Warmstart adversarial training from autoencoder + hyperprior ckpt.", action="store_true")
    warmstart_args.add_argument("-ckpt", "--warmstart_ckpt", default=None, help="Path to autoencoder + hyperprior ckpt.")

    cmd_args = parser.parse_args()

    if (cmd_args.gpu != 0) or (cmd_args.force_set_gpu is True):
        torch.cuda.set_device(cmd_args.gpu)

    if cmd_args.model_type == ModelTypes.COMPRESSION:
        args = mse_lpips_args
    elif cmd_args.model_type == ModelTypes.COMPRESSION_GAN:
        args = hific_args
    elif cmd_args.model_type == ModelTypes.CLASSI_ONLY:
        args = classi_only

    start_time = time.time()
    is_gpu=True
    device = utils.get_device(is_gpu=is_gpu)

    # Override default arguments from config file with provided command line arguments
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    args_d, cmd_args_d = dictify(args), vars(cmd_args)
    args_d.update(cmd_args_d)
    args = utils.Struct(**args_d)
    args = utils.setup_generic_signature(args, special_info=args.model_type)
    args.target_rate = args.target_rate_map[args.regime]
    args.lambda_A = args.lambda_A_map[args.regime]
    args.n_steps = int(args.n_steps)

    storage = defaultdict(list)
    storage_test = defaultdict(list)
    logger = utils.logger_setup(logpath=os.path.join(args.snapshot, 'logs'), filepath=os.path.abspath(__file__))

    if args.warmstart is True:
        assert args.warmstart_ckpt is not None, 'Must provide checkpoint to previously trained AE/HP model.'
        logger.info('Warmstarting discriminator/generator from autoencoder/hyperprior model.')
        if args.model_type != ModelTypes.COMPRESSION_GAN:
            logger.warning('Should warmstart compression-gan model.')
        args, model, optimizers = utils.load_model(args.warmstart_ckpt, logger, device,
            model_type=args.model_type, current_args_d=dictify(args), strict=False, prediction=False)
    else:
        model = create_model(args, device, logger, storage, storage_test)
        model = model.to(device)
        amortization_parameters = itertools.chain.from_iterable(
            [am.parameters() for am in model.amortization_models])

        hyperlatent_likelihood_parameters = model.Hyperprior.hyperlatent_likelihood.parameters()

        amortization_opt = torch.optim.Adam(amortization_parameters,
            lr=args.learning_rate)
        hyperlatent_likelihood_opt = torch.optim.Adam(hyperlatent_likelihood_parameters,
            lr=args.learning_rate)
        optimizers = dict(amort=amortization_opt, hyper=hyperlatent_likelihood_opt)

        if model.use_discriminator is True:
            discriminator_parameters = model.Discriminator.parameters()
            disc_opt = torch.optim.Adam(discriminator_parameters, lr=args.learning_rate)
            optimizers['disc'] = disc_opt

    classi_parameters = model.Classi.parameters()
    #classi_opt = torch.optim.Adam(classi_parameters, lr=1e-3, weight_decay=1e-4)
    classi_opt = torch.optim.Adam(classi_parameters, lr=1e-3, weight_decay=1e-4)
    #classi_opt = torch.optim.Adam(classi_parameters, lr=args.learning_rate)
    optimizers['classi'] = classi_opt

    for params in model.Encoder.parameters():
       params.requires_grad = False
    for params in model.Generator.parameters():
       params.requires_grad = False


    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and args.multigpu is True:
        # Not supported at this time
        raise NotImplementedError('MultiGPU not supported yet.')
        logger.info('Using {} GPUs.'.format(n_gpus))
        model = nn.DataParallel(model)

    logger.info('MODEL TYPE: {}'.format(args.model_type))
    logger.info('MODEL MODE: {}'.format(args.model_mode))
    logger.info('BITRATE REGIME: {}'.format(args.regime))
    logger.info('SAVING LOGS/CHECKPOINTS/RECORDS TO {}'.format(args.snapshot))
    logger.info('USING DEVICE {}'.format(device))
    logger.info('USING GPU ID {}'.format(args.gpu))
    logger.info('USING DATASET: {}'.format(args.dataset))
    C101Root = '/space/csprh/DASA/DATABASES/'
    args.C101Root = C101Root

    W = 256
    H = 256
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #transform = transforms.Compose([transforms.Resize((W, H)),  transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #transform = transforms.Compose([transforms.Grayscale(3), transforms.Resize((W, H)),  transforms.ToTensor()])
    #transform = transforms.Compose([transforms.Resize((W, H)),  transforms.ToTensor()])

    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.Resize((W,H)),
    transforms.RandomCrop((W,H), pad_if_needed=True, padding_mode='edge'),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    #transform = transforms.Compose(
    #[transforms.ToPILImage(),
    # transforms.Resize((W, H)),
     #transforms.ToTensor(),
     #transforms.Normalize(mean=[0.485, 0.456, 0.406],
     #                     std=[0.229, 0.224, 0.225])])

    wholeset = torchvision.datasets.Caltech101(root=C101Root,
                                        download=False, transform=transform)
    wholeset.image_dims = (3, W, H)

    trainset, valset, testset = train_test_val_dataset(wholeset, test_split=0.1, val_split=0.1, random_state=1)
    #trainset, testset = train_val_dataset(wholeset, val_split=0.25)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    #test_loader = datasets.get_dataloaders(args.dataset,
    #                            root=args.dataset_path,
    #                            batch_size=args.batch_size,
    #                            logger=logger,
    #                            mode='validation',
    #                            shuffle=True,
    #                            normalize=args.normalize_input_image)

    #train_loader = datasets.get_dataloaders(args.dataset,
    #                            root=args.dataset_path,
    #                            batch_size=args.batch_size,
    #                            logger=logger,
    #                            mode='train',
    #                            shuffle=True,
    #                            normalize=args.normalize_input_image)

    args.n_data = len(train_loader.dataset)
    args.image_dims = wholeset.image_dims
    logger.info('Training elements: {}'.format(args.n_data))
    logger.info('Input Dimensions: {}'.format(args.image_dims))
    logger.info('Optimizers: {}'.format(optimizers))
    logger.info('Using device {}'.format(device))

    metadata = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('__') or 'logger' in n))
    logger.info(metadata)

    """
    Train
    """
    model, ckpt_path = train(args, model, train_loader, val_loader, device, logger, optimizers=optimizers, bpp = 8*W*H*3)

    """

    python3 -m pudb.run train_C101.py --model_type compression_gan --regime low --n_steps 1e6 --warmstart -ckpt /space/csprh/DASA/HIFIGC/models/hific_low.pt
    python3 -m pudb.run train_C101.py -bs 8 --model_type classi_only --regime low --n_steps 1e6 --warmstart -ckpt /space/csprh/DASA/HIFIGC/models/hific_low.pt

    TODO
    Generate metrics
    """
