import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP
from torch.cuda import amp
import torch.distributed as dist

from prettytable import PrettyTable

def dict2table(x):
    t = PrettyTable()
    t.title = x['title']
    t.field_names = x['field_names']
    t.add_row(x['row'])

    return t

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("TOPReID.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    best_index = {'mAP': 0, "Rank1": 0, 'Rank5': 0, 'Rank10': 0}
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                output = model(img, label=target, cam_label=target_cam, view_label=target_view)
                loss = 0
                if cfg.MODEL.RE:
                    index = len(output) - 1
                    for i in range(0, index, 2):
                        loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                        loss = loss + loss_tmp
                    loss = loss + output[-1]
                else:
                    index = len(output)
                    for i in range(0, index, 2):
                        loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                        loss = loss + loss_tmp
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(output, list):
                acc = (output[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                # print(scheduler._get_lr(epoch))
                logger.info("Epoch[{}/{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, epochs, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = {'RGB': img['RGB'].to(device),
                                   'NI': img['NI'].to(device),
                                   'TI': img['TI'].to(device)}
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            if cfg.DATASETS.NAMES == "MSVR310":
                                evaluator.update((feat, vid, camid, target_view, _))
                            else:
                                evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    ret = {
                        'title': f'Validation Results - Epoch: {epoch}',
                        'field_names': ['Rank1', 'Rank5', 'Rank10', 'mAP'],
                        'row': [f'{x:.2%}' for x in [cmc[0], cmc[4], cmc[9], mAP]]
                    }
                    logger.info(f'\n{dict2table(ret)}')


                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = {'RGB': img['RGB'].to(device),
                               'NI': img['NI'].to(device),
                               'TI': img['TI'].to(device)}
                        camids = camids.to(device)
                        scenceids = target_view
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        if cfg.DATASETS.NAMES == "MSVR310":
                            evaluator.update((feat, vid, camid, scenceids, _))
                        else:
                            evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                ret = {
                    'title': f'Validation Results - Epoch: {epoch}',
                    'field_names': ['Rank1', 'Rank5', 'Rank10', 'mAP'],
                    'row': [f'{x:.2%}' for x in [cmc[0], cmc[4], cmc[9], mAP]]
                }
                logger.info(f'\n{dict2table(ret)}')
                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank1'] = cmc[0]
                    best_index['Rank5'] = cmc[4]
                    best_index['Rank10'] = cmc[9]
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best.pth'))

                ret = {
                    'title': 'Best Results',
                    'field_names': ['Rank1', 'Rank5', 'Rank10', 'mAP'],
                    'row': [f'{best_index[x]:.2%}' for x in ['Rank1', 'Rank5', 'Rank10', 'mAP']]
                }
                logger.info(f'\n{dict2table(ret)}')

                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("TOPReID.test")
    logger.info("Enter inferencing")

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, pid, camid, scenceids, imgpath))
            else:
                evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
