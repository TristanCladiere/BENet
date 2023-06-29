from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
from utils.utils import AverageMeter


def do_train(cfg, model, data_loader, loss_factory, optimizer, epoch, writer_dict, val=False, val_step=1):

    if val and (epoch % val_step == 0 or epoch == cfg.TRAIN.END_EPOCH - 1):
        phases = ["train", "val"]
    else:
        phases = ["train"]
    logger = logging.getLogger("train")

    main_database = cfg.DATASET.TRAIN.split("_")[0]
    mixed_database = cfg.DATASET.MIXED_WITH

    for phase in phases:

        hm_loss_meter = [AverageMeter() for wtl in cfg.LOSS.WITH_HM_LOSS if wtl]
        hw_loss_meter = [AverageMeter() for wtl in cfg.LOSS.WITH_HW_LOSS if wtl]
        bu_cat_loss_meter = [AverageMeter() for wtl in cfg.LOSS.WITH_BU_CAT_LOSS if wtl]
        bu_cont_loss_meter = [AverageMeter() for wtl in cfg.LOSS.WITH_BU_CONT_LOSS if wtl]
        pc_cat_loss_meter = [AverageMeter() for wtl in cfg.LOSS.WITH_PC_CAT_LOSS if wtl]
        pc_cont_loss_meter = [AverageMeter() for wtl in cfg.LOSS.WITH_PC_CONT_LOSS if wtl]
        context_cat_loss_meter = [AverageMeter() for wtl in cfg.LOSS.WITH_CONTEXT_CAT_LOSS if wtl]
        context_cont_loss_meter = [AverageMeter() for wtl in cfg.LOSS.WITH_CONTEXT_CONT_LOSS if wtl]
        fusion_cat_loss_meter = [AverageMeter() for wtl in cfg.LOSS.WITH_FUSION_CAT_LOSS if wtl]
        fusion_cont_loss_meter = [AverageMeter() for wtl in cfg.LOSS.WITH_FUSION_CONT_LOSS if wtl]

        if mixed_database:
            bu_cat_loss_meter += [AverageMeter() for wtl in cfg.LOSS.WITH_BU_CAT_LOSS if wtl]
            bu_cont_loss_meter += [AverageMeter() for wtl in cfg.LOSS.WITH_BU_CONT_LOSS if wtl]
            pc_cat_loss_meter += [AverageMeter() for wtl in cfg.LOSS.WITH_PC_CAT_LOSS if wtl]
            pc_cont_loss_meter += [AverageMeter() for wtl in cfg.LOSS.WITH_PC_CONT_LOSS if wtl]
            context_cat_loss_meter += [AverageMeter() for wtl in cfg.LOSS.WITH_CONTEXT_CAT_LOSS if wtl]
            context_cont_loss_meter += [AverageMeter() for wtl in cfg.LOSS.WITH_CONTEXT_CONT_LOSS if wtl]
            fusion_cat_loss_meter += [AverageMeter() for wtl in cfg.LOSS.WITH_FUSION_CAT_LOSS if wtl]
            fusion_cont_loss_meter += [AverageMeter() for wtl in cfg.LOSS.WITH_FUSION_CONT_LOSS if wtl]

        # switch emotions model to train or val mode
        if phase == "train":
            model.train()
        else:
            model.eval()

        for i, custom_batch in enumerate(data_loader[phase]):
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(custom_batch.pop("inputs"))

            hm_losses, hw_losses, bu_cat_losses, bu_cont_losses, pc_cat_losses, pc_cont_losses, context_cat_losses, \
                context_cont_losses, fusion_cat_losses, fusion_cont_losses = loss_factory(outputs, custom_batch)

            loss = 0
            for idx in range(len(hm_losses)):
                if hm_losses[idx] is not None:
                    hm_loss = hm_losses[idx].mean(dim=0)
                    hm_loss_meter[idx].update(
                        hm_loss.item(), len(hm_losses[idx])
                    )
                    loss = loss + hm_loss

                if hw_losses[idx] is not None:
                    hw_loss = hw_losses[idx].mean(dim=0)
                    hw_loss_meter[idx].update(
                        hw_loss.item(), len(hw_losses[idx])
                    )
                    loss = loss + hw_loss

            for idx in range(len(bu_cat_losses)):
                if bu_cat_losses[idx] is not None:
                    bu_cat_loss = bu_cat_losses[idx].mean(dim=0)
                    bu_cat_loss_meter[idx].update(
                        bu_cat_loss.item(), len(bu_cat_losses[idx])
                    )
                    loss = loss + bu_cat_loss
            for idx in range(len(bu_cont_losses)):
                if bu_cont_losses[idx] is not None:
                    bu_cont_loss = bu_cont_losses[idx].mean(dim=0)
                    bu_cont_loss_meter[idx].update(
                        bu_cont_loss.item(), len(bu_cont_losses[idx])
                    )
                    loss = loss + bu_cont_loss

            for idx in range(len(pc_cat_losses)):
                if pc_cat_losses[idx] is not None:
                    pc_cat_loss = pc_cat_losses[idx].mean(dim=0)
                    pc_cat_loss_meter[idx].update(
                        pc_cat_loss.item(), len(pc_cat_losses[idx])
                    )
                    loss = loss + pc_cat_loss
            for idx in range(len(pc_cont_losses)):
                if pc_cont_losses[idx] is not None:
                    pc_cont_loss = pc_cont_losses[idx].mean(dim=0)
                    pc_cont_loss_meter[idx].update(
                        pc_cont_loss.item(), len(pc_cont_losses[idx])
                    )
                    loss = loss + pc_cont_loss

            for idx in range(len(context_cat_losses)):
                if context_cat_losses[idx] is not None:
                    context_cat_loss = context_cat_losses[idx].mean(dim=0)
                    context_cat_loss_meter[idx].update(
                        context_cat_loss.item(), len(context_cat_losses[idx])
                    )
                    loss = loss + context_cat_loss
            for idx in range(len(context_cont_losses)):
                if context_cont_losses[idx] is not None:
                    context_cont_loss = context_cont_losses[idx].mean(dim=0)
                    context_cont_loss_meter[idx].update(
                        context_cont_loss.item(), len(context_cont_losses[idx])
                    )
                    loss = loss + context_cont_loss

            for idx in range(len(fusion_cat_losses)):
                if fusion_cat_losses[idx] is not None:
                    fusion_cat_loss = fusion_cat_losses[idx].mean(dim=0)
                    fusion_cat_loss_meter[idx].update(
                        fusion_cat_loss.item(), len(fusion_cat_losses[idx])
                    )
                    loss = loss + fusion_cat_loss
            for idx in range(len(fusion_cont_losses)):
                if fusion_cont_losses[idx] is not None:
                    fusion_cont_loss = fusion_cont_losses[idx].mean(dim=0)
                    fusion_cont_loss_meter[idx].update(
                        fusion_cont_loss.item(), len(fusion_cont_losses[idx])
                    )
                    loss = loss + fusion_cont_loss

            # compute gradient and do update step
            if phase == "train":
                optimizer.zero_grad()
                loss.backward()

                # # Can be usefull to check for unused parameters
                # for name, param in model.named_parameters():
                #     if param.grad is None:
                #         print(name)
                optimizer.step()

        if cfg.RANK == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Phase: {phase}\t' \
                  '{hm}{hw}{bu_cat}{bu_cont}{pc_cat}{pc_cont}{context_cat}{context_cont}{fusion_cat}{fusion_cont}'.format(
                      epoch, i+1, len(data_loader[phase]),
                      phase=phase,
                      hm=_get_loss_info(hm_loss_meter, 'hm', f"{main_database}{'_' if mixed_database else ''}{mixed_database}", 0),
                      hw=_get_loss_info(hw_loss_meter, 'hw', f"{main_database}{'_' if mixed_database else ''}{mixed_database}", 0),
                      bu_cat=_get_loss_info(bu_cat_loss_meter, 'bu_cat', main_database, mixed_database),
                      bu_cont=_get_loss_info(bu_cont_loss_meter, 'bu_cont', main_database, mixed_database),
                      pc_cat=_get_loss_info(pc_cat_loss_meter, 'pc_cat', main_database, mixed_database),
                      pc_cont=_get_loss_info(pc_cont_loss_meter, "pc_cont", main_database, mixed_database),
                      context_cat=_get_loss_info(context_cat_loss_meter, 'context_cat', main_database, mixed_database),
                      context_cont=_get_loss_info(context_cont_loss_meter, 'context_cont', main_database, mixed_database),
                      fusion_cat=_get_loss_info(fusion_cat_loss_meter, 'fusion_cat', main_database, mixed_database),
                      fusion_cont=_get_loss_info(fusion_cont_loss_meter, 'fusion_cont', main_database, mixed_database)
                  )
            logger.info(msg)
            if writer_dict is not None:
                if phase == "train":
                    writer_dict['global_steps'] += 1
                writer = writer_dict[f'{phase}_writer']
                global_steps = writer_dict['global_steps']
                for idx in range(len(hm_loss_meter)):
                    writer.add_scalar(
                        f"{main_database}{'_' if mixed_database else ''}{mixed_database}_stage{idx}_hm_loss",
                        hm_loss_meter[idx].avg,
                        global_steps
                    )
                    writer.add_scalar(
                        f"{main_database}{'_' if mixed_database else ''}{mixed_database}_stage{idx}_hw_loss",
                        hw_loss_meter[idx].avg,
                        global_steps
                    )

                size_loss_meter = len(bu_cat_loss_meter)
                for idx in range(size_loss_meter):
                    if idx >= size_loss_meter / 2 and mixed_database:
                        database_name = mixed_database
                        idx_str = idx - (size_loss_meter / 2)
                        idx_str = int(idx_str)
                    else:
                        database_name = main_database
                        idx_str = idx
                    writer.add_scalar(
                        f'{database_name}_stage{idx_str}_bu_cat_loss',
                        bu_cat_loss_meter[idx].avg,
                        global_steps
                    )

                size_loss_meter = len(bu_cont_loss_meter)
                for idx in range(size_loss_meter):
                    if idx >= size_loss_meter / 2 and mixed_database:
                        database_name = mixed_database
                        idx_str = idx - (size_loss_meter / 2)
                        idx_str = int(idx_str)
                    else:
                        database_name = main_database
                        idx_str = idx
                    writer.add_scalar(
                        f'{database_name}_stage{idx_str}_bu_cont_loss',
                        bu_cont_loss_meter[idx].avg,
                        global_steps
                    )

                size_loss_meter = len(pc_cat_loss_meter)
                for idx in range(size_loss_meter):
                    if idx >= size_loss_meter / 2 and mixed_database:
                        database_name = mixed_database
                    else:
                        database_name = main_database
                    writer.add_scalar(
                        f'{database_name}_pc_cat_loss',
                        pc_cat_loss_meter[idx].avg,
                        global_steps
                    )

                size_loss_meter = len(pc_cont_loss_meter)
                for idx in range(size_loss_meter):
                    if idx >= size_loss_meter / 2 and mixed_database:
                        database_name = mixed_database
                    else:
                        database_name = main_database
                    writer.add_scalar(
                        f'{database_name}_pc_cont_loss',
                        pc_cont_loss_meter[idx].avg,
                        global_steps
                    )

                size_loss_meter = len(context_cat_loss_meter)
                for idx in range(size_loss_meter):
                    if idx >= size_loss_meter / 2 and mixed_database:
                        database_name = mixed_database
                    else:
                        database_name = main_database
                    writer.add_scalar(
                        f'{database_name}_context_cat_loss',
                        context_cat_loss_meter[idx].avg,
                        global_steps
                    )

                size_loss_meter = len(context_cont_loss_meter)
                for idx in range(size_loss_meter):
                    if idx >= size_loss_meter / 2 and mixed_database:
                        database_name = mixed_database
                    else:
                        database_name = main_database
                    writer.add_scalar(
                        f'{database_name}_context_cont_loss',
                        context_cont_loss_meter[idx].avg,
                        global_steps
                    )

                size_loss_meter = len(fusion_cat_loss_meter)
                for idx in range(size_loss_meter):
                    if idx >= size_loss_meter / 2 and mixed_database:
                        database_name = mixed_database
                    else:
                        database_name = main_database
                    writer.add_scalar(
                        f'{database_name}_fusion_cat_loss',
                        fusion_cat_loss_meter[idx].avg,
                        global_steps
                    )

                size_loss_meter = len(fusion_cont_loss_meter)
                for idx in range(size_loss_meter):
                    if idx >= size_loss_meter / 2 and mixed_database:
                        database_name = mixed_database
                    else:
                        database_name = main_database
                    writer.add_scalar(
                        f'{database_name}_fusion_cont_loss',
                        fusion_cont_loss_meter[idx].avg,
                        global_steps
                    )

            all_losses = [
                sum([hm_loss_meter[idx].avg for idx in range(len(hm_loss_meter))]),
                sum([hw_loss_meter[idx].avg for idx in range(len(hw_loss_meter))]),
                sum([bu_cat_loss_meter[idx].avg for idx in range(len(bu_cat_loss_meter))]),
                sum([bu_cont_loss_meter[idx].avg for idx in range(len(bu_cont_loss_meter))]),
                sum([pc_cat_loss_meter[idx].avg for idx in range(len(pc_cat_loss_meter))]),
                sum([pc_cont_loss_meter[idx].avg for idx in range(len(pc_cont_loss_meter))]),
                sum([context_cat_loss_meter[idx].avg for idx in range(len(context_cat_loss_meter))]),
                sum([context_cont_loss_meter[idx].avg for idx in range(len(context_cont_loss_meter))]),
                sum([fusion_cat_loss_meter[idx].avg for idx in range(len(fusion_cat_loss_meter))]),
                sum([fusion_cont_loss_meter[idx].avg for idx in range(len(fusion_cont_loss_meter))])
            ]

            if writer_dict is not None:
                writer.add_scalar(
                    f'total_loss',
                    sum(all_losses), global_steps
                )

    if phase == 'val':
        return sum(all_losses)
    else:
        return None


def _get_loss_info(loss_meters, loss_name, main_database, mixed_database):
    msg = ''
    size_loss_meters = len(loss_meters)
    database_name = main_database

    for i, meter in enumerate(loss_meters):
        if i >= size_loss_meters / 2 and mixed_database:
            database_name = mixed_database
            i -= size_loss_meters / 2
            i = int(i)
        msg += f'{database_name}_stage{i}-{loss_name}: {meter.val:.3e} ({meter.avg:.3e})  |  '

    return msg
