import os
import torch
from tqdm import tqdm
from utils.utils import get_lr

# 类别映射：和你的strawberry_classes.txt一致（0=good，1=bad）
class_names = ['good', 'bad']


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss = 0
    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()

    # ---------------------- 训练集循环（核心：加类别打印） ----------------------
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, bboxes = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            # 前向传播（获取模型输出）
            outputs = model_train(images)
            loss_value = yolo_loss(outputs, bboxes)
            # 反向传播
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images)
                loss_value = yolo_loss(outputs, bboxes)
            # 反向传播
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

        if ema:
            ema.update(model_train)

        loss += loss_value.item()

        # ====================== 核心修正：训练集类别打印（适配你的输出格式） ======================
        try:
            # 遍历每个输出层，先打印shape确认维度（仅前3个批次打印，避免刷屏）
            if iteration < 3:
                print(f"\n【调试】批次{iteration}，outputs长度：{len(outputs)}")
                for i, out in enumerate(outputs):
                    if out is not None:
                        print(f"output[{i}] shape: {out.shape}")

            # 适配不同版本YOLOv8的输出：找类别索引（优先判断维度）
            for output in outputs:
                if output is None or len(output.shape) != 2:
                    continue
                # 版本适配：如果维度是4，说明类别信息在其他位置，先打印所有值看结构
                if output.shape[1] == 4:
                    print(f"【提示】output维度为4，类别信息位置需调整，当前output前5行：{output[:5].cpu().numpy().round(2)}")
                    continue
                # 维度>=5时，取索引5作为类别（兼容多数版本）
                elif output.shape[1] >= 5:
                    pred_class_idx = output[:, 5].cpu().numpy().astype(int)
                    pred_class_names = [class_names[idx] if idx < len(class_names) else '未知' for idx in
                                        pred_class_idx]
                    # 真实类别：从bboxes中取（bboxes[:,4]是类别索引）
                    true_class_idx = bboxes[:, 4].cpu().numpy().astype(int)
                    true_class_names = [class_names[idx] if idx < len(class_names) else '未知' for idx in
                                        true_class_idx]
                    # 仅打印前5个样本（避免刷屏）
                    print(f"\n===== 训练批次 {iteration} | Epoch {epoch} =====")
                    print(f"真实类别（前5个）：{true_class_names[:5]}")
                    print(f"预测类别（前5个）：{pred_class_names[:5]}")
                    print("-" * 40)
        except Exception as e:
            print(f"【类别打印提示】批次{iteration}：{e}（不影响训练，可忽略）")
        # ====================== 类别打印结束 ======================

        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    # ---------------------- 验证集循环（仅保留核心逻辑，暂不加打印） ----------------------
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, bboxes = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)

            outputs = model_train_eval(images)
            loss_value = yolo_loss(outputs, bboxes)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

        # 保存权值
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
            epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))