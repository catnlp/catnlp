import numpy as np


def get_labels(predictions, references, label_list, masks, decode_type="general", device="cpu", is_flat=True):
    if device == "cpu":
        y_pred = predictions.detach().clone().numpy()
        y_true = references.detach().clone().numpy()
        masks = masks.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()
        masks = masks.detach().cpu().clone().numpy()
    if decode_type == "general":
        return get_general_labels(y_pred, y_true, label_list, masks)
    elif decode_type == "biaffine":
        return get_biaffine_labels(y_pred, y_true, label_list, masks, is_flat)
    else:
        raise ValueError


def get_general_labels(y_pred, y_true, label_list, masks):
    preds = list()
    golds = list()
    for pred, gold, mask in zip(y_pred, y_true, masks):
        tmp_preds = list()
        tmp_golds = list()
        for p, g, m in zip(pred, gold, mask):
            if m == 0:
                continue
            if g == 0:
                tmp_golds.append("O")
            else:
                tmp_golds.append(label_list[g])
            if p == 0:
                tmp_preds.append("O")
            else:
                tmp_preds.append(label_list[p])
        preds.append(tmp_preds)
        golds.append(tmp_golds)
    return preds, golds


def get_biaffine_labels(y_pred, y_true, label_list, masks, is_flat):
    preds = list()
    golds = list()
    for pred, gold, mask in zip(y_pred, y_true, masks):
        pred_entities = list()
        gold_entities = list()
        offset_dict = dict()
        count = -1
        for idx, m in enumerate(mask):
            if idx == 0:
                offset_dict[idx] = 0
                continue
            if m == 1:
                count += 1
            offset_dict[idx] = count
        max_len = len(mask)
        for i in range(1, max_len):
            for j in range(i, max_len):
                if mask[i] == 0 or mask[j] == 0:
                    continue
                pred_scores = pred[i][j]
                pred_label_id = np.argmax(pred_scores)
                gold_label_id = gold[i][j]
                start_idx = offset_dict[i]
                end_idx = offset_dict[j+1]
                if gold_label_id > 0:
                    gold_entities.append([start_idx, end_idx, label_list[gold_label_id]])
                if pred_label_id > 0:
                    pred_entities.append([start_idx, end_idx, label_list[pred_label_id], pred_scores[pred_label_id]])

        pred_entities = sorted(pred_entities, reverse=True, key=lambda x:x[3])
        new_pred_entities = list()
        for pred_entity in pred_entities:
            start, end, tag, _ = pred_entity
            flag = True
            for new_pred_entity in new_pred_entities:
                new_start, new_end, _ = new_pred_entity
                if start < new_start <= end < new_end or new_start < start <= end < new_end:
                    flag = False
                    break
                if is_flat and start < new_end and new_start < end:
                    #for flat ner nested mentions are not allowed
                    flag = False
                    break
            if flag:
                new_pred_entities.append([start, end, tag])
        pred_entities = new_pred_entities
        count += 1
        # tmp_preds = ["O"] * (count)
        # tmp_golds = ["O"] * (count)
        # for entity in pred_entities:
        #     start, end, tag = entity
        #     tmp_preds[start] = f"B-{tag}"
        #     for i in range(start+1, end):
        #         tmp_preds[i] = f"I-{tag}"
        # for entity in gold_entities:
        #     start, end, tag = entity
        #     tmp_golds[start] = f"B-{tag}"
        #     for i in range(start+1, end):
        #         tmp_golds[i] = f"I-{tag}"
        # # print("entities")
        # # print(pred_entities)
        # # print(gold_entities)
        # # print(masks)
        # # print(offset_dict)
        # preds.append(tmp_preds)
        # golds.append(tmp_golds)
        preds.append(pred_entities)
        golds.append(gold_entities)
    return preds, golds
