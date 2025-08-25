import json
import math
import os
import cv2
import SimpleITK
import numpy as np

from ellipse import drawline_AOD
from pathlib import Path
from tqdm import tqdm

IS_LOCAL_TEST = False


class Evaluation:
    def __init__(self, preds_dir, truths_dir, output_path, fix):
        self.predictions_path = Path(preds_dir)
        self.ground_truth_path = Path(truths_dir)
        self.output_path = Path(output_path)
        self.fix = fix
        self.results = {}

    def load_image(self, image_path) -> SimpleITK.Image:
        image = SimpleITK.ReadImage(str(image_path))
        return image

    def evaluation(self, pred: SimpleITK.Image, label: SimpleITK.Image):
        # aop
        pred_aop = self.cal_aop(pred)
        label_aop = self.cal_aop(label)
        aop = abs(pred_aop - label_aop)

        result = dict()
        result['TP'] = 0
        result['TN'] = 0
        result['FN'] = 0
        result['FP'] = 0
        result['pred_aop'] = pred_aop
        result['label_aop'] = label_aop
        if label_aop >= 120 and pred_aop >= 120:
            result['TP'] = 1
        elif label_aop >= 120 and pred_aop < 120:
            result['FN'] = 1
        elif label_aop < 120 and pred_aop < 120:
            result['TN'] = 1
        elif label_aop < 120 and pred_aop >= 120:
            result['FP'] = 1
        result['aop'] = float(aop)
        # ps
        pred_data_ps = SimpleITK.GetArrayFromImage(pred)
        pred_data_ps[pred_data_ps == 2] = 0
        pred_ps = SimpleITK.GetImageFromArray(pred_data_ps)

        label_data_ps = SimpleITK.GetArrayFromImage(label)
        label_data_ps[label_data_ps == 2] = 0
        label_ps = SimpleITK.GetImageFromArray(label_data_ps)
        if (pred_data_ps == 0).all():
            result['asd_ps'] = 100.0
            result['dice_ps'] = 0.0
            result['hd_ps'] = 100.0
        else:
            result['asd_ps'] = float(self.cal_asd(pred_ps, label_ps))
            result['dice_ps'] = float(self.cal_dsc(pred_ps, label_ps))
            result['hd_ps'] = float(self.cal_hd(pred_ps, label_ps))

        # fh
        pred_data_head = SimpleITK.GetArrayFromImage(pred)
        pred_data_head[pred_data_head == 1] = 0
        pred_data_head[pred_data_head == 2] = 1
        pred_head = SimpleITK.GetImageFromArray(pred_data_head)

        label_data_head = SimpleITK.GetArrayFromImage(label)
        label_data_head[label_data_head == 1] = 0
        label_data_head[label_data_head == 2] = 1
        label_head = SimpleITK.GetImageFromArray(label_data_head)

        if (pred_data_head == 0).all():
            result['asd_fh'] = 100.0
            result['dice_fh'] = 0.0
            result['hd_fh'] = 100.0
        else:
            result['asd_fh'] = float(self.cal_asd(pred_head, label_head))
            result['dice_fh'] = float(self.cal_dsc(pred_head, label_head))
            result['hd_fh'] = float(self.cal_hd(pred_head, label_head))

        # all
        pred_data_all = SimpleITK.GetArrayFromImage(pred)
        pred_data_all[pred_data_all == 2] = 1
        pred_all = SimpleITK.GetImageFromArray(pred_data_all)

        label_data_all = SimpleITK.GetArrayFromImage(label)
        label_data_all[label_data_all == 2] = 1
        label_all = SimpleITK.GetImageFromArray(label_data_all)
        if (pred_data_all == 0).all():
            result['asd_all'] = 100.0
            result['dice_all'] = 0.0
            result['hd_all'] = 100.0
        else:
            result['asd_all'] = float(self.cal_asd(pred_all, label_all))
            result['dice_all'] = float(self.cal_dsc(pred_all, label_all))
            result['hd_all'] = float(self.cal_hd(pred_all, label_all))
        return result

    def process(self):
        metrics = dict()
        predictions_path = str(self.predictions_path)
        ground_truth_path = str(self.ground_truth_path)
        names = sorted(os.listdir(predictions_path),key=lambda x: int(x.split('.')[0]))

        for pre_name in tqdm(names):
            truth_name = str(int((pre_name.split("_")[-1].split(".")[0]))) + ".png"
            pre_image = self.load_image(predictions_path + "/" + pre_name)
            truth_image = self.load_image(ground_truth_path + "/" + truth_name)
            result = self.evaluation(pre_image, truth_image)

            self.results[pre_name] = result

        score, self.aggregates = self.cal_score([i for i in self.results.values()])

        metrics["aggregates"] = self.aggregates
        metrics["score"] = score

        metrics["each_case"] = self.results

        with open(self.output_path, "w") as f:
            f.write(json.dumps(metrics))

    def cal_asd(self, a, b):
        filter1 = SimpleITK.SignedMaurerDistanceMapImageFilter()  
        filter1.SetUseImageSpacing(True)  
        filter1.SetSquaredDistance(False) 
        a_dist = filter1.Execute(a)
        a_dist = SimpleITK.GetArrayFromImage(a_dist)
        a_dist = np.abs(a_dist)
        a_edge = np.zeros(a_dist.shape, a_dist.dtype)
        a_edge[a_dist == 0] = 1
        a_num = np.sum(a_edge)

        filter2 = SimpleITK.SignedMaurerDistanceMapImageFilter()
        filter2.SetUseImageSpacing(True)
        filter2.SetSquaredDistance(False)
        b_dist = filter2.Execute(b)

        b_dist = SimpleITK.GetArrayFromImage(b_dist)
        b_dist = np.abs(b_dist)
        b_edge = np.zeros(b_dist.shape, b_dist.dtype)
        b_edge[b_dist == 0] = 1
        b_num = np.sum(b_edge)

        a_dist[b_edge == 0] = 0.0
        b_dist[a_edge == 0] = 0.0

        asd = (np.sum(a_dist) + np.sum(b_dist)) / (a_num + b_num)

        return asd

    def cal_dsc(self, pd, gt):
        pd = SimpleITK.GetArrayFromImage(pd).astype(np.uint8)
        gt = SimpleITK.GetArrayFromImage(gt).astype(np.uint8)
        y = (np.sum(pd * gt) * 2 + 1) / (np.sum(pd * pd + gt * gt) + 1)
        return y

    def cal_hd(self, a, b):
        a = SimpleITK.Cast(SimpleITK.RescaleIntensity(a), SimpleITK.sitkUInt8)
        b = SimpleITK.Cast(SimpleITK.RescaleIntensity(b), SimpleITK.sitkUInt8)
        filter1 = SimpleITK.HausdorffDistanceImageFilter()
        filter1.Execute(a, b)
        hd = filter1.GetHausdorffDistance()
        return hd

    def onehot_to_mask(self, mask):
        ret = np.zeros([3, 256, 256])
        tmp = mask.copy()
        tmp[tmp == 1] = 255
        tmp[tmp == 2] = 0
        ret[1] = tmp
        tmp = mask.copy()
        tmp[tmp == 2] = 255
        tmp[tmp == 1] = 0
        ret[2] = tmp
        b = ret[0]
        r = ret[1]
        g = ret[2]
        ret = cv2.merge([b, r, g])
        mask = ret.transpose([0, 1, 2])
        return mask

    def cal_aop(self, pred):
        aop = 0.0
        ellipse = None
        ellipse2 = None
        pred_data = SimpleITK.GetArrayFromImage(pred)
        aop_pred = np.array(self.onehot_to_mask(pred_data)).astype(np.uint8)
        contours, _ = cv2.findContours(cv2.medianBlur(aop_pred[:, :, 1], 1), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        contours2, _ = cv2.findContours(cv2.medianBlur(aop_pred[:, :, 2], 1), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        maxindex1 = 0
        maxindex2 = 0
        max1 = 0
        max2 = 0
        flag1 = 0
        flag2 = 0
        for j in range(len(contours)):
            if contours[j].shape[0] > max1:
                maxindex1 = j
                max1 = contours[j].shape[0]
            if j == len(contours) - 1:
                approxCurve = cv2.approxPolyDP(contours[maxindex1], 1, closed=True)
                if approxCurve.shape[0] > 5:
                    ellipse = cv2.fitEllipse(approxCurve)
                flag1 = 1
        for k in range(len(contours2)):
            if contours2[k].shape[0] > max2:
                maxindex2 = k
                max2 = contours2[k].shape[0]
            if k == len(contours2) - 1:
                approxCurve2 = cv2.approxPolyDP(contours2[maxindex2], 1, closed=True)
                if approxCurve2.shape[0] > 5:
                    ellipse2 = cv2.fitEllipse(approxCurve2)
                flag2 = 1
        if flag1 == 1 and flag2 == 1 and ellipse2 != None and ellipse != None:
            aop = drawline_AOD(ellipse2, ellipse)
        return aop

    def cal_score(self, result):
        m = len(result)
        dice_all_score = 0.
        dice_fh_score = 0.
        dice_ps_score = 0.
        aop_score = 0.
        hd_ps_score = 0.
        hd_all_score = 0.
        hd_fh_score = 0.
        asd_all_score = 0.
        asd_fh_score = 0.
        asd_ps_score = 0.
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(m):
            # dice
            dice_all_score += float(result[i].get("dice_all"))
            dice_ps_score += float(result[i].get("dice_ps"))
            dice_fh_score += float(result[i].get("dice_fh"))
            # asa
            asd_all_score += float(result[i].get("asd_all"))
            asd_fh_score += float(result[i].get("asd_fh"))
            asd_ps_score += float(result[i].get("asd_ps"))
            # hd
            hd_all_score += float(result[i].get("hd_all"))
            hd_ps_score += float(result[i].get("hd_ps"))
            hd_fh_score += float(result[i].get("hd_fh"))
            # aop
            aop_score += float(result[i].get("aop"))
            tn += int(result[i].get("TN"))
            tp += int(result[i].get("TP"))
            fp += int(result[i].get("FP"))
            fn += int(result[i].get("FN"))
        dice_score = (dice_all_score + dice_ps_score + dice_fh_score) / (3 * m)
        hd_score = (hd_all_score + hd_ps_score + hd_fh_score) / (3 * m)
        asd_score = (asd_all_score + asd_ps_score + asd_fh_score) / (3 * m)
        aop_score /= m

        score = 0.25 * round(dice_score, 8) + 0.125 * (1 - round(hd_score / 100.0, 8)) + 0.125 * (1 - round(
            asd_score / 100.0, 8)) + 0.5 * (1 - aop_score / 180.0)

        aggregates = dict()
        aggregates['aop'] = aop_score
        aggregates['dice_ps'] = dice_ps_score / m
        aggregates['dice_fh'] = dice_fh_score / m
        aggregates['dice_all'] = dice_all_score / m
        aggregates['hd_ps'] = hd_ps_score / m
        aggregates['hd_fh'] = hd_fh_score / m
        aggregates['hd_all'] = hd_all_score / m
        aggregates['asd_ps'] = asd_ps_score / m
        aggregates['asd_fh'] = asd_fh_score / m
        aggregates['asd_all'] = asd_all_score / m
        aggregates['TN'] = tn
        aggregates['TP'] = tp
        aggregates['FN'] = fn
        aggregates['FP'] = fp
        aggregates['precision'] = tp / (tp + fp)
        aggregates['recall'] = tp / (tp + fn)
        aggregates['f1-score'] = 2 * aggregates['precision'] * aggregates['recall'] / (
                aggregates['precision'] + aggregates['recall'])
        aggregates['specificity'] = tn / (fp + tn)
        aggregates['wuzhen'] = 1 - aggregates['specificity']
        aggregates['louzhen'] = 1 - aggregates['recall']

        return score, aggregates


def kfold_result(result_folder, save_path):
    fold_fix = "fold"

    results = [json.load(open(os.path.join(result_folder, f"{fold_fix}{i}", "result.txt"), "r")) for i in
               range(5)]
    aop_all = [result['aggregates']['aop'] for result in results]
    dice_all = [result['aggregates']['dice_all'] for result in results]
    hd_all = [result['aggregates']['hd_all'] for result in results]
    asd_all = [result['aggregates']['asd_all'] for result in results]

    dice_ps = [result['aggregates']['dice_ps'] for result in results]
    hd_ps = [result['aggregates']['hd_ps'] for result in results]
    asd_ps = [result['aggregates']['asd_ps'] for result in results]

    dice_fh = [result['aggregates']['dice_fh'] for result in results]
    hd_fh = [result['aggregates']['hd_fh'] for result in results]
    asd_fh = [result['aggregates']['asd_fh'] for result in results]
    tn = [result['aggregates']['TN'] for result in results]
    tp = [result['aggregates']['TP'] for result in results]
    fn = [result['aggregates']['FN'] for result in results]
    fp = [result['aggregates']['FP'] for result in results]
    precision = [result['aggregates']['precision'] for result in results]
    recall = [result['aggregates']['recall'] for result in results]
    f1_score = [result['aggregates']['f1-score'] for result in results]
    specificity = [result['aggregates']['specificity'] for result in results]
    wuzhen = [result['aggregates']['wuzhen'] for result in results]
    louzhen = [result['aggregates']['louzhen'] for result in results]

    rs = {}
    rs['aop'] = (np.mean(aop_all), np.std(aop_all))
    rs['dice_all'] = (np.mean(dice_all), np.std(dice_all))
    rs['hd_all'] = (np.mean(hd_all), np.std(hd_all))
    rs['asd_all'] = (np.mean(asd_all), np.std(asd_all))
    rs['dice_ps'] = (np.mean(dice_ps), np.std(dice_ps))
    rs['hd_ps'] = (np.mean(hd_ps), np.std(hd_ps))
    rs['asd_ps'] = (np.mean(asd_ps), np.std(asd_ps))
    rs['dice_fh'] = (np.mean(dice_fh), np.std(dice_fh))
    rs['hd_fh'] = (np.mean(hd_fh), np.std(hd_fh))
    rs['asd_fh'] = (np.mean(asd_fh), np.std(asd_fh))
    rs['tn'] = (np.mean(tn), np.std(tn))
    rs['tp'] = (np.mean(tp), np.std(tp))
    rs['fn'] = (np.mean(fn), np.std(fn))
    rs['fp'] = (np.mean(fp), np.std(fp))
    rs['precision'] = (np.mean(precision), np.std(precision))
    rs['recall'] = (np.mean(recall), np.std(recall))
    rs['f1_score'] = (np.mean(f1_score), np.std(f1_score))
    rs['specificity'] = (np.mean(specificity), np.std(specificity))
    rs['wuzhen'] = (np.mean(wuzhen), np.std(wuzhen))
    rs['louzhen'] = (np.mean(louzhen), np.std(louzhen))

    fp = open(os.path.join(save_path, f"kfold_result.json"), "w")
    json.dump(rs, fp)


if __name__ == "__main__":

    fix = "eval"
    os.makedirs(f"./output", exist_ok=True)
    os.makedirs(f"./output/{fix}", exist_ok=True)

    preds_dir = "./pred"
    truths_dir = "./gt"

    output_path = os.path.join(f"./output/{fix}", "result.json")

    Evaluation(preds_dir, truths_dir, output_path,fix).process()
