import os
import joblib
import numpy as np
import torch


def to_ava_labels(predictions):
    """Converts the predictions to the AVA labels."""
    # TODO: fix ava labels paths
    # root = "/private/home/jathushan/Tracking/PHALP_v3"
    label_map, _    = joblib.load("/private/home/jathushan/Tracking/PHALP_v3/_BACKUP/_DATA2/datasets/ava/ava_labels.pkl")
    class_map_id    = joblib.load("/private/home/jathushan/Tracking/PHALP_v3/_BACKUP/_DATA2/datasets/ava/ava_class_mappping.pkl")

    label_map       = {
        1: 'bend', 
        3: 'crouch', 
        4: 'dance', 
        5: 'fall down', 
        6: 'get up', 
        7: 'jump', 
        8: 'lie', 
        9: 'martial art', 
        10: 'run', 
        11: 'sit', 
        12: 'stand',
        13: 'swim', 
        14: 'walk', 
        15: 'answer phone', 
        17: 'carry (an O)', 
        20: 'climb', 
        22: 'close', 
        24: 'cut', 
        26: 'dress', 
        27: 'drink', 
        28: 'drive', 
        29: 'eat', 
        30: 'enter', 
        34: 'hit (an O)', 
        36: 'lift', 
        37: 'listen', 
        38: 'open', 
        41: 'play instrument', 
        43: 'point to (an O)', 
        45: 'pull (an O)', 
        46: 'push (an O)', 
        47: 'put down', 
        48: 'read', 
        49: 'ride', 
        51: 'sail', 
        52: 'shoot', 
        54: 'smoke', 
        56: 'take photo',
        57: 'text', 
        58: 'throw', 
        59: 'touch (an O)', 
        60: 'turn', 
        61: 'watch (an O)', 
        62: 'work on computer', 
        63: 'write', 
        64: 'fight (a P)', 
        65: 'give (an O) to (a P)', 
        66: 'grab (a P)', 
        67: 'clap', 
        68: 'hand shake', 
        69: 'wave', 
        70: 'hug (a P)', 
        72: 'kiss (a P)', 
        73: 'lift (a P)', 
        74: 'listen to (a P)', 
        76: 'push (a P)', 
        77: 'sing', 
        78: 'take (an O) from (a P)', 
        79: 'talk', 
        80: 'watch (a P)'
    }

    pred_label      = torch.sigmoid(predictions[0])
    _, order        = torch.topk(pred_label, k=3)
    
    top_labels = []
    top_probs  = []
    for i in order: 
        class_map_id_ = class_map_id[i.item()+1]
        label_ = label_map.get(class_map_id_, "n/a")
        top_labels.append(label_)
        top_probs.append(pred_label[i].item())
    
    label_str = ["{:0.0f}% : {}".format(np.round(p*100.0, 2), l) for l, p in zip(top_labels, top_probs)]

    top_labels_all = []
    top_probs_all  = []
    for i in range(len(pred_label)):
        class_map_id_ = class_map_id[i+1]
        label_ = label_map.get(class_map_id_, "n/a")
        top_labels_all.append(label_)
        top_probs_all.append(pred_label[i].item())

    return label_str, {"labels": top_labels_all, "probs": top_probs_all}